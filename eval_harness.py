"""
eval_harness.py

Runs the identical held-out test set through:
  1. Your fine-tuned local model (QLoRA adapter on the base model)
  2. GPT (OpenAI API)
  3. Claude (Anthropic API)
  4. Gemini (Google API)

...using the same prompt template for every model, then reports
accuracy (macro F1), latency and estimated cost per 1,000 classifications
in a single comparison table.

Run:
    python eval_harness.py --model local
    python eval_harness.py --model gpt
    python eval_harness.py --model claude
    python eval_harness.py --model gemini
    python eval_harness.py --model all       # runs everything, writes results/comparison.csv
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import load_from_disk
from dotenv import load_dotenv
from sklearn.metrics import f1_score, classification_report

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
SPLITS_DIR = Path(__file__).resolve().parent / "data" / "splits"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

LABELS = ["negative", "neutral", "positive"]

# Rough $ per 1M tokens (input/output). Update these before publishing —
# API pricing changes, and your model card should state the date you priced this.
PRICING_PER_1M_TOKENS = {
    "gpt": {"input": 0.40, "output": 1.60},
    "claude": {"input": 3.00, "output": 15.00},
    "gemini": {"input": 0.10, "output": 0.40},
}


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_label(raw_text: str) -> str:
    """Extracts one of the three labels from a free-text model response."""
    text = raw_text.strip().lower()
    for label in LABELS:
        if label in text:
            return label
    return "neutral"  # fallback if the model returns something unparseable


def run_local(cfg, test_ds):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    hf_cfg = cfg["huggingface"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model, hf_cfg["hub_model_id"], token=os.environ.get("HF_TOKEN")
    )
    model.eval()

    predictions, latencies = [], []
    for example in test_ds:
        inputs = tokenizer(example["prompt"], return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        latencies.append(time.perf_counter() - start)
        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions.append(parse_label(decoded))

    # Local inference cost is amortized GPU electricity, not per-call —
    # report it separately in the writeup rather than forcing it into
    # the same $/1000-calls column as the hosted APIs.
    return predictions, latencies, 0.0


def call_with_retry(call_fn, prompt, max_retries=6, base_delay=8):
    """Retries a model call with exponential backoff for transient errors
    (rate limits, timeouts, server errors). Does NOT retry 400-style
    'bad request' errors, since those mean the request itself is invalid
    and retrying identically will just fail the same way every time."""
    for attempt in range(max_retries):
        try:
            return call_fn(prompt)
        except Exception as e:
            error_str = str(e)
            is_bad_request = (
                "BadRequestError" in type(e).__name__
                or "400" in error_str[:20]
            )
            if is_bad_request:
                print(f"    non-retryable error: {type(e).__name__}: {error_str}")
                raise
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (attempt + 1)
            print(f"    retry {attempt + 1}/{max_retries} after {type(e).__name__}: {error_str}. Waiting {wait}s...")
            time.sleep(wait)


def run_api(provider: str, cfg, test_ds):
    label_names = LABELS
    predictions, latencies, total_input_tok, total_output_tok = [], [], 0, 0

    # Gemini's free tier is heavily rate-limited (as low as 5 requests/minute
    # on some models) — pace requests to stay under that instead of just
    # retrying after the fact.
    inter_call_delay = {"gpt": 0, "claude": 0, "gemini": 13}.get(provider, 0)

    if provider == "gpt":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_name = cfg["frontier_models"]["openai_model"]

        def call(prompt):
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=300,  # headroom for models that spend
                # part of the budget on internal reasoning before visible output
            )
            usage = resp.usage
            return resp.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens

    elif provider == "claude":
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        model_name = cfg["frontier_models"]["anthropic_model"]

        def call(prompt):
            resp = client.messages.create(
                model=model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens

    elif provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(cfg["frontier_models"]["google_model"])

        def call(prompt):
            resp = model.generate_content(prompt)
            usage = resp.usage_metadata
            return resp.text, usage.prompt_token_count, usage.candidates_token_count

    else:
        raise ValueError(f"Unknown provider: {provider}")

    for i, example in enumerate(test_ds):
        start = time.perf_counter()
        text, in_tok, out_tok = call_with_retry(call, example["prompt"])
        latencies.append(time.perf_counter() - start)

        if i < 3:
            print(f"    [debug] raw response {i}: {text!r}")

        predictions.append(parse_label(text or ""))
        total_input_tok += in_tok
        total_output_tok += out_tok

        if inter_call_delay:
            time.sleep(inter_call_delay)

        if (i + 1) % 20 == 0:
            print(f"    ...{i + 1}/{len(test_ds)} done")

    price = PRICING_PER_1M_TOKENS[provider]
    avg_cost_per_1000 = (
        (total_input_tok / len(test_ds) * price["input"] / 1_000_000)
        + (total_output_tok / len(test_ds) * price["output"] / 1_000_000)
    ) * 1000

    return predictions, latencies, avg_cost_per_1000


def evaluate(name, predictions, latencies, cost_per_1000, y_true):
    macro_f1 = f1_score(y_true, predictions, labels=LABELS, average="macro")
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)] * 1000

    print(f"\n=== {name} ===")
    print(classification_report(y_true, predictions, labels=LABELS, zero_division=0))

    return {
        "model": name,
        "macro_f1": round(macro_f1, 4),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "p95_latency_ms": round(p95_latency_ms, 1),
        "cost_per_1000_usd": round(cost_per_1000, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["local", "gpt", "claude", "gemini", "all"], default="all"
    )
    args = parser.parse_args()

    cfg = load_config()
    test_ds = load_from_disk(str(SPLITS_DIR))["test"]
    y_true = [LABELS[label] for label in test_ds["label"]]

    providers = ["local", "gpt", "claude", "gemini"] if args.model == "all" else [args.model]
    results = []

    for provider in providers:
        if provider == "local":
            preds, latencies, cost = run_local(cfg, test_ds)
        else:
            preds, latencies, cost = run_api(provider, cfg, test_ds)
        results.append(evaluate(provider, preds, latencies, cost, y_true))

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison table to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
