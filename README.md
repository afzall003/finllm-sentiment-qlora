# finllm-sentiment-qlora

**A QLoRA-fine-tuned 7B model that matched or beat GPT-6 and Claude Haiku 4.5 on financial sentiment classification — trained entirely on a single consumer laptop GPU, at zero marginal inference cost.**

## What this project answers

When does fine-tuning a small open model actually beat calling a frontier API for a narrow task — and by how much, on accuracy, latency, and cost? Most people either assume fine-tuning is always cheaper, or never bother benchmarking it against a general-purpose API at all. This project builds a defensible, reproducible answer using real numbers rather than intuition.

## What this project achieved

- **Fine-tuned Qwen2.5-7B-Instruct end-to-end on a single RTX 4060 (8GB VRAM)** using QLoRA — no cloud compute, no rented GPU, just a laptop.
- **Beat both frontier APIs it was benchmarked against**, on the same held-out test set, using the identical prompt for every model:

  | Model | Macro F1 | Avg Latency | Cost / 1,000 calls |
  |---|---|---|---|
  | **This model** | **0.9722** | **349.3 ms** | **$0.00*** |
  | Claude Haiku 4.5 | 0.9252 | 800.1 ms | $0.2704 |
  | GPT-6 Luna | 0.9248 | 1,031.1 ms | $0.068 |

  \* Amortized local GPU cost, not a hosted API charge — not directly comparable, but effectively negligible at this scale.

- **Answered the question above concretely:** for a narrow, well-defined task, a fine-tuned small model can match frontier general-purpose APIs on accuracy while being 2–3x faster and free to run — at the cost of losing general-purpose flexibility.
- **Published the trained adapter publicly** on [Hugging Face](https://huggingface.co/afzal2003/finllm-sentiment-qlora), with the full comparison table and methodology in the model card.
- **Built a reusable, provider-agnostic evaluation harness** that runs identical prompts against a local model and multiple hosted APIs and reports accuracy, latency, and cost side by side — reusable for any future model comparison, not just this one.

Full methodology, limitations, and usage instructions for the model itself are on the [Hugging Face model card](https://huggingface.co/afzal2003/finllm-sentiment-qlora). This README covers reproducing the training and evaluation pipeline.

## Task and data

3-class sentiment classification (negative / neutral / positive) on [FinancialPhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank), all-agreement subset (2,264 examples where all annotators agreed on the label). Split 80/10/10 into train/validation/test with a fixed seed.

## Minimum hardware requirements

**An NVIDIA GPU with CUDA support is required for training and for the quantized local-inference path.** `bitsandbytes` (the 4-bit quantization library this project relies on) has no functional Apple Silicon or CPU-only support — macOS and CPU-only machines cannot run `train_qlora.py` or the local-model portion of `eval_harness.py` as written.

- **GPU VRAM:** 8GB minimum (this project was built and tested on exactly that — an RTX 4060 Laptop GPU). Training uses 4-bit quantization, batch size 1 with gradient accumulation, and gradient checkpointing specifically to fit this budget. Less than 8GB will likely need `max_seq_length` or LoRA rank reduced further in `config.yaml`.
- **System RAM:** 16GB+ recommended (32GB is comfortable) — the base model, tokenizer, and dataset all load into system memory alongside GPU operations.
- **Disk space:** ~20GB free — the base Qwen2.5-7B-Instruct download alone is ~15GB, plus checkpoints saved during training (one per epoch by default).
- **OS:** Windows via WSL2 + Ubuntu, or native Linux. `bitsandbytes` is markedly less reliable on native Windows CUDA, which is why this project used WSL2.
- **Without a CUDA GPU:** the GPT/Claude portions of `eval_harness.py` still work anywhere, since they're just API calls with no GPU dependency — only the local-model training and inference paths are GPU-gated.

## Repository contents

finllm-sentiment-qlora/
├── config.yaml # single source of truth for model, LoRA, training, and eval settings
├── requirements.txt # pinned dependency versions (see "Environment notes" — these matter)
├── data/
│ └── prepare_data.py # downloads FinancialPhraseBank, builds the fixed train/val/test split
├── train_qlora.py # QLoRA fine-tuning script, sized for 8GB VRAM
├── eval_harness.py # runs the local model + GPT/Claude on the identical test set
└── results/
└── comparison.csv # the final results shown above


## Reproducing this

### 1. Environment

```bash
wsl --install -d Ubuntu-22.04
```

Inside WSL:

```bash
conda create -n finllm python=3.11
conda activate finllm
pip install -r requirements.txt
```

### 2. Environment notes — read before you hit the same walls I did

The dependency versions in `requirements.txt` are **pinned deliberately, not casually** — this stack was a moving target during development:

- `transformers`, `trl`, `peft`, and `accelerate` all have interdependent minimum-version requirements that aren't obvious from any single package's docs. Installing `trl` with an unpinned `>=` requirement pulled in a version whose `SFTTrainer` API had changed enough to break the training script outright.
- `bitsandbytes` needs to match your installed CUDA version specifically — pinning it to an old version to satisfy `transformers`/`trl` compatibility caused it to silently fall back to a CPU-only build with no compiled CUDA kernel, which fails at model-load time, not install time.
- If you hit `AttributeError: 'AdamW' object has no attribute 'train'` or `Could not find the bitsandbytes CUDA binary`, it's a version mismatch in this exact chain — check `transformers`, `trl`, `accelerate`, and `bitsandbytes` versions against each other, not just against your CUDA version in isolation.

### 3. Data preparation

```bash
python data/prepare_data.py
```

Note: this pulls from a plain-Parquet community mirror of FinancialPhraseBank rather than the original `takala/financial_phrasebank` repo. That original repo is script-based (a format recent versions of the `datasets` library have dropped support for), and its Hugging Face-hosted auto-conversion to Parquet was intermittently failing (500 errors) during development. If the mirror used here ever goes stale, the fix is the same: find any Parquet-format re-upload of the dataset and point `PARQUET_URL` in `prepare_data.py` at it.

### 4. Training

```bash
python train_qlora.py
```

Took roughly 2.5 hours for 3 epochs on an RTX 4060. Training loss dropped from 3.32 to ~1.0; validation loss tracked closely throughout with no overfitting. The script pushes the resulting adapter to Hugging Face Hub automatically if `HF_TOKEN` is set in a `.env` file.

### 5. Evaluation

```bash
python eval_harness.py --model local
python eval_harness.py --model gpt
python eval_harness.py --model claude
```

Requires `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in `.env`. Each run overwrites `results/comparison.csv` with just that model's row — merge manually if running them separately, as done here.

**A note on frontier API stability:** model names, parameter names, and pricing tiers for GPT and Claude all changed at least once during the few weeks this project was built. `max_tokens` was deprecated in favor of `max_completion_tokens` for newer OpenAI models mid-project, and `temperature` overrides were restricted entirely on some models. If a model call fails with a `BadRequestError` or `404`, check the provider's current model list and API reference before assuming the code is at fault — it may simply be pointing at a deprecated model name or parameter.

**On Gemini:** it was excluded from the final comparison. Google's free tier enforces a very low daily request cap (as low as 20 requests/day on some models) that made a 227-example test set impractical without billing enabled, and billing setup was blocked by an account-verification failure unrelated to this code. This is documented as a limitation, not silently omitted.

## License

Apache 2.0.

## Related

- [Trained model + full model card](https://huggingface.co/afzal2003/finllm-sentiment-qlora)
- [FinancialPhraseBank citation](https://huggingface.co/datasets/takala/financial_phrasebank)
