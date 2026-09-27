"""
train_qlora.py

QLoRA fine-tunes a base model (default: Qwen2.5-7B-Instruct) on
FinancialPhraseBank sentiment classification. Sized to fit an 8GB VRAM
card (RTX 4060): 4-bit quantization, small batch size with gradient
accumulation, gradient checkpointing, and a paged 8-bit optimizer.

Run:
    python data/prepare_data.py     # once, to build the split
    python train_qlora.py
"""

import os
import yaml
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
SPLITS_DIR = Path(__file__).resolve().parent / "data" / "splits"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def format_example(example):
    """Turns a (prompt, label) pair into the single text string SFTTrainer trains on."""
    label_names = ["negative", "neutral", "positive"]
    label_text = label_names[example["label"]]
    example["text"] = f"{example['prompt']} {label_text}"
    return example


def main():
    cfg = load_config()
    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    hf_cfg = cfg["huggingface"]

    print(f"Loading base model: {model_cfg['base_model']}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # sanity check: should be a small % of total params

    print(f"Loading dataset splits from {SPLITS_DIR}")
    splits = load_from_disk(str(SPLITS_DIR))
    train_ds = splits["train"].map(format_example)
    val_ds = splits["validation"].map(format_example)

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        report_to=train_cfg["report_to"],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    final_dir = Path(train_cfg["output_dir"]) / "final_adapter"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final LoRA adapter to {final_dir}")

    if hf_cfg["push_to_hub"]:
        print(f"Pushing adapter to Hugging Face Hub: {hf_cfg['hub_model_id']}")
        trainer.model.push_to_hub(hf_cfg["hub_model_id"], token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(hf_cfg["hub_model_id"], token=os.environ.get("HF_TOKEN"))


if __name__ == "__main__":
    main()
