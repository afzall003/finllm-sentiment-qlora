"""
prepare_data.py

Loads FinancialPhraseBank (all-agreement subset), builds a fixed
train/val/test split, and saves it to disk. Both train_qlora.py and
eval_harness.py read from this saved split, so the test set is identical
(and never seen during training) no matter how many times you re-run
either script.

We pull from a plain-Parquet community mirror (szlazakm/SentimentAnalysis)
rather than the original takala/financial_phrasebank repo, since that
repo's script-based format and its HF-hosted auto-conversion pipeline
have both been failing as of testing this (September 2026).
"""

import io
import yaml
from pathlib import Path

import requests
import pandas as pd
from datasets import Dataset, DatasetDict

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"

PARQUET_URL = (
    "https://huggingface.co/datasets/szlazakm/SentimentAnalysis/resolve/main/"
    "FinancialPhraseBank/financial_phrasebank_AllAgree.parquet"
)

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

PROMPT_TEMPLATE = (
    "Classify the sentiment of the following financial statement as "
    "exactly one of: negative, neutral, positive.\n\n"
    "Statement: {sentence}\n\n"
    "Sentiment:"
)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_prompt(example):
    example["prompt"] = PROMPT_TEMPLATE.format(sentence=example["sentence"])
    return example


def normalize_label(value):
    """Handles either string labels ('positive') or integer labels (0/1/2)."""
    if isinstance(value, str):
        return LABEL_TO_ID[value.strip().lower()]
    return int(value)


def fetch_parquet_dataset() -> Dataset:
    print(f"Downloading: {PARQUET_URL}")
    resp = requests.get(PARQUET_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))

    print(f"Columns found: {list(df.columns)}")
    print(df.head(2))

    sentence_col = "sentence" if "sentence" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]

    df = df.rename(columns={sentence_col: "sentence", label_col: "label"})[
        ["sentence", "label"]
    ]
    df["label"] = df["label"].apply(normalize_label)

    return Dataset.from_pandas(df, preserve_index=False)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]

    raw = fetch_parquet_dataset()
    raw = raw.map(build_prompt)

    shuffled = raw.shuffle(seed=data_cfg["seed"])
    n = len(shuffled)
    n_train = int(n * data_cfg["train_split"])
    n_val = int(n * data_cfg["val_split"])

    splits = DatasetDict(
        {
            "train": shuffled.select(range(0, n_train)),
            "validation": shuffled.select(range(n_train, n_train + n_val)),
            "test": shuffled.select(range(n_train + n_val, n)),
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(OUTPUT_DIR))

    print(f"Saved splits to {OUTPUT_DIR}")
    for name, split in splits.items():
        print(f"  {name}: {len(split)} examples")


if __name__ == "__main__":
    main()
