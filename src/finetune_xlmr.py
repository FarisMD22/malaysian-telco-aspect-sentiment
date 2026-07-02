"""
XLM-RoBERTa fine-tuning on Malaysian telco sentiment data.

Owner: Faris
Input:  data/labeled/labeled_main_train.csv   (the 960-row TRAIN split only)
Output: models/xlmr_final/   (config + weights + tokenizer)
Run:    python src/finetune_xlmr.py            (full run; use a GPU)
        python src/finetune_xlmr.py --smoke     (1-step CPU sanity check)

WHY train on labeled_main_train.csv (not labeled_main.csv):
    `src/evaluate.py` scores the in-domain tier on `labeled_main_test.csv` (240 rows).
    That file is the disjoint held-out complement of labeled_main_train.csv (0 id overlap,
    verified). Fine-tuning on the FULL labeled_main.csv would let the model see those 240 test
    rows -> data leakage -> an inflated, non-comparable in-domain number. We therefore train on
    the 960-row train split ONLY, and carve a small validation slice FROM the train split for
    early stopping / best-checkpoint selection, so the test set stays genuinely unseen until
    evaluate.py runs. This mirrors the baseline (baseline.py trains on an 80% split of the same
    data) so the XLM-R vs LogReg comparison is fair.

Colab tip: free-tier GPU is enough. Upload data/labeled/labeled_main_train.csv, then
    pip install -U "transformers>=4.40" datasets accelerate
    python src/finetune_xlmr.py
Download the resulting models/xlmr_final/ and run src/evaluate.py locally for the 3-tier table.

This script is written to run on both transformers 4.x and 5.x (the Trainer `tokenizer` ->
`processing_class` and `evaluation_strategy` -> `eval_strategy` renames are handled at runtime).
"""
import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset

MODEL_NAME = "xlm-roberta-base"
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
MAX_LEN = 128
SEED = 42
VAL_SIZE = 0.15  # carved FROM the train split (not the held-out test set)

TRAIN_PATH = "data/labeled/labeled_main_train.csv"
OUT_DIR = "models/xlmr_final"
CKPT_DIR = "models/xlmr_ckpt"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def build_training_args(epochs: int, use_fp16: bool, smoke: bool) -> TrainingArguments:
    """Construct TrainingArguments, tolerating the 4.x/5.x eval-strategy kwarg rename."""
    common = dict(
        output_dir=CKPT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        seed=SEED,
        fp16=use_fp16,
        report_to="none",
    )
    if smoke:
        common.update(max_steps=1, num_train_epochs=1, save_total_limit=1)
    # The eval-each-epoch kwarg was renamed evaluation_strategy -> eval_strategy in transformers 4.46.
    arg_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in arg_names:
        common["eval_strategy"] = "epoch"
    else:
        common["evaluation_strategy"] = "epoch"
    return TrainingArguments(**common)


def make_trainer(model, args, train_ds, val_ds, tokenizer) -> Trainer:
    """Pass the tokenizer via processing_class (5.x) or tokenizer (4.x), whichever exists."""
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer_params = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN_PATH, help="path to the TRAIN split CSV")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 1-step run on a handful of rows to sanity-check the pipeline")
    a = ap.parse_args()

    set_seed(SEED)

    df = pd.read_csv(a.train)
    for col in ("cleaned_text", "sentiment_label"):
        if col not in df.columns:
            raise SystemExit(f"{a.train} is missing required column '{col}'")
    df = df[["cleaned_text", "sentiment_label"]].copy()
    df["cleaned_text"] = df["cleaned_text"].fillna("")
    df["label"] = df["sentiment_label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if a.smoke:
        df = df.groupby("label", group_keys=False).head(4).reset_index(drop=True)

    # Validation slice carved FROM the train split -> the test set stays untouched.
    # (smoke mode has only a handful of rows, so use an integer val size of one-per-class.)
    vsize = 3 if a.smoke else VAL_SIZE
    tr_df, val_df = train_test_split(
        df, test_size=vsize, stratify=df["label"], random_state=SEED,
    )
    print(f"train rows: {len(tr_df)} | val rows: {len(val_df)} "
          f"(held-out test stays in labeled_main_test.csv, scored later by src/evaluate.py)")
    print(f"train balance: {tr_df['sentiment_label'].value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["cleaned_text"], truncation=True,
                         padding="max_length", max_length=MAX_LEN)

    cols = ["cleaned_text", "label"]
    train_ds = Dataset.from_pandas(tr_df[cols], preserve_index=False).map(tok, batched=True)
    val_ds = Dataset.from_pandas(val_df[cols], preserve_index=False).map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID_TO_LABEL, label2id=LABEL_MAP,
    )

    use_fp16 = torch.cuda.is_available() and not a.smoke
    args = build_training_args(a.epochs, use_fp16, a.smoke)
    trainer = make_trainer(model, args, train_ds, val_ds, tokenizer)

    trainer.train()
    print("Validation metrics (on the train-carved val slice):", trainer.evaluate())

    Path("models").mkdir(exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved fine-tuned model to {OUT_DIR}")
    print("Next: run `python src/evaluate.py` for the real 3-tier table "
          "(in-domain scored on the untouched labeled_main_test.csv).")


if __name__ == "__main__":
    main()
