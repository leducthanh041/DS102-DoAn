# -*- coding: utf-8 -*-

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)



import matplotlib.pyplot as plt

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# =========================================================
# 1) Utilities: load JSON + label handling
# =========================================================
def load_json_array(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            return data
        except json.JSONDecodeError:
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows

def guess_fields(sample: dict):
    text_keys = ["review", "text", "content", "body", "sentence", "title"]
    label_keys = ["sentiment", "label", "y", "tag"]

    text_field = next((k for k in text_keys if k in sample), None)
    label_field = next((k for k in label_keys if k in sample), None)

    if text_field is None:
        raise ValueError(f"Text field not found in keys: {sample.keys()}")
    if label_field is None:
        raise ValueError(f"Label field not found in keys: {sample.keys()}")

    return text_field, label_field

def normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None

    label = str(label).strip().lower()
    mapping = {
        "pos": "positive", "+": "positive", "positive": "positive",
        "neg": "negative", "-": "negative", "negative": "negative",
        "neu": "neutral", "0": "neutral", "neutral": "neutral"
    }
    return mapping.get(label, label)

def build_label_maps(labels: List[str]):
    unique_labels = sorted(set(labels))
    preferred_order = ["positive", "neutral", "negative"]

    ordered_labels = (
        [l for l in preferred_order if l in unique_labels] +
        [l for l in unique_labels if l not in preferred_order]
    )

    label2id = {label: idx for idx, label in enumerate(ordered_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label

# =========================================================
# 2) Torch Dataset (BERT: no word segmentation)
# =========================================================
class SentDataset(Dataset):
    def __init__(self, rows, text_field, label_field, tokenizer, label2id, max_length):
        self.samples = []

        for row in rows:
            text = row.get(text_field)
            label = normalize_label(row.get(label_field))

            if text is not None and label is not None:
                self.samples.append({
                    "text": str(text),
                    "label": label
                })

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(
            self.label2id[sample["label"]],
            dtype=torch.long
        )
        return item

# =========================================================
# 3) Metrics
# =========================================================
@dataclass
class MetricsCfg:
    id2label: Dict[int, str]

def compute_metrics_builder(cfg: MetricsCfg):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")
        }

    return compute_metrics

# =========================================================
# 4) Confusion Matrix Plotting
# =========================================================
def plot_and_save_cm(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    save_path: str,
    normalize: bool = False
):
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, value, ha="center", va="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

# =========================================================
# 5) Main
# =========================================================
def main():
    set_seed(42)

    train_json = "./train-preprocessed.json"
    test_json = "./test-preprocessed.json"

    model_name = "bert-base-multilingual-cased"
    output_dir = "./bert-ft-final"

    cm_save_dir = "/datastore/uittogether2/LuuTru/Thanhhn/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_rows = load_json_array(train_json)
    test_rows = load_json_array(test_json)

    text_field, label_field = guess_fields(train_rows[0])

    train_labels = [
        normalize_label(row.get(label_field))
        for row in train_rows
        if row.get(label_field) is not None
    ]

    label2id, id2label = build_label_maps(train_labels)
    print("Label mapping:", label2id)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", f"{total_params:,}")

    max_length = 256
    train_ds = SentDataset(train_rows, text_field, label_field, tokenizer, label2id, max_length)
    test_ds = SentDataset(test_rows, text_field, label_field, tokenizer, label2id, max_length)

    amp = {"fp16": True} if torch.cuda.is_available() else {}

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        seed=42,
        report_to="none",
        **amp
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_builder(MetricsCfg(id2label))
    )

    trainer.train()

    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    print("Test Accuracy:", accuracy_score(y_true, y_pred))
    print("Test F1-macro:", f1_score(y_true, y_pred, average="macro"))

    labels_sorted = [id2label[i] for i in sorted(id2label.keys())]

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels_sorted, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    plot_and_save_cm(
        cm,
        labels_sorted,
        "Confusion Matrix - BERT (Test Set)",
        os.path.join(cm_save_dir, "confusion_matrix_bert.png"),
        normalize=False
    )

    plot_and_save_cm(
        cm,
        labels_sorted,
        "Normalized Confusion Matrix - BERT (Test Set)",
        os.path.join(cm_save_dir, "confusion_matrix_bert_normalized.png"),
        normalize=True
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": label2id, "id2label": id2label},
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Training and evaluation completed successfully.")

if __name__ == "__main__":
    main()
