import os
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Optional
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# CONFIG
# =========================
MODEL_DIR  = "./phobert-ft-tuned"   # OUTPUT_DIR bạn đã save model tốt nhất
MODEL_NAME = "vinai/phobert-base-v2"
DEV_FILE   = "./preprocessed-dataset/dev_processed.json"
MAX_LENGTH = 256

# Label map PHẢI KHỚP với lúc train
label_list = ["negative", "neutral", "positive"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# =========================
# DATA LOADER
# =========================
def load_json_lines(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def normalize_label(s: Optional[str]) -> Optional[str]:
    if s is None: 
        return None
    s = str(s).strip().lower()
    mp = {
        "pos": "positive", "+": "positive", "positive": "positive",
        "neg": "negative", "-": "negative", "negative": "negative",
        "neu": "neutral", "0": "neutral", "neutral": "neutral"
    }
    return mp.get(s, s)

# =========================
# DATASET
# =========================
class PhoBertDataset(Dataset):
    def __init__(self, rows, tokenizer, label2id, max_length):
        self.samples = []
        for r in rows:
            text = r.get("review")
            lab  = normalize_label(r.get("sentiment"))
            if text and lab in label2id:
                self.samples.append({"text": text, "label": lab})

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.label2id[s["label"]], dtype=torch.long)
        return item

# =========================
# MAIN EVAL
# =========================
def main():
    # 1) Load dev
    dev_rows = load_json_lines(DEV_FILE)

    # 2) Load tokenizer + model từ MODEL_DIR (best model đã lưu)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # 3) Dataset dev
    dev_ds = PhoBertDataset(dev_rows, tokenizer, label2id, MAX_LENGTH)
    print(f"Dev samples: {len(dev_ds)}")

    # 4) Trainer chỉ để predict (không train)
    args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "eval_dev_tmp"),
        per_device_eval_batch_size=32,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_drop_last=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer
    )

    # 5) Predict on dev
    out = trainer.predict(dev_ds)
    preds = np.argmax(out.predictions, axis=1)
    labels = out.label_ids

    # 6) Metrics
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")

    print(f"\nDev Accuracy : {acc:.4f}")
    print(f"Dev F1-macro: {f1m:.4f}")

    print("\n=== Classification report (DEV) ===")
    print(classification_report(labels, preds, target_names=label_list, digits=4))

    # 7) Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
    print("\n=== Confusion matrix (DEV) ===")
    print(cm_df)

    # Save artifacts
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix_dev.csv")
    cm_df.to_csv(cm_path, encoding="utf-8")
    print("Saved:", cm_path)

    # 8) Error analysis on dev
    errors = []
    for idx, (p, l) in enumerate(zip(preds, labels)):
        if p != l:
            sample = dev_ds.samples[idx]
            errors.append({
                "text": sample["text"],
                "true_label": sample["label"],
                "pred_label": id2label[int(p)]
            })

    err_df = pd.DataFrame(errors)
    err_path = os.path.join(MODEL_DIR, "error_analysis_dev.csv")
    err_df.to_csv(err_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(errors)} errors to:", err_path)

    print("\n[Done] Evaluated best saved model on DEV.")

if __name__ == "__main__":
    main()
