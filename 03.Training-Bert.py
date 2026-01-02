# -*- coding: utf-8 -*-

import os
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# =========================================================
# 1) Utilities: Load data & Preprocessing
# =========================================================

def load_json_data(path: str) -> List[dict]:
    """Hàm đọc file JSON (hỗ trợ cả Array và Lines)"""
    with open(path, "r", encoding="utf-8") as f:
        try:
            # Thử đọc dạng mảng JSON chuẩn
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            return data
        except json.JSONDecodeError:
            # Nếu lỗi, thử đọc dạng JSON Lines
            f.seek(0)
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows

def normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    label = str(label).strip().lower()
    # Mapping nhãn về chuẩn
    mapping = {
        "pos": "positive", "+": "positive", "positive": "positive",
        "neg": "negative", "-": "negative", "negative": "negative",
        "neu": "neutral", "0": "neutral", "neutral": "neutral"
    }
    return mapping.get(label, label)

def build_label_maps(labels: List[str]):
    unique_labels = sorted(set(labels))
    # Ưu tiên thứ tự này để confusion matrix đẹp
    preferred_order = ["positive", "neutral", "negative"]
    
    ordered_labels = (
        [l for l in preferred_order if l in unique_labels] + 
        [l for l in unique_labels if l not in preferred_order]
    )
    
    label2id = {label: idx for idx, label in enumerate(ordered_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label

# =========================================================
# 2) Torch Dataset
# =========================================================
class SentDataset(Dataset):
    def __init__(self, rows, tokenizer, label2id, max_length):
        self.samples = []
        
        # Tự động đoán tên trường text/label
        if not rows:
            return
            
        first_row = rows[0]
        text_keys = ["review_clean", "review", "text", "content"]
        label_keys = ["sentiment", "label", "y"]
        
        text_field = next((k for k in text_keys if k in first_row), None)
        label_field = next((k for k in label_keys if k in first_row), None)

        if not text_field or not label_field:
            raise ValueError(f"Không tìm thấy trường text/label. Keys: {first_row.keys()}")

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
        # Nếu nhãn trong data không có trong map (ví dụ data lỗi), bỏ qua hoặc handle
        if sample["label"] in self.label2id:
            item["labels"] = torch.tensor(self.label2id[sample["label"]], dtype=torch.long)
        return item

# =========================================================
# 3) Metrics
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# =========================================================
# 4) Confusion Matrix Utils
# =========================================================
def plot_and_save_cm(cm, labels, title, save_path):
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# =========================================================
# 5) Main
# =========================================================
def main():
    set_seed(42)
    
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    train_path = r"preprocessed-dataset\train_processed.json"
    dev_path   = r"preprocessed-dataset\dev_processed.json"
    test_path  = r"preprocessed-dataset\test_processed.json"
    
    output_dir = "./bert-finetuned-model"
    # Lưu Confusion Matrix vào thư mục riêng cho BERT
    cm_save_dir = "Confusion-Matrix/BERT" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1. Load Data
    print("Loading data...")
    train_rows = load_json_data(train_path)
    dev_rows   = load_json_data(dev_path)
    test_rows  = load_json_data(test_path)
    
    print(f"Train: {len(train_rows)} | Dev: {len(dev_rows)} | Test: {len(test_rows)}")

    # 2. Xây dựng Label Map từ tập Train
    # Giả định cấu trúc giống nhau, lấy mẫu row đầu để tìm key
    temp_dataset = SentDataset(train_rows[:5], None, None, None) # Chỉ để check key
    
    # Lấy toàn bộ nhãn từ train để build map
    all_train_labels = []
    # Lặp thủ công nhẹ để lấy label raw (vì class dataset ở trên cần tokenizer)
    label_key = "sentiment" # Mặc định theo data của bạn
    # Kiểm tra lại key thực tế
    if "sentiment" not in train_rows[0]:
         # Fallback tìm key
         keys = train_rows[0].keys()
         if "label" in keys: label_key = "label"
    
    for row in train_rows:
        val = normalize_label(row.get(label_key))
        if val: all_train_labels.append(val)

    label2id, id2label = build_label_maps(all_train_labels)
    print("Label Mapping:", label2id)

    # 3. Tokenizer & Model
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    ).to(device)

    # 4. Tạo Dataset
    max_len = 256
    train_ds = SentDataset(train_rows, tokenizer, label2id, max_len)
    dev_ds   = SentDataset(dev_rows, tokenizer, label2id, max_len)
    test_ds  = SentDataset(test_rows, tokenizer, label2id, max_len)

    # 5. Training Arguments (CÓ DÙNG DEV ĐỂ TUNE)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Hyperparameters
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5, # Chạy khoảng 5-10 epoch
        weight_decay=0.01,
        
        # Validation & Saving Strategy (Quan trọng cho việc Tune)
        evaluation_strategy="epoch",  # Đánh giá trên Dev sau mỗi epoch
        save_strategy="epoch",        # Lưu checkpoint sau mỗi epoch
        load_best_model_at_end=True,  # Load lại model có F1 tốt nhất trên Dev
        metric_for_best_model="f1_macro",
        
        logging_steps=50,
        seed=42,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,      # Đưa tập Dev vào đây
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 6. Train
    print("\n--- Bắt đầu Train & Tune trên tập Dev ---")
    trainer.train()

    # 7. Evaluate trên Test (Tập độc lập)
    print("\n--- Đánh giá trên tập Test ---")
    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    # In kết quả
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1-Macro: {f1:.4f}")
    
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # 8. Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    save_path = os.path.join(cm_save_dir, "confusion_matrix.png")
    plot_and_save_cm(cm, target_names, "Confusion Matrix - BERT", save_path)
    print(f"Đã lưu Confusion Matrix tại: {save_path}")

    # Lưu model cuối cùng (là model tốt nhất do load_best_model_at_end=True)
    trainer.save_model(output_dir)
    print("Hoàn tất.")

if __name__ == "__main__":
    main()