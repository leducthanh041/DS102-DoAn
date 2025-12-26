import os
import json
import random
import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# Cài đặt thư viện cần thiết cho tune tham số nếu chưa có
try:
    import optuna
except ImportError:
    print("Vui lòng cài đặt optuna: pip install optuna")
    # raise ImportError("Cần cài đặt thư viện optuna để chạy hyperparameter search.")

# =========================================================
# 1) CẤU HÌNH & UTILS
# =========================================================
set_seed(42)
MODEL_NAME = "vinai/phobert-base-v2"
OUTPUT_DIR = "./phobert-ft-tuned"
MAX_LENGTH = 256

# Đường dẫn file dữ liệu (Cập nhật đường dẫn thực tế của bạn)
TRAIN_FILE = "./preprocessed-dataset/train_processed.json"
DEV_FILE   = "./preprocessed-dataset/dev_processed.json"
TEST_FILE  = "./preprocessed-dataset/test_processed.json"

def load_json_lines(path: str) -> List[dict]:
    """Đọc file JSON lines (mỗi dòng 1 object)"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def normalize_label(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = str(s).strip().lower()
    mp = {
        "pos": "positive", "+": "positive", "positive": "positive",
        "neg": "negative", "-": "negative", "negative": "negative",
        "neu": "neutral", "0": "neutral", "neutral": "neutral"
    }
    return mp.get(s, s)

# =========================================================
# 2) DATASET
# =========================================================
class PhoBertDataset(Dataset):
    def __init__(self, rows, tokenizer, label2id, max_length):
        self.samples = []
        for r in rows:
            # Giả định dataset đã có cột 'review' và 'sentiment' chuẩn
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
        # Text đã được pre-segmented, nên không cần xử lý thêm
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

# =========================================================
# 3) METRICS & MODEL INIT
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

def model_init(trial=None):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={0: "positive", 1: "neutral", 2: "negative"}, # Cập nhật đúng map sau khi load data
        label2id={"positive": 0, "neutral": 1, "negative": 2}
    )

# =========================================================
# 4) CHUẨN BỊ DỮ LIỆU
# =========================================================
# Load dữ liệu
print("Loading data...")
train_rows = load_json_lines(TRAIN_FILE)
dev_rows   = load_json_lines(DEV_FILE)
test_rows  = load_json_lines(TEST_FILE)

# Xây dựng Label Map cố định để đảm bảo nhất quán
label_list = ["negative", "neutral", "positive"] # Sắp xếp theo thứ tự mong muốn
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}
print("Label Map:", label2id)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Tạo Dataset
train_ds = PhoBertDataset(train_rows, tokenizer, label2id, MAX_LENGTH)
dev_ds   = PhoBertDataset(dev_rows, tokenizer, label2id, MAX_LENGTH)
test_ds  = PhoBertDataset(test_rows, tokenizer, label2id, MAX_LENGTH)

print(f"Data Loaded: Train={len(train_ds)}, Dev={len(dev_ds)}, Test={len(test_ds)}")

# =========================================================
# 5) HYPERPARAMETER TUNING (RANDOM SEARCH)
# =========================================================
# Hàm định nghĩa lại model_init để truyền đúng config
def get_model():
    cfg = AutoConfig.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)

# Training Arguments cơ bản
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",  # Đánh giá trên tập Dev mỗi epoch
    save_strategy="epoch",
    learning_rate=2e-5,           # Giá trị mặc định, sẽ được tune
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,           # Số epoch tối đa cho mỗi lần thử
    weight_decay=0.01,
    load_best_model_at_end=True,  # Load model tốt nhất theo metric
    metric_for_best_model="f1_macro",
    save_total_limit=1,
    report_to="none",             # Tắt wandb/tensorboard cho gọn
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model_init=get_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,          # Dùng tập Dev để tune
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Định nghĩa không gian tìm kiếm (Search Space)
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }

print("\n[Start Tuning] Bắt đầu tìm kiếm tham số (Random Search)...")
# n_trials: Số lượng thử nghiệm ngẫu nhiên
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=5,  # Bạn có thể tăng lên 10-20 nếu có nhiều thời gian/GPU
)

print("\n[Best Run] Tham số tốt nhất:")
print(best_run.hyperparameters)

# =========================================================
# 6) TRAIN FINAL MODEL WITH BEST PARAMS
# =========================================================
print("\n[Final Training] Huấn luyện lại với tham số tốt nhất...")
# Áp dụng tham số tốt nhất
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

# Train lại
trainer.train()

# Lưu model cuối cùng
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Đã lưu model vào: {OUTPUT_DIR}")

# =========================================================
# 7) EVALUATE ON TEST SET & ERROR ANALYSIS
# =========================================================
print("\n[Evaluate on TEST] Đánh giá trên tập Test...")
predictions = trainer.predict(test_ds)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# 7.1 Metrics
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="macro")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1 Macro: {f1:.4f}")
print(classification_report(labels, preds, target_names=label_list))

# 7.2 Confusion Matrix
cm = confusion_matrix(labels, preds)
cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
print("\nConfusion Matrix:")
print(cm_df)
cm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))

# 7.3 Lưu các câu bị dự đoán sai (Misclassified)
print("\n[Analysis] Đang trích xuất các câu sai...")
errors = []
for idx, (p, l) in enumerate(zip(preds, labels)):
    if p != l:
        # Lấy lại text gốc từ dataset
        # Lưu ý: dataset trả về tensor, ta cần lấy từ list samples ban đầu
        sample = test_ds.samples[idx]
        errors.append({
            "text": sample["text"],
            "true_label": sample["label"],
            "pred_label": id2label[p]
        })

# Lưu xuống file CSV/JSON
error_df = pd.DataFrame(errors)
error_path = os.path.join(OUTPUT_DIR, "error_analysis_test.csv")
error_df.to_csv(error_path, index=False, encoding="utf-8-sig") # sig cho tiếng Việt excel

print(f"Đã lưu {len(errors)} câu dự đoán sai vào: {error_path}")
print("Hoàn tất quy trình!")