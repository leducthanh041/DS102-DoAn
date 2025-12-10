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

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# =========================================================
# 1) WORD SEGMENTATION (PhoBERT khuyến nghị)
# =========================================================
_SEGMENTER_BACKEND = None
_SEGMENTER_OBJ = None

def _init_segmenter(prefer_py_vncorenlp: bool = True):
    """
    Ưu tiên py_vncorenlp, nếu không có thì dùng underthesea.
    """
    global _SEGMENTER_BACKEND, _SEGMENTER_OBJ
    if _SEGMENTER_BACKEND:
        return
    if prefer_py_vncorenlp:
        try:
            from py_vncorenlp import VnCoreNLP
            _SEGMENTER_OBJ = VnCoreNLP(annotators=["wseg"])
            _SEGMENTER_BACKEND = "py_vncorenlp"
            print("[Segmenter] Using py_vncorenlp")
            return
        except Exception:
            pass
    try:
        import underthesea
        _SEGMENTER_OBJ = underthesea
        _SEGMENTER_BACKEND = "underthesea"
        print("[Segmenter] Using underthesea")
    except Exception as e:
        raise RuntimeError("Không tìm thấy segmenter. Cài py_vncorenlp hoặc underthesea.", e)

def vn_word_segment(text: str) -> str:
    """
    Word segmentation cho câu tiếng Việt.
    Dataset của bạn đã tiền xử lý (lower, bỏ số/dấu/stopword) nên ở đây
    chỉ segment theo tool, không thêm xử lý nào khác.
    """
    if not text:
        return ""
    if _SEGMENTER_BACKEND is None:
        _init_segmenter(prefer_py_vncorenlp=True)
    if _SEGMENTER_BACKEND == "py_vncorenlp":
        return _SEGMENTER_OBJ.word_segment(text)
    tokens = _SEGMENTER_OBJ.word_tokenize(text)
    return tokens if isinstance(tokens, str) else " ".join(tokens)


# =========================================================
# 2) LOAD DATA + LABEL MAP
# =========================================================
def load_json_array(path: str) -> List[dict]:
    """
    Đọc JSON dạng:
    - 1 mảng lớn [ {...}, {...}, ... ]
    - hoặc JSON lines (mỗi dòng 1 object)
    """
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
    t = next((k for k in text_keys if k in sample), None)
    y = next((k for k in label_keys if k in sample), None)
    if not t:
        raise ValueError(f"Không tìm thấy cột text trong keys={sample.keys()}")
    if not y:
        raise ValueError(f"Không tìm thấy cột nhãn trong keys={sample.keys()}")
    return t, y

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

def build_label_maps(labels: List[str]):
    uniq = sorted(set(labels))
    # Ưu tiên thứ tự này nếu có
    order = ["positive", "neutral", "negative"]
    ordered = [c for c in order if c in uniq] + [c for c in uniq if c not in order]
    label2id = {c: i for i, c in enumerate(ordered)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label


# =========================================================
# 3) TORCH DATASET (KHÔNG preprocess thêm, chỉ segment)
# =========================================================
class SentDataset(Dataset):
    def __init__(self, rows, text_field, label_field, tokenizer, label2id, max_length):
        self.samples = []
        for r in rows:
            text = r.get(text_field)
            lab  = normalize_label(r.get(label_field))
            # test có label → dùng được cho đánh giá
            if text is not None and (lab is not None or label_field not in r):
                self.samples.append({"text": text, "label": lab})
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # KHÔNG gọi preprocess_text_vi nữa, text đã preprocessed
        clean = s["text"]
        seg   = vn_word_segment(clean)
        enc = self.tokenizer(
            seg,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        x = {k: v.squeeze(0) for k, v in enc.items()}
        if s["label"] is not None:
            x["labels"] = torch.tensor(self.label2id[s["label"]], dtype=torch.long)
        return x


# =========================================================
# 4) METRICS
# =========================================================
@dataclass
class MetricsCfg:
    id2label: Dict[int, str]

def compute_metrics_builder(cfg: MetricsCfg):
    def _fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")
        }
    return _fn


# =========================================================
# 5) MAIN TRAIN + TEST (NO EVAL SET)
# =========================================================
set_seed(42)

train_json = "./train-preprocessed.json"
test_json  = "./test-preprocessed.json"
model_name = "vinai/phobert-base-v2"
output_dir = "./phobert-ft-final"

# 5.1. Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 5.2. Load dữ liệu train/test
train_rows = load_json_array(train_json)
test_rows  = load_json_array(test_json)
if not train_rows or not test_rows:
    raise RuntimeError("Thiếu dữ liệu train hoặc test.")

# 5.3. Xác định cột text / label (hoặc tự set)
t_field, y_field = guess_fields(train_rows[0])   # sẽ ra "review", "sentiment"
print(f"[Fields] text={t_field} | label={y_field}")

train_labels = [normalize_label(r.get(y_field)) for r in train_rows if r.get(y_field)]
label2id, id2label = build_label_maps(train_labels)
print("[Labels]", label2id)

# 5.4. Load PhoBERT + tokenizer
cfg = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg).to(device)

# 5.5. In tổng số tham số của mô hình
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 5.6. Dataset train/test
max_length = 256

train_ds = SentDataset(
    train_rows, t_field, y_field,
    tokenizer, label2id, max_length
)
test_ds = SentDataset(
    test_rows, t_field, y_field,
    tokenizer, label2id, max_length
)

print("Train samples:", len(train_ds))
print("Test samples :", len(test_ds))

# 5.7. TrainingArguments – KHÔNG eval giữa chừng, chỉ train
amp = {}
if torch.cuda.is_available():
    # Chạy fp16 cho GPU, nếu bạn muốn bf16 thì đổi ở đây (tùy GPU)
    amp["fp16"] = True

train_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,

    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,          # bạn có thể tăng/giảm (30 như code gốc nếu muốn)
    weight_decay=0.01,
    warmup_ratio=0.1,

    evaluation_strategy="no",     # KHÔNG dùng eval set
    save_strategy="epoch",        # vẫn lưu mỗi epoch (tùy bạn)
    save_total_limit=1,

    logging_steps=50,
    seed=42,
    report_to="none",
    **amp
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    # KHÔNG truyền eval_dataset
    compute_metrics=compute_metrics_builder(MetricsCfg(id2label=id2label)),
    tokenizer=tokenizer,
)

# 5.8. Train
trainer.train()

# 5.9. Đánh giá trên TEST
print("\n[Evaluate on TEST]")
test_output = trainer.predict(test_ds)

y_true = test_output.label_ids
y_pred = np.argmax(test_output.predictions, axis=1)

acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
print(f"Test Accuracy : {acc:.4f}")
print(f"Test F1-macro: {f1_macro:.4f}")

print("\n=== Classification report (TEST) ===")
print(classification_report(
    y_true, y_pred,
    target_names=[id2label[i] for i in sorted(id2label.keys())],
    digits=4
))

print("\n=== Confusion matrix (TEST) ===")
print(confusion_matrix(y_true, y_pred))

# 5.10. Lưu model + tokenizer + label maps
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
with open(os.path.join(output_dir, "label_maps.json"), "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print("\n[Done] Model saved to:", output_dir)