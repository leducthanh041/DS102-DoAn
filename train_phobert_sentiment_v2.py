# -*- coding: utf-8 -*-
import os, re, json, unicodedata, argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# =========================================================
# 0) PREPROCESS: Chuẩn hoá văn bản tiếng Việt (bản mở rộng)
# =========================================================
import re, unicodedata

RE_CTRL_ZW    = re.compile(r"[\u200B-\u200D\uFEFF]")
RE_SPACES     = re.compile(r"\s+")
RE_URL        = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
RE_EMAIL      = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_MENTION    = re.compile(r"(?<!\w)@[\w_]+")
RE_DIGIT      = re.compile(r"\d")
RE_PUNCT_RUN  = re.compile(r"([!?]){2,}")     # !!!, ??? -> ! / ?
RE_ELLIPSIS   = re.compile(r"\.{3,}")         # ... -> "..."
RE_QUOTES_SP  = re.compile(r"\s*([\"'])\s*")

# Gạch ngang nằm giữa hai ký tự chữ/số -> coi như ngăn cách từ
RE_INNER_DASH = re.compile(r"(?<=\w)[\-\u2013\u2014\u2212](?=\w)")

# Ký tự không mong muốn (giữ chữ/số/khoảng trắng và một số dấu câu cơ bản + bộ tiếng Việt)
RE_BAD_SYM = re.compile(
    r"[^\w\s\.\,\!\?\:\;\-\(\)\"\'/àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
    r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄ"
    r"ÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮ"
    r"ỲÝỴỶỸĐ]"
)

# Bảng thay thế dấu “kiểu Word” -> ASCII an toàn
TRANS_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2026": "..."
})

def normalize_numbers_and_dates(text: str) -> str:
    # thay mọi chữ số -> '0' (giữ cấu trúc, tránh token lạ)
    return RE_DIGIT.sub("0", text)

def _normalize_colon_semicolon_comma_and_dots(s: str) -> str:
    """
    Chuẩn hoá dấu : ; , . và khoảng trắng xung quanh
    - Bảo toàn số dạng 10:30, 3.14, 1,234 trước khi format khoảng trắng
    - Loại bỏ lặp dấu ::: ;;; ,,, .... (.... -> ...)
    - Xoá space TRƯỚC dấu, đảm bảo 1 space SAU dấu (với :, ;, ,) khi cần
    """
    # 0) Bảo toàn các mẫu số: 10 : 30 -> 10:30 ; 3 . 14 -> 3.14 ; 1 , 234 -> 1,234
    s = re.sub(r"(\d)\s*:\s*(\d)", r"\1:\2", s)  # time
    s = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s) # decimal
    s = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", s)  # 1,234

    # 1) Rút gọn chuỗi dấu lặp
    s = re.sub(r":{2,}", ":", s)
    s = re.sub(r";{2,}", ";", s)
    # dấu chấm: để RE_ELLIPSIS xử lý thành "..." trước đó, còn lại ".." -> "."
    s = re.sub(r"(?<!\.)\.\.(?!\.)", ".", s)
    # dấu phẩy lặp
    s = re.sub(r",\s*,+", ",", s)

    # 2) Xoá khoảng trắng THỪA trước dấu câu
    s = re.sub(r"\s+([,;:\.\?\!])", r"\1", s)

    # 3) Đảm bảo 1 space SAU :, ;, , (trừ khi hết dòng)
    s = re.sub(r"([,;:])(?!\s|$)", r"\1 ", s)

    # 4) Không ép space sau '.' để tránh phá viết tắt, chỉ xoá space thừa trước '.'
    # (đã làm ở bước 2). Nếu muốn, có thể bật rule: thêm 1 space sau '.' khi không phải số.
    # Ví dụ (tuỳ chọn, bỏ comment nếu muốn):
    # s = re.sub(r"(?<!\d)\.(?!\d|\s|$)", ". ", s)

    # 5) Chuẩn hoá khoảng trắng quanh ngoặc
    s = re.sub(r"\(\s+", "(", s)   # "(" + space -> "("
    s = re.sub(r"\s+\)", ")", s)   # space + ")" -> ")"

    return s

def preprocess_text_vi(s: str,
                       remove_urls=True,
                       remove_emails=True,
                       mask_mentions=True,
                       normalize_numbers=True,
                       drop_bad_symbols=True,
                       keep_exclam_question=True,
                       strip_dash=False) -> str:
    if s is None:
        return ""
    # 1) Unicode + dọn ký tự vô hình + thay dấu “kiểu Word”
    s = unicodedata.normalize("NFKC", s)
    s = RE_CTRL_ZW.sub("", s).translate(TRANS_TABLE)

    # 2) URL/email/mention
    if remove_urls:   s = RE_URL.sub(" ", s)
    if remove_emails: s = RE_EMAIL.sub(" ", s)
    if mask_mentions: s = RE_MENTION.sub("@user", s)

    # 3) Rút gọn chấm lửng & chuỗi !/? lặp
    s = RE_ELLIPSIS.sub("...", s)           # mọi "...", "....." -> "..."
    s = RE_PUNCT_RUN.sub(lambda m: m.group(1), s)  # "!!!"->"!", "???"->"?"

    # 4) Gạch ngang nội bộ
    if strip_dash:
        s = s.replace("-", " ")
    else:
        s = RE_INNER_DASH.sub(" ", s)

    # 5) Chuẩn hoá số/ngày
    if normalize_numbers:
        s = normalize_numbers_and_dates(s)

    # 6) Lọc ký tự lạ
    if drop_bad_symbols:
        s = RE_BAD_SYM.sub(" ", s)

    # 7) Chuẩn hoá dấu : ; , . và khoảng trắng xung quanh (có bảo toàn số)
    s = _normalize_colon_semicolon_comma_and_dots(s)

    # 8) Dọn khoảng trắng quanh nháy, nén space
    s = RE_QUOTES_SP.sub(r"\1", s)
    s = RE_SPACES.sub(" ", s).strip()

    # 9) Tuỳ chọn bỏ !? nếu muốn
    if not keep_exclam_question:
        s = s.replace("!", " ").replace("?", " ")
        s = RE_SPACES.sub(" ", s).strip()

    return s


# =========================================================
# 1) WORD SEGMENTATION (PhoBERT yêu c?u)
# =========================================================
_SEGMENTER_BACKEND = None
_SEGMENTER_OBJ = None

def _init_segmenter(prefer_py_vncorenlp: bool = True):
    global _SEGMENTER_BACKEND, _SEGMENTER_OBJ
    if _SEGMENTER_BACKEND: return
    if prefer_py_vncorenlp:
        try:
            from py_vncorenlp import VnCoreNLP
            _SEGMENTER_OBJ = VnCoreNLP(annotators=["wseg"])
            _SEGMENTER_BACKEND = "py_vncorenlp"
            return
        except Exception:
            pass
    try:
        import underthesea
        _SEGMENTER_OBJ = underthesea
        _SEGMENTER_BACKEND = "underthesea"
    except Exception as e:
        raise RuntimeError("Không tìm th?y segmenter. Cài py_vncorenlp ho?c underthesea. L?i:", e)

def vn_word_segment(text: str) -> str:
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
    if not t: raise ValueError(f"Không tìm th?y c?t text trong keys={sample.keys()}")
    if not y: raise ValueError(f"Không tìm th?y c?t nhãn trong keys={sample.keys()}")
    return t, y

def normalize_label(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = str(s).strip().lower()
    mp = {"pos": "positive", "+": "positive", "positive": "positive",
          "neg": "negative", "-": "negative", "negative": "negative",
          "neu": "neutral", "0": "neutral", "neutral": "neutral"}
    return mp.get(s, s)

def build_label_maps(labels: List[str]):
    uniq = sorted(set(labels))
    order = ["positive", "neutral", "negative"]
    ordered = [c for c in order if c in uniq] + [c for c in uniq if c not in order]
    label2id = {c: i for i, c in enumerate(ordered)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label

# =========================================================
# 3) TORCH DATASET
# =========================================================
class SentDataset(Dataset):
    def __init__(self, rows, text_field, label_field, tokenizer, label2id, max_length, pp_kwargs):
        self.samples = []
        for r in rows:
            text = r.get(text_field)
            lab  = normalize_label(r.get(label_field))
            if text is not None and (lab is not None or label_field not in r):
                self.samples.append({"text": text, "label": lab})
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.pp_kwargs = pp_kwargs

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        clean = preprocess_text_vi(s["text"], **self.pp_kwargs)
        seg   = vn_word_segment(clean)
        enc = self.tokenizer(seg, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
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
        return {"accuracy": accuracy_score(labels, preds),
                "f1_macro": f1_score(labels, preds, average="macro")}
    return _fn

def detailed_report(trainer, dataset, id2label):
    out = trainer.predict(dataset)
    preds = np.argmax(out.predictions, axis=1)
    print("\n=== Classification report ===")
    print(classification_report(out.label_ids, preds,
          target_names=[id2label[i] for i in sorted(id2label)], digits=4))
    print("\n=== Confusion matrix ===")
    print(confusion_matrix(out.label_ids, preds))

# =========================================================
# 5) MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="./Round-1/final-round1.json")
    parser.add_argument("--eval_json",  type=str, default="gold-data.json")
    parser.add_argument("--test_json",  type=str, default=None)
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument("--label_field",type=str, default=None)
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--output_dir", type=str, default="phobert-ft")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--keep_exclam_question", action="store_true")
    parser.add_argument("--strip_dash", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_rows = load_json_array(args.train_json)
    eval_rows  = load_json_array(args.eval_json)
    if not train_rows or not eval_rows:
        raise RuntimeError("Thi?u d? li?u train ho?c eval.")

    if not args.text_field or not args.label_field:
        t, y = guess_fields(train_rows[0])
        args.text_field, args.label_field = args.text_field or t, args.label_field or y
    print(f"[Fields] text={args.text_field} | label={args.label_field}")

    train_labels = [normalize_label(r.get(args.label_field)) for r in train_rows if r.get(args.label_field)]
    label2id, id2label = build_label_maps(train_labels)
    print("[Labels]", label2id)

    cfg = AutoConfig.from_pretrained(args.model_name, num_labels=len(label2id),
                                     id2label=id2label, label2id=label2id)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=cfg).to(device)

    pp_kwargs = dict(remove_urls=True, remove_emails=True, mask_mentions=True,
                     normalize_numbers=True, drop_bad_symbols=True,
                     keep_exclam_question=args.keep_exclam_question,
                     strip_dash=args.strip_dash)

    train_ds = SentDataset(train_rows, args.text_field, args.label_field, tok, label2id, args.max_length, pp_kwargs)
    eval_ds  = SentDataset(eval_rows,  args.text_field, args.label_field, tok, label2id, args.max_length, pp_kwargs)

    amp = {}
    if torch.cuda.is_available():
        if args.bf16: amp["bf16"] = True
        elif args.fp16: amp["fp16"] = True
        else:
            major, _ = torch.cuda.get_device_capability(0)
            amp["bf16" if major >= 8 else "fp16"] = True

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,        # cho phép ghi đè toàn bộ output_dir khi chạy lại
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        evaluation_strategy="epoch",      # evaluate mỗi epoch
        save_strategy="epoch",            # chỉ save mỗi epoch
        save_total_limit=1,               # GIỮ TỐI ĐA 1 checkpoint (cũ bị xóa)
        load_best_model_at_end=True,      # cuối training load checkpoint tốt nhất
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        logging_steps=50,
        seed=args.seed,
        report_to="none",
        **amp
    )


    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_builder(MetricsCfg(id2label=id2label)),
        tokenizer=tok,
    )

    trainer.train()
    print("\n[Evaluate]")
    res = trainer.evaluate(eval_dataset=eval_ds)
    print(res)
    detailed_report(trainer, eval_ds, id2label)

    # Optional test predictions
    if args.test_json:
        test_rows = load_json_array(args.test_json)
        test_ds = SentDataset(test_rows, args.text_field, args.label_field, tok, label2id, args.max_length, pp_kwargs)
        preds = trainer.predict(test_ds)
        pred_ids = np.argmax(preds.predictions, axis=1)
        pred_labels = [id2label[i] for i in pred_ids]
        import pandas as pd
        pd.DataFrame({
            "text": [r.get(args.text_field, "") for r in test_rows],
            "prediction": pred_labels
        }).to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False, encoding="utf-8")
        print("[Test] Saved predictions to:", os.path.join(args.output_dir, "test_predictions.csv"))

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print("\n[Done] Saved model to:", args.output_dir)

if __name__ == "__main__":
    main()
