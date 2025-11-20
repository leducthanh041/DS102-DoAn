# -*- coding: utf-8 -*-
"""
Predict sentiment cho file CSV (vd: label_disagreementsCopy.csv)
- Cột text mặc định: 'review'
- Dùng PhoBERT đã fine-tune
- Preprocess + word segmentation + batched_predict giống eval_phobert_sentiment.py
"""

import os
import re
import json
import argparse
import unicodedata
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# 1. PREPROCESS (giống đoạn bạn đưa)
# ============================================================

RE_CTRL_ZW   = re.compile(r"[\u200B-\u200D\uFEFF]")
RE_SPACES    = re.compile(r"\s+")
RE_URL       = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
RE_EMAIL     = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_MENTION   = re.compile(r"(?<!\w)@[\w_]+")
RE_DIGIT     = re.compile(r"\d")
RE_PUNCT_RUN = re.compile(r"([!?]){2,}")
RE_ELLIPSIS  = re.compile(r"\.{3,}")
RE_QUOTES_SP = re.compile(r"\s*([\"'])\s*")
RE_INNER_DASH= re.compile(r"(?<=\w)[\-–—−](?=\w)")
RE_BAD_SYM   = re.compile(
    r"[^\w\s\.\,\!\?\:\;\-\(\)\"\'/àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
    r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄ"
    r"ÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮ"
    r"ỲÝỴỶỸĐ]"
)

TRANS_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2026": "..."
})

def normalize_numbers_and_dates(text: str) -> str:
    return RE_DIGIT.sub("0", text)

def preprocess_text_vi(
    s: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    mask_mentions: bool = True,
    normalize_numbers: bool = True,
    drop_bad_symbols: bool = True,
    keep_exclam_question: bool = True,
    strip_dash: bool = False
) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = RE_CTRL_ZW.sub("", s).translate(TRANS_TABLE)
    if remove_urls:
        s = RE_URL.sub(" ", s)
    if remove_emails:
        s = RE_EMAIL.sub(" ", s)
    if mask_mentions:
        s = RE_MENTION.sub("@user", s)
    s = RE_ELLIPSIS.sub("...", s)
    s = RE_PUNCT_RUN.sub(lambda m: m.group(1), s)
    if strip_dash:
        s = s.replace("-", " ")
    else:
        s = RE_INNER_DASH.sub(" ", s)
    if normalize_numbers:
        s = normalize_numbers_and_dates(s)
    if drop_bad_symbols:
        s = RE_BAD_SYM.sub(" ", s)
    s = RE_QUOTES_SP.sub(r"\1", s)
    s = RE_SPACES.sub(" ", s).strip()
    if not keep_exclam_question:
        s = s.replace("!", " ").replace("?", " ")
        s = RE_SPACES.sub(" ", s).strip()
    return s

# ============================================================
# 2. WORD SEGMENTATION
# ============================================================

_SEGMENTER_BACKEND = None
_SEGMENTER_OBJ = None

def _init_segmenter(prefer_py_vncorenlp: bool = True):
    global _SEGMENTER_BACKEND, _SEGMENTER_OBJ
    if _SEGMENTER_BACKEND:
        return
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
        raise RuntimeError("No segmenter. Install py_vncorenlp or underthesea. Err:", e)

def vn_word_segment(text: str) -> str:
    if not text:
        return ""
    if _SEGMENTER_BACKEND is None:
        _init_segmenter(prefer_py_vncorenlp=True)
    if _SEGMENTER_BACKEND == "py_vncorenlp":
        return _SEGMENTER_OBJ.word_segment(text)
    tokens = _SEGMENTER_OBJ.word_tokenize(text)
    return tokens if isinstance(tokens, str) else " ".join(tokens)

# ============================================================
# 3. SOFTMAX, ENTROPY, BATCHED PREDICT
# ============================================================

def entropy_row(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def batched_predict(
    texts: List[str],
    tokenizer,
    model,
    max_length: int = 256,
    batch_size: int = 32,
    keep_exclam_question: bool = True,
    strip_dash: bool = False
):
    device = next(model.parameters()).device
    preds, probs, clean_list, seg_list = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_raw = texts[i:i + batch_size]
            batch_clean = [
                preprocess_text_vi(
                    t,
                    remove_urls=True,
                    remove_emails=True,
                    mask_mentions=True,
                    normalize_numbers=True,
                    drop_bad_symbols=True,
                    keep_exclam_question=keep_exclam_question,
                    strip_dash=strip_dash
                )
                for t in batch_raw
            ]
            batch_seg = [vn_word_segment(t) for t in batch_clean]

            enc = tokenizer(
                batch_seg,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            logits = model(**enc).logits
            p = torch.softmax(logits, dim=1).cpu().numpy()
            y = p.argmax(axis=1)

            preds.extend(y.tolist())
            probs.extend(p.tolist())
            clean_list.extend(batch_clean)
            seg_list.extend(batch_seg)

    return np.array(preds), np.array(probs), clean_list, seg_list

# ============================================================
# 4. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Thư mục model PhoBERT đã fine-tune (phobert-ft, ...)")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Đường dẫn tới file CSV (vd: label_disagreementsCopy.csv)")
    parser.add_argument("--text_col", type=str, default="review",
                        help="Tên cột chứa text (mặc định: review)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)

    # giống eval_phobert_sentiment.py
    parser.add_argument("--keep_exclam_question", action="store_true",
                        help="Giữ !, ? sau preprocess")
    parser.add_argument("--strip_dash", action="store_true",
                        help="Đổi '-' thành space trước khi tokenize")

    parser.add_argument("--output_csv", type=str, default="prediction_from_csv.csv",
                        help="File CSV output")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) Load CSV
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{args.text_col}' trong file CSV. Columns = {list(df.columns)}")

    texts = df[args.text_col].astype(str).tolist()
    print(f"[Input] Số dòng cần predict: {len(texts)}")

    # 2) Load model + tokenizer
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    # 3) Load label mapping
    label_maps_path = os.path.join(model_dir, "label_maps.json")
    if os.path.exists(label_maps_path):
        with open(label_maps_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        id2label = {int(k): v for k, v in lm["id2label"].items()}
        label2id = {k: int(v) for k, v in lm["label2id"].items()}
    else:
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label2id = {k: int(v) for k, v in model.config.label2id.items()}

    # 4) Predict
    pred_ids, probs, clean_list, seg_list = batched_predict(
        texts,
        tokenizer,
        model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        keep_exclam_question=args.keep_exclam_question,
        strip_dash=args.strip_dash
    )

    pred_labels = [id2label[int(i)] for i in pred_ids]

    # Lấy xác suất từng class (nếu tên nhãn là positive / neutral / negative)
    idx_pos = label2id.get("positive", None)
    idx_neu = label2id.get("neutral", None)
    idx_neg = label2id.get("negative", None)

    ppos = [float(row[idx_pos]) if idx_pos is not None else np.nan for row in probs]
    pneu = [float(row[idx_neu]) if idx_neu is not None else np.nan for row in probs]
    pneg = [float(row[idx_neg]) if idx_neg is not None else np.nan for row in probs]

    ent = [entropy_row(p) for p in probs]

    # 5) Gắn vào DataFrame gốc
    df["text_clean"]      = clean_list
    df["text_segmented"]  = seg_list
    df["pred"]            = pred_labels
    df["p_positive"]      = ppos
    df["p_neutral"]       = pneu
    df["p_negative"]      = pneg
    df["entropy"]         = ent

    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print("[DONE] Saved:", args.output_csv)


if __name__ == "__main__":
    main()