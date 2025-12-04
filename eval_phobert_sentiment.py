# -*- coding: utf-8 -*-
"""
Evaluation script for PhoBERT sentiment
- Loads fine-tuned model from --model_dir
- Consistent preprocessing + word segmentation
- Exports full predictions, mispredictions, top-uncertain
"""

import os, re, json, unicodedata, argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Preprocess (ASCII/Unicode-safe)
# -----------------------------
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
RE_BAD_SYM   = re.compile(r"[^\w\s\.\,\!\?\:\;\-\(\)\"\'/àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                          r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
                          r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄ"
                          r"ÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮ"
                          r"ỲÝỴỶỸĐ]")

TRANS_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2026": "..."
})

def normalize_numbers_and_dates(text: str) -> str:
    return RE_DIGIT.sub("0", text)

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
    s = unicodedata.normalize("NFKC", s)
    s = RE_CTRL_ZW.sub("", s).translate(TRANS_TABLE)
    if remove_urls:  s = RE_URL.sub(" ", s)
    if remove_emails:s = RE_EMAIL.sub(" ", s)
    if mask_mentions:s = RE_MENTION.sub("@user", s)
    s = RE_ELLIPSIS.sub("...", s)
    s = RE_PUNCT_RUN.sub(lambda m: m.group(1), s)
    if strip_dash: s = s.replace("-", " ")
    else:          s = RE_INNER_DASH.sub(" ", s)
    if normalize_numbers: s = normalize_numbers_and_dates(s)
    if drop_bad_symbols:  s = RE_BAD_SYM.sub(" ", s)
    s = RE_QUOTES_SP.sub(r'\1', s)
    s = RE_SPACES.sub(" ", s).strip()
    if not keep_exclam_question:
        s = s.replace("!", " ").replace("?", " ")
        s = RE_SPACES.sub(" ", s).strip()
    return s

# -----------------------------
# Word segmentation
# -----------------------------
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
        raise RuntimeError("No segmenter. Install py_vncorenlp or underthesea. Err:", e)

def vn_word_segment(text: str) -> str:
    if not text: return ""
    if _SEGMENTER_BACKEND is None:
        _init_segmenter(prefer_py_vncorenlp=True)
    if _SEGMENTER_BACKEND == "py_vncorenlp":
        return _SEGMENTER_OBJ.word_segment(text)
    tokens = _SEGMENTER_OBJ.word_tokenize(text)
    return tokens if isinstance(tokens, str) else " ".join(tokens)

# -----------------------------
# Utils
# -----------------------------
def load_json_array(path: str):
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict): data = [data]
            return data
        except json.JSONDecodeError:
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows

def guess_fields(sample: dict):
    text_keys  = ["review","text","content","body","sentence","title"]
    label_keys = ["sentiment","label","y","tag"]
    t = next((k for k in text_keys  if k in sample), None)
    y = next((k for k in label_keys if k in sample), None)
    if not t: raise ValueError(f"No text field in {list(sample.keys())}")
    return t, y  # y may be None for test json

def normalize_label(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = str(s).strip().lower()
    mp = {"pos":"positive","+":"positive","positive":"positive",
          "neg":"negative","-":"negative","negative":"negative",
          "neu":"neutral","0":"neutral","neutral":"neutral"}
    return mp.get(s, s)

def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def entropy_row(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

# -----------------------------
# Predict helper
# -----------------------------
def batched_predict(texts: List[str], tokenizer, model, max_length=256, batch_size=32,
                    keep_exclam_question=True, strip_dash=False):
    device = next(model.parameters()).device
    preds, probs, clean_list, seg_list = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_raw = texts[i:i+batch_size]
            batch_clean = [preprocess_text_vi(t,
                                              remove_urls=True,
                                              remove_emails=True,
                                              mask_mentions=True,
                                              normalize_numbers=True,
                                              drop_bad_symbols=True,
                                              keep_exclam_question=keep_exclam_question,
                                              strip_dash=strip_dash) for t in batch_raw]
            batch_seg = [vn_word_segment(t) for t in batch_clean]
            enc = tokenizer(batch_seg, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=1).cpu().numpy()
            y = p.argmax(axis=1)
            preds.extend(y.tolist())
            probs.extend(p.tolist())
            clean_list.extend(batch_clean)
            seg_list.extend(batch_seg)
    return np.array(preds), np.array(probs), clean_list, seg_list

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="phobert-ft")
    ap.add_argument("--eval_json", type=str, default="gold-data.json")
    ap.add_argument("--test_json", type=str, default=None)
    ap.add_argument("--text_field", type=str, default=None)
    ap.add_argument("--label_field", type=str, default=None)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--keep_exclam_question", action="store_true")
    ap.add_argument("--strip_dash", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) Load model + tokenizer + label maps
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    # label maps
    label_maps_path = os.path.join(args.model_dir, "label_maps.json")
    if os.path.exists(label_maps_path):
        with open(label_maps_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        id2label = {int(k): v for k, v in lm["id2label"].items()}
        label2id = {k: int(v) for k, v in lm["label2id"].items()}
    else:
        # fallback from config
        id2label = model.config.id2label
        label2id = model.config.label2id
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {k: int(v) for k, v in label2id.items()}

    # 2) Evaluate on gold (has labels)
    eval_rows = load_json_array(args.eval_json)
    t_field, y_field = args.text_field, args.label_field
    if t_field is None or y_field is None:
        t_guess, y_guess = guess_fields(eval_rows[0])
        t_field = t_field or t_guess
        y_field = y_field or y_guess
    texts = [r.get(t_field, "") for r in eval_rows]
    gold_labels_str = [normalize_label(r.get(y_field)) for r in eval_rows]
    gold_ids = np.array([label2id[g] for g in gold_labels_str])

    pred_ids, probs, clean_list, seg_list = batched_predict(
        texts, tokenizer, model,
        max_length=args.max_length, batch_size=args.batch_size,
        keep_exclam_question=args.keep_exclam_question, strip_dash=args.strip_dash
    )

    acc = accuracy_score(gold_ids, pred_ids)
    f1m = f1_score(gold_ids, pred_ids, average="macro")
    print("\n[Eval] accuracy=%.4f  macroF1=%.4f" % (acc, f1m))
    print("\n[Eval] Classification report:")
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(classification_report(gold_ids, pred_ids, target_names=target_names, digits=4))
    print("[Eval] Confusion matrix:")
    print(confusion_matrix(gold_ids, pred_ids))

    # 3) Export CSVs
    import pandas as pd
    pred_labels = [id2label[int(i)] for i in pred_ids]
    ppos = []
    pneu = []
    pneg = []
    # Ensure order
    idx_pos = label2id.get("positive", None)
    idx_neu = label2id.get("neutral", None)
    idx_neg = label2id.get("negative", None)
    for row in probs:
        ppos.append(float(row[idx_pos]) if idx_pos is not None else float("nan"))
        pneu.append(float(row[idx_neu]) if idx_neu is not None else float("nan"))
        pneg.append(float(row[idx_neg]) if idx_neg is not None else float("nan"))

    entropy = [entropy_row(p) for p in probs]

    df_all = pd.DataFrame({
        "text_raw": texts,
        "text_clean": clean_list,
        "text_segmented": seg_list,
        "gold": gold_labels_str,
        "pred": pred_labels,
        "p_positive": ppos,
        "p_neutral":  pneu,
        "p_negative": pneg,
        "entropy": entropy
    })
    out_dir = args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    df_all.to_csv(os.path.join(out_dir, "eval_predictions.csv"), index=False, encoding="utf-8-sig")

    df_err = df_all[df_all["gold"] != df_all["pred"]].copy()
    df_err.to_csv(os.path.join(out_dir, "eval_mispredictions.csv"), index=False, encoding="utf-8-sig")

    df_uncertain = df_all.sort_values("entropy", ascending=False).head(200).copy()
    df_uncertain.to_csv(os.path.join(out_dir, "eval_top_uncertain.csv"), index=False, encoding="utf-8-sig")

    print("[Eval] Saved:", os.path.join(out_dir, "eval_predictions.csv"))
    print("[Eval] Saved:", os.path.join(out_dir, "eval_mispredictions.csv"))
    print("[Eval] Saved:", os.path.join(out_dir, "eval_top_uncertain.csv"))

    # 4) Optional: predict test (no labels)
    if args.test_json:
        test_rows = load_json_array(args.test_json)
        if args.text_field is None:
            t_field, _ = guess_fields(test_rows[0])
        texts_t = [r.get(t_field, "") for r in test_rows]
        pred_ids_t, probs_t, clean_t, seg_t = batched_predict(
            texts_t, tokenizer, model,
            max_length=args.max_length, batch_size=args.batch_size,
            keep_exclam_question=args.keep_exclam_question, strip_dash=args.strip_dash
        )
        pred_labels_t = [id2label[int(i)] for i in pred_ids_t]
        ppos_t, pneu_t, pneg_t = [], [], []
        for row in probs_t:
            ppos_t.append(float(row[idx_pos]) if idx_pos is not None else float("nan"))
            pneu_t.append(float(row[idx_neu]) if idx_neu is not None else float("nan"))
            pneg_t.append(float(row[idx_neg]) if idx_neg is not None else float("nan"))

        df_test = pd.DataFrame({
            "text_raw": texts_t,
            "text_clean": clean_t,
            "text_segmented": seg_t,
            "pred": pred_labels_t,
            "p_positive": ppos_t,
            "p_neutral":  pneu_t,
            "p_negative": pneg_t
        })
        out_csv = os.path.join(out_dir, "test_predictions.csv")
        df_test.to_csv(out_csv, index=False, encoding="utf-8")
        print("[Test] Saved:", out_csv)

    # 5) Quick demo
    demo = ["Bài báo này rất tuyệt vời!", "Rất tệ, không nên đọc."]
    _, _ = None, None
    demo_ids, demo_probs, _, _ = batched_predict(demo, tokenizer, model,
                                                 max_length=args.max_length, batch_size=8,
                                                 keep_exclam_question=args.keep_exclam_question,
                                                 strip_dash=args.strip_dash)
    print("\n[Demo]")
    for t, i in zip(demo, demo_ids):
        print(" ", t, "->", id2label[int(i)])

if __name__ == "__main__":
    main()
