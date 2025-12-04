# -*- coding: utf-8 -*-
"""
Predict sentiment cho file data-3.txt bằng PhoBERT fine-tuned

- Đọc file text có cấu trúc:
    #701
    <đoạn tin 701>

    #702
    <đoạn tin 702>
    ...
- Dùng CHÍNH XÁC pipeline:
    preprocess_text_vi + vn_word_segment + batched_predict
  đã định nghĩa trong eval_phobert_sentiment.py
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import từ file eval_phobert_sentiment.py
from eval_phobert_sentiment import batched_predict, entropy_row


def parse_data3_txt(path):
    """
    Parse file data-3.txt dạng:

        #701
        text...

        #702
        text...
        ...

    Trả về list dict: [{"id": 701, "text": "..."} , ...]
    """
    items = []
    cur_id = None
    cur_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Nếu là dòng id dạng #701, #702,...
            if line.startswith("#") and line[1:].strip().isdigit():
                # nếu đang có 1 bài trước đó thì push vào
                if cur_id is not None and cur_lines:
                    text = " ".join(l.strip() for l in cur_lines if l.strip())
                    items.append({"id": cur_id, "text": text})
                    cur_lines = []

                cur_id = int(line[1:].strip())
                continue

            # nếu là dòng trống
            if not line.strip():
                if cur_id is not None:
                    # có thể thêm break đoạn, ở đây giữ lại như 1 khoảng trắng
                    cur_lines.append("")
                continue

            # dòng text bình thường
            if cur_id is not None:
                cur_lines.append(line)
            else:
                # text xuất hiện trước #ID thì bỏ qua
                pass

    # push bài cuối nếu còn
    if cur_id is not None and cur_lines:
        text = " ".join(l.strip() for l in cur_lines if l.strip())
        items.append({"id": cur_id, "text": text})

    return items


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Thư mục model PhoBERT đã fine-tune (có config, weights, label_maps.json)")
    parser.add_argument("--input_txt", type=str, required=True,
                        help="Đường dẫn tới data-3.txt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)

    # GIỐNG eval_phobert_sentiment.py
    parser.add_argument("--keep_exclam_question", action="store_true",
                        help="Giữ !, ? sau preprocess")
    parser.add_argument("--strip_dash", action="store_true",
                        help="Đổi '-' thành space trước khi tokenize")

    parser.add_argument("--output_csv", type=str, default="prediction_data3.csv",
                        help="File CSV output")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 1) Parse txt
    items = parse_data3_txt(args.input_txt)
    print(f"[Parse] Đọc được {len(items)} đoạn tin từ {args.input_txt}")
    if not items:
        print("[Parse] Không có đoạn nào, kiểm tra lại file input.")
        return

    ids = [it["id"] for it in items]
    texts = [it["text"] for it in items]

    # 2) Load model + tokenizer
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    # label mapping
    label_maps_path = os.path.join(model_dir, "label_maps.json")
    if os.path.exists(label_maps_path):
        with open(label_maps_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        id2label = {int(k): v for k, v in lm["id2label"].items()}
        label2id = {k: int(v) for k, v in lm["label2id"].items()}
    else:
        # fallback theo config model
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label2id = {k: int(v) for k, v in model.config.label2id.items()}

    # 3) Predict với batched_predict (đã có preprocess_text_vi + vn_word_segment bên trong)
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

    # Lấy xác suất từng class (nếu tên class là positive / neutral / negative)
    idx_pos = label2id.get("positive", None)
    idx_neu = label2id.get("neutral", None)
    idx_neg = label2id.get("negative", None)

    ppos = [float(row[idx_pos]) if idx_pos is not None else np.nan for row in probs]
    pneu = [float(row[idx_neu]) if idx_neu is not None else np.nan for row in probs]
    pneg = [float(row[idx_neg]) if idx_neg is not None else np.nan for row in probs]

    ent = [entropy_row(p) for p in probs]

    # 4) Xuất CSV
    df = pd.DataFrame({
        "id": ids,
        "text_raw": texts,
        "text_clean": clean_list,
        "text_segmented": seg_list,
        "pred": pred_labels,
        "p_positive": ppos,
        "p_neutral": pneu,
        "p_negative": pneg,
        "entropy": ent
    })

    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print("[DONE] Saved:", args.output_csv)


if __name__ == "__main__":
    main()