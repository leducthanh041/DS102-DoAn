import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# =========================================================
# 1) CONFIG
# =========================================================
DEV_FILE   = "./preprocessed-dataset/dev_processed.json"
OUTPUT_DIR = "./xgboost_results"

MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_best_model.pkl")
VECT_PATH  = os.path.join(leading_output := OUTPUT_DIR, "tfidf_vectorizer.pkl")
LE_PATH    = os.path.join(OUTPUT_DIR, "label_encoder.pkl")


def load_data(file_path: str) -> pd.DataFrame:
    """Đọc file JSON lines và trả về DataFrame (fallback nếu không phải jsonl)."""
    try:
        return pd.read_json(file_path, orient="records", lines=True)
    except ValueError:
        return pd.read_json(file_path)


def plot_confusion_matrix(cm, labels, title, normalize=False, figsize=(7, 6)):
    """
    Vẽ confusion matrix đẹp.
    normalize=True => chuẩn hoá theo hàng (true label) để ra tỷ lệ.
    """
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


def main():
    # =========================================================
    # 2) LOAD BEST ARTIFACTS
    # =========================================================
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    if not os.path.exists(VECT_PATH):
        raise FileNotFoundError(f"Không tìm thấy TF-IDF vectorizer: {VECT_PATH}")
    if not os.path.exists(LE_PATH):
        raise FileNotFoundError(f"Không tìm thấy label encoder: {LE_PATH}")

    best_model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECT_PATH)
    le = joblib.load(LE_PATH)

    print("[Loaded] model, tfidf, label_encoder from:", OUTPUT_DIR)
    print("Label classes:", list(le.classes_))

    # =========================================================
    # 3) LOAD DEV DATA
    # =========================================================
    df_dev = load_data(DEV_FILE)
    if "review" not in df_dev.columns or "sentiment" not in df_dev.columns:
        raise ValueError("DEV file phải có 2 cột: 'review' và 'sentiment'.")

    X_dev = df_dev["review"].astype(str)
    y_dev = le.transform(df_dev["sentiment"].astype(str))  # mã hoá theo mapping đã fit

    # =========================================================
    # 4) TRANSFORM + PREDICT ON DEV
    # =========================================================
    X_dev_tfidf = tfidf.transform(X_dev)
    y_dev_pred = best_model.predict(X_dev_tfidf)

    # =========================================================
    # 5) METRICS
    # =========================================================
    acc = accuracy_score(y_dev, y_dev_pred)
    f1m = f1_score(y_dev, y_dev_pred, average="macro")

    print("\n[DEV Evaluation]")
    print(f"Dev Accuracy : {acc:.4f}")
    print(f"Dev F1-macro: {f1m:.4f}")

    print("\n=== Classification Report (DEV) ===")
    print(classification_report(y_dev, y_dev_pred, target_names=le.classes_, digits=4))

    # =========================================================
    # 6) CONFUSION MATRIX (raw + normalized) + SAVE CSV
    # =========================================================
    cm = confusion_matrix(y_dev, y_dev_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_dev.csv")
    cm_df.to_csv(cm_path, encoding="utf-8")
    print("\nSaved confusion matrix to:", cm_path)

    plot_confusion_matrix(
        cm,
        labels=le.classes_,
        title="Confusion Matrix – XGBoost (DEV)",
        normalize=False
    )

    plot_confusion_matrix(
        cm,
        labels=le.classes_,
        title="Normalized Confusion Matrix – XGBoost (DEV)",
        normalize=True
    )

    # =========================================================
    # 7) ERROR ANALYSIS ON DEV (SAVE CSV)
    # =========================================================
    id2label = {i: lab for i, lab in enumerate(le.classes_)}

    errors = []
    for idx, (true_id, pred_id) in enumerate(zip(y_dev, y_dev_pred)):
        if true_id != pred_id:
            errors.append({
                "review": df_dev.iloc[idx]["review"],
                "true_label": id2label[int(true_id)],
                "pred_label": id2label[int(pred_id)]
            })

    err_df = pd.DataFrame(errors)
    err_path = os.path.join(OUTPUT_DIR, "error_analysis_dev.csv")
    err_df.to_csv(err_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(err_df)} misclassified samples to:", err_path)

    print("\n[Done] Evaluated XGBoost best model on DEV.")


if __name__ == "__main__":
    main()
