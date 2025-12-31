import os
import random
import json
import re
import unicodedata
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from scikeras.wrappers import KerasClassifier

# =========================
# 0) Config + Seed
# =========================
TRAIN_PATH = "./preprocessed-dataset/train_processed.json"
VAL_PATH   = "./preprocessed-dataset/dev_processed.json"
TEST_PATH  = "./preprocessed-dataset/test_processed.json"

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# =========================
# 1) Load dữ liệu (không dùng pandas)
# =========================
def load_json_flexible(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"File rỗng: {path}")

    if raw.lstrip().startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"JSON array không hợp lệ (không phải list): {path}")
        print(f"Đọc JSON dạng array OK: {path}")
        return data

    items = []
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as err:
            raise ValueError(f"Lỗi JSON Lines tại dòng {i} trong {path}: {err}") from err

    print(f"Đọc JSON dạng lines OK: {path}")
    return items


def validate_and_extract(records, split_name: str):
    reviews, sentiments = [], []
    for idx, r in enumerate(records):
        if not isinstance(r, dict):
            raise ValueError(f"{split_name}: phần tử {idx} không phải object JSON")
        if "review" not in r or "sentiment" not in r:
            raise ValueError(f"{split_name}: phần tử {idx} thiếu 'review' hoặc 'sentiment'")
        reviews.append(r["review"])
        sentiments.append(r["sentiment"])
    return reviews, sentiments


train_records = load_json_flexible(TRAIN_PATH)
val_records   = load_json_flexible(VAL_PATH)
test_records  = load_json_flexible(TEST_PATH)

X_train_raw, y_train_raw = validate_and_extract(train_records, "train")
X_val_raw,   y_val_raw   = validate_and_extract(val_records, "val")
X_test_raw,  y_test_raw  = validate_and_extract(test_records, "test")

print("Train size:", len(X_train_raw))
print("Val size:", len(X_val_raw))
print("Test size:", len(X_test_raw))

# =========================
# 2) Preprocessing: giữ như trước + bỏ dấu tiếng Việt
# =========================
def clean_review_basic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_vietnamese_accents(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = unicodedata.normalize("NFC", text)
    return text


def preprocess_review(text):
    return remove_vietnamese_accents(clean_review_basic(text))


X_train = [preprocess_review(t) for t in X_train_raw]
X_val   = [preprocess_review(t) for t in X_val_raw]
X_test  = [preprocess_review(t) for t in X_test_raw]


def normalize_label(x):
    return str(x).lower().strip()


y_train_norm = [normalize_label(x) for x in y_train_raw]
y_val_norm   = [normalize_label(x) for x in y_val_raw]
y_test_norm  = [normalize_label(x) for x in y_test_raw]

# =========================
# 3) Encode label: fit trên train, transform cho val/test
# =========================
le = LabelEncoder()
le.fit(y_train_norm)

train_labels_set = set(le.classes_)
val_extra = set(y_val_norm) - train_labels_set
test_extra = set(y_test_norm) - train_labels_set
if val_extra or test_extra:
    raise ValueError(
        "Val/Test có nhãn không xuất hiện trong Train.\n"
        f"Val extra labels: {sorted(val_extra)}\n"
        f"Test extra labels: {sorted(test_extra)}\n"
        f"Train labels: {list(le.classes_)}"
    )

y_train = le.transform(y_train_norm)
y_val   = le.transform(y_val_norm)
y_test  = le.transform(y_test_norm)

classes_str = [str(c) for c in le.classes_]
print("Classes:", classes_str)

# =========================
# 4) TF-IDF + GRU + GridSearchCV
# =========================
num_classes = len(classes_str)

def build_gru_model(meta, gru_units=128, lr=1e-3, n_classes=num_classes):
    input_shape = meta["X_shape_"][1:]  # (timesteps, features)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.GRU(gru_units),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

to_dense = FunctionTransformer(
    lambda X: X.toarray().astype(np.float32),
    accept_sparse=True
)

to_3d = FunctionTransformer(
    lambda X: X.reshape((X.shape[0], X.shape[1], 1)),
    validate=False
)

pipe = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
    )),
    ("dense", to_dense),
    ("to3d", to_3d),
    ("clf", KerasClassifier(
        model=build_gru_model,
        verbose=0,
        random_state=SEED,
    )),
])

param_grid = {
    "tfidf__max_features": [11400],
    "tfidf__ngram_range": [(1, 2)],
    "tfidf__min_df": [1],
    "tfidf__max_df": [0.2],
    "tfidf__sublinear_tf": [True],
    "tfidf__norm": ["l2"],

    "clf__model__gru_units": [128],
    "clf__model__lr": [1e-3],
    "clf__batch_size": [64],
    "clf__epochs": [10],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=1,
    verbose=2,
    refit=True,
)

grid.fit(X_train, y_train)

print("\nBest CV F1-macro:", grid.best_score_)
print("Best params:", grid.best_params_)

# =========================
# 5) Train final: fit best params trên TRAIN + VAL
# =========================
X_trainval = X_train + X_val
y_trainval = np.concatenate([y_train, y_val], axis=0)

final_model = grid.best_estimator_
final_model.fit(X_trainval, y_trainval)

# =========================
# 6) Evaluate TEST
# =========================
y_test_pred = final_model.predict(X_test)

test_acc = accuracy_score(y_test, y_test_pred)
test_f1m = f1_score(y_test, y_test_pred, average="macro")

print("\nTEST Accuracy:", test_acc)
print("TEST F1-macro:", test_f1m)
print("\nClassification report (TEST):")
print(classification_report(y_test, y_test_pred, target_names=classes_str))

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_str)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d", colorbar=True)
ax.set_title("Confusion Matrix - TF-IDF + GRU (TEST)")
plt.tight_layout()
plt.show()