import pandas as pd
import re
import unicodedata

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================
# 1) Load dữ liệu (train/val/test)
# =========================

TRAIN_PATH = "preprocessed-dataset/train_processed.json"
VAL_PATH   = "preprocessed-dataset/dev_processed.json"
TEST_PATH  = "preprocessed-dataset/test_processed.json"


def load_json_flexible(path: str) -> pd.DataFrame:
    # Nếu file là 1 list các object JSON
    try:
        df = pd.read_json(path)
        print(f"Đọc JSON dạng array OK: {path}")
        return df
    except ValueError:
        # Nếu là JSON Lines (mỗi dòng 1 object)
        df = pd.read_json(path, lines=True)
        print(f"Đọc JSON dạng lines OK: {path}")
        return df


df_train = load_json_flexible(TRAIN_PATH)
df_val   = load_json_flexible(VAL_PATH)
df_test  = load_json_flexible(TEST_PATH)

for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    missing = {"review", "sentiment"} - set(d.columns)
    if missing:
        raise ValueError(f"Dataset {name} thiếu cột: {missing}. Các cột hiện có: {list(d.columns)}")

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)
print("Test shape:", df_test.shape)


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
    text = clean_review_basic(text)
    text = remove_vietnamese_accents(text)
    return text


def normalize_sentiment_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df["review_clean"] = df["review"].apply(preprocess_review)
    return df


df_train = normalize_sentiment_col(df_train)
df_val   = normalize_sentiment_col(df_val)
df_test  = normalize_sentiment_col(df_test)


# =========================
# 3) Encode label: fit trên train, transform cho val/test
# =========================

le = LabelEncoder()
le.fit(df_train["sentiment"])

train_labels_set = set(le.classes_)
val_extra = set(df_val["sentiment"].unique()) - train_labels_set
test_extra = set(df_test["sentiment"].unique()) - train_labels_set
if val_extra or test_extra:
    raise ValueError(
        "Val/Test có nhãn không xuất hiện trong Train.\n"
        f"Val extra labels: {sorted(val_extra)}\n"
        f"Test extra labels: {sorted(test_extra)}\n"
        f"Train labels: {list(le.classes_)}"
    )

df_train["label_id"] = le.transform(df_train["sentiment"])
df_val["label_id"]   = le.transform(df_val["sentiment"])
df_test["label_id"]  = le.transform(df_test["sentiment"])

print("Classes:", list(le.classes_))
print("Train label distribution:\n", df_train["sentiment"].value_counts())


X_train = df_train["review_clean"].values
y_train = df_train["label_id"].values

X_val = df_val["review_clean"].values
y_val = df_val["label_id"].values

X_test = df_test["review_clean"].values
y_test = df_test["label_id"].values


# =========================
# 4) Pipeline + Grid search theo VAL (không dùng train_test_split nữa)
# =========================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("dt", DecisionTreeClassifier(random_state=42)),
])

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 3), (1, 5), (1, 7)],
    "tfidf__max_df": [0.5, 0.6, 0.7, 0.8],

    "dt__criterion": ["gini", "entropy"],
    "dt__max_depth": [None, 20, 40, 60, 80],
    "dt__min_samples_leaf": [2, 4, 6, 8],
}

best_params = None
best_val_acc = -1.0

print("Đang huấn luyện và chọn tham số tốt nhất dựa trên VAL...")
for params in ParameterGrid(param_grid):
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params

print("Best VAL Accuracy:", best_val_acc)
print("Best params:")
for k, v in best_params.items():
    print(f"{k}: {v}")


# =========================
# 5) Đánh giá trên VAL (model tốt nhất)
# =========================

best_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("dt", DecisionTreeClassifier(random_state=42)),
]).set_params(**best_params)

best_model.fit(X_train, y_train)

y_val_pred = best_model.predict(X_val)
print("\nVAL Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification report (VAL):")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))
print("\nConfusion matrix (VAL):")
print(confusion_matrix(y_val, y_val_pred))


# =========================
# 6) Refit trên (train + val) rồi test (thực tế hơn)
# =========================

X_trainval = pd.concat([df_train["review_clean"], df_val["review_clean"]], axis=0).values
y_trainval = pd.concat([df_train["label_id"], df_val["label_id"]], axis=0).values

final_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("dt", DecisionTreeClassifier(random_state=42)),
]).set_params(**best_params)

final_model.fit(X_trainval, y_trainval)

y_test_pred = final_model.predict(X_test)
print("\nTEST Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification report (TEST):")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))
print("\nConfusion matrix (TEST):")
print(confusion_matrix(y_test, y_test_pred))
