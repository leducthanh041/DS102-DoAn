import pandas as pd
import numpy as np
import json
import os
import random
import joblib  # Để lưu model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =========================================================
# 1. CẤU HÌNH & LOAD DỮ LIỆU
# =========================================================
TRAIN_FILE = './preprocessed-dataset/train_processed.json'
DEV_FILE   = './preprocessed-dataset/dev_processed.json'
TEST_FILE  = './preprocessed-dataset/test_processed.json'
OUTPUT_DIR = './xgboost_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data(file_path):
    """Đọc file JSON lines và trả về DataFrame"""
    try:
        return pd.read_json(file_path, orient='records', lines=True)
    except ValueError:
        return pd.read_json(file_path)

print("Đang load dữ liệu...")
df_train = load_data(TRAIN_FILE)
df_dev   = load_data(DEV_FILE)
df_test  = load_data(TEST_FILE)

print(f"Số lượng mẫu: Train={len(df_train)}, Dev={len(df_dev)}, Test={len(df_test)}")

# =========================================================
# 2. XỬ LÝ NHÃN (LABEL ENCODING)
# =========================================================
# XGBoost cần nhãn dạng số (0, 1, 2)
le = LabelEncoder()
# Fit trên tập train để học mapping
y_train = le.fit_transform(df_train['sentiment'])
# Transform cho Dev và Test
y_dev   = le.transform(df_dev['sentiment'])
y_test  = le.transform(df_test['sentiment'])

# Lưu lại mapping để dùng sau này
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
id2label = {i: label for label, i in label_mapping.items()}
print("Label mapping:", label_mapping)

# =========================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG (TF-IDF)
# =========================================================
print("Đang vector hóa văn bản (TF-IDF)...")
# Chỉ sử dụng top 5000 từ quan trọng nhất để giảm chiều dữ liệu
# ngram_range=(1,2): Lấy cả từ đơn và từ ghép 2 từ liền nhau
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# QUAN TRỌNG: Chỉ fit trên tập Train để tránh rò rỉ dữ liệu (Data Leakage)
X_train_tfidf = tfidf.fit_transform(df_train['review'])
X_dev_tfidf   = tfidf.transform(df_dev['review'])
X_test_tfidf  = tfidf.transform(df_test['review'])

print(f"Kích thước vector Train: {X_train_tfidf.shape}")

# =========================================================
# 4. TUNING THAM SỐ (RANDOM SEARCH ON DEV)
# =========================================================
print("\nBắt đầu quá trình Tuning tham số trên tập Dev...")

# Không gian tham số để tìm kiếm
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],           # Tỷ lệ mẫu dùng để train mỗi cây
    'colsample_bytree': [0.6, 0.8, 1.0],    # Tỷ lệ feature dùng cho mỗi cây
    'objective': ['multi:softprob'],
    'num_class': [len(le.classes_)],
    'eval_metric': ['mlogloss']
}

best_score = -1
best_params = None
best_model = None

# Số lần thử ngẫu nhiên
N_ITER = 45 

for i in range(N_ITER):
    # Random chọn tham số
    params = {k: random.choice(v) for k, v in param_dist.items() if isinstance(v, list)}
    
    # Khởi tạo model
    model = XGBClassifier(**params, random_state=42, n_jobs=-1)
    
    # Train trên Train, đánh giá trên Dev
    # early_stopping_rounds=10: Dừng nếu sau 10 vòng ko cải thiện trên tập Dev
    model.fit(
        X_train_tfidf, y_train,
        eval_set=[(X_dev_tfidf, y_dev)],
        verbose=False
    )
    
    # Dự đoán trên Dev để tính F1 Score
    y_dev_pred = model.predict(X_dev_tfidf)
    f1 = f1_score(y_dev, y_dev_pred, average='macro')
    
    print(f"Trial {i+1}/{N_ITER} | F1-Dev: {f1:.4f} | Params: {params}")
    
    # Lưu lại model tốt nhất
    if f1 > best_score:
        best_score = f1
        best_params = params
        best_model = model

print("-" * 50)
print(f"BEST F1-Macro trên Dev: {best_score:.4f}")
print("Best Params:", best_params)

# =========================================================
# 5. ĐÁNH GIÁ TRÊN TẬP TEST
# =========================================================
print("\nĐánh giá model tốt nhất trên tập Test...")

# Dùng model tốt nhất đã tìm được để dự đoán Test
y_test_pred = best_model.predict(X_test_tfidf)

# 1. Classification Report
acc = accuracy_score(y_test, y_test_pred)
macro_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1-Macro: {macro_f1:.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# 2. Lưu Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
cm_file = os.path.join(OUTPUT_DIR, 'confusion_matrix.csv')
cm_df.to_csv(cm_file)
print(f"Đã lưu Confusion Matrix vào: {cm_file}")

# 3. Phân tích lỗi (Error Analysis)
# Tìm các câu bị dự đoán sai
errors = []
for idx, (true_label_id, pred_label_id) in enumerate(zip(y_test, y_test_pred)):
    if true_label_id != pred_label_id:
        errors.append({
            "review": df_test.iloc[idx]['review'], # Lấy nội dung review
            "true_label": id2label[true_label_id],
            "pred_label": id2label[pred_label_id]
        })

error_df = pd.DataFrame(errors)
error_file = os.path.join(OUTPUT_DIR, 'error_analysis_test.csv')
error_df.to_csv(error_file, index=False, encoding='utf-8-sig')
print(f"Đã lưu {len(error_df)} câu bị sai vào: {error_file}")

# =========================================================
# 6. LƯU CÁC FILE CẦN THIẾT
# =========================================================
# Lưu model và vectorizer để dùng sau này (Inference)
joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'xgboost_best_model.pkl'))
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl'))
joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))

print("\nHoàn tất toàn bộ pipeline!")