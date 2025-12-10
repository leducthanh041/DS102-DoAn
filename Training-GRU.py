import json
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # Nếu muốn vẽ biểu đồ đẹp (pip install seaborn)

# --- CẤU HÌNH THAM SỐ (Đã tối ưu) ---
CONFIG = {
    'vocab_size': 20000,       # Tăng lên 20k từ để bắt được nhiều từ vựng hơn
    'embedding_dim': 200,      # Tăng lên 200 để vector ngữ nghĩa dày hơn
    'max_length': 120,
    'trunc_type': 'post',
    'padding_type': 'pre',     # QUAN TRỌNG: Đổi sang 'pre' cho GRU
    'oov_tok': '<OOV>',
    'batch_size': 32,
    'epochs': 10,              # Đặt 100, nhưng sẽ dừng sớm nếu cần
    'patience': 10,            # Kiên nhẫn 10 epoch không tăng mới dừng (tránh dừng ở epoch 5)
    'gru_units': 128,          # Tăng số nơ-ron để mô hình "thông minh" hơn
    'dropout_rate': 0.4,
    'model_path': 'best_gru_model.h5',
    'tokenizer_path': 'tokenizer.pickle',
    'encoder_path': 'label_encoder.pickle',
    'train_file': 'train-preprocessed.json',
    'test_file': 'test-preprocessed.json'
}

def load_data_from_json(filepath):
    if not os.path.exists(filepath):
        print(f"Lỗi: Không tìm thấy file {filepath}")
        return [], []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    reviews = [item.get('review', '') for item in data]
    labels = [item.get('sentiment', '') for item in data]
    return reviews, labels

def create_model(vocab_size, embedding_dim, max_length, num_classes):
    model = Sequential([
        # SỬA LỖI UNBUILT: Khai báo kích thước đầu vào rõ ràng
        Input(shape=(max_length,)),
        
        Embedding(vocab_size, embedding_dim),
        
        # Mô hình mạnh hơn với 128 units
        Bidirectional(GRU(CONFIG['gru_units'], return_sequences=False)),
        
        Dropout(CONFIG['dropout_rate']),
        
        Dense(64, activation='relu'), # Lớp trung gian dày hơn chút
        Dropout(0.3),                 # Thêm dropout phụ
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

def main():
    # 1. LOAD DATA
    print(">>> 1. Đang tải dữ liệu...")
    train_reviews, train_labels = load_data_from_json(CONFIG['train_file'])
    test_reviews, test_labels = load_data_from_json(CONFIG['test_file'])

    if not train_reviews: return

    # 2. XỬ LÝ NHÃN & TÍNH TRỌNG SỐ (CLASS WEIGHTS)
    print(">>> 2. Đang xử lý nhãn và tính toán trọng số...")
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(train_labels)
    y_test_enc = label_encoder.transform(test_labels)

    # Tính class weights để cân bằng dữ liệu (Sửa lỗi Accuracy thấp do lệch mẫu)
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_enc),
        y=y_train_enc
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print(f"   Trọng số các lớp (Class Weights): {class_weights_dict}")

    # Chuyển sang One-hot
    num_classes = len(np.unique(y_train_enc))
    y_train = to_categorical(y_train_enc, num_classes)
    y_test = to_categorical(y_test_enc, num_classes)

    # Lưu Label Encoder
    with open(CONFIG['encoder_path'], 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. TOKENIZER & PADDING
    print(">>> 3. Đang mã hóa văn bản...")
    tokenizer = Tokenizer(num_words=CONFIG['vocab_size'], oov_token=CONFIG['oov_tok'])
    tokenizer.fit_on_texts(train_reviews)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_reviews), 
                            maxlen=CONFIG['max_length'], 
                            padding=CONFIG['padding_type'], 
                            truncating=CONFIG['trunc_type'])
    
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_reviews), 
                           maxlen=CONFIG['max_length'], 
                           padding=CONFIG['padding_type'], 
                           truncating=CONFIG['trunc_type'])

    # Lưu Tokenizer
    with open(CONFIG['tokenizer_path'], 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 4. BUILD MODEL
    print(">>> 4. Khởi tạo mô hình...")
    model = create_model(CONFIG['vocab_size'], CONFIG['embedding_dim'], 
                         CONFIG['max_length'], num_classes)
    model.summary() # Giờ sẽ hiện đầy đủ Param #

    # 5. TRAINING
    print(f">>> 5. Bắt đầu train (Max Epochs: {CONFIG['epochs']})...")
    
    # Callback
    early_stop = EarlyStopping(monitor='val_loss', patience=CONFIG['patience'], restore_best_weights=True)
    checkpoint = ModelCheckpoint(CONFIG['model_path'], monitor='val_accuracy', save_best_only=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=CONFIG['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        class_weight=class_weights_dict, # QUAN TRỌNG: Áp dụng trọng số
        verbose=1
    )

# --- BƯỚC 6: ĐÁNH GIÁ CHI TIẾT (ACCURACY & F1-SCORE) ---
    print("\n>>> 6. Đánh giá chi tiết mô hình:")
    
    # 1. Dự đoán trên tập test
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1) # Chuyển xác suất thành nhãn (0, 1, 2)
    y_true_classes = np.argmax(y_test, axis=1)       # Chuyển one-hot ground truth thành nhãn
    
    # 2. Lấy tên các lớp (Positive, Negative, Neutral)
    class_names = label_encoder.classes_
    
    # 3. Tính toán và in báo cáo (Classification Report)
    # Đây là bảng quan trọng nhất chứa Precision, Recall, F1
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print("\nBẢNG BÁO CÁO CHI TIẾT:")
    print(report)
    
    # 4. Tính F1-Score tổng thể
    f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')
    f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    print(f"Macro F1-Score: {f1_macro:.4f} (Trung bình cộng các lớp)")
    print(f"Weighted F1-Score: {f1_weighted:.4f} (Trung bình có trọng số theo số lượng mẫu)")
    
    # 5. (Tùy chọn) Vẽ Ma trận nhầm lẫn (Confusion Matrix) để xem nó hay nhầm lớp nào với lớp nào
    # Bạn có thể bỏ qua phần này nếu không cài seaborn
    try:
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Dự đoán (Predicted)')
        plt.ylabel('Thực tế (Actual)')
        plt.title('Confusion Matrix')
        plt.show()
    except Exception as e:
        print("Không thể vẽ biểu đồ (có thể do thiếu thư viện seaborn/matplotlib).")

    # 7. DEMO
    print("\n>>> 7. Demo dự đoán:")
    demo_sentences = [
        "phim noi dung qua chan xem phi thoi gian",
        "dich vu tuyet voi nhan vien nhiet tinh",
        "hang tam on khong co gi dac sac"
    ]
    
    for text in demo_sentences:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=CONFIG['max_length'], 
                               padding=CONFIG['padding_type'], 
                               truncating=CONFIG['trunc_type'])
        pred = model.predict(padded)
        label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        print(f"   '{text}' -> {label} ({np.max(pred)*100:.1f}%)")

if __name__ == '__main__':
    main()