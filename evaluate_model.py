# evaluate_model.py
import json
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

# seaborn là optional (chỉ để vẽ heatmap cho đẹp)
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Cảnh báo: Không tìm thấy seaborn, sẽ vẽ confusion matrix bằng matplotlib thuần.")

# --- CẤU HÌNH THAM SỐ PHẢI GIỐNG LÚC TRAIN ---
CONFIG = {
    'max_length': 120,
    'trunc_type': 'post',
    'padding_type': 'pre',
    'model_path': 'best_gru_model.h5',
    'tokenizer_path': 'tokenizer.pickle',
    'encoder_path': 'label_encoder.pickle',
    'test_file': 'test-preprocessed.json',
    'cm_image_path': 'confusion_matrix.png'  # file ảnh sẽ được lưu
}


def load_data_from_json(filepath):
    """Đọc file JSON và lấy về list reviews + labels."""
    if not os.path.exists(filepath):
        print(f"Lỗi: Không tìm thấy file {filepath}")
        return [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    reviews = [item.get('review', '') for item in data]
    labels = [item.get('sentiment', '') for item in data]
    return reviews, labels


def main():
    # 1. LOAD DỮ LIỆU TEST
    print(">>> 1. Đang tải dữ liệu test...")
    test_reviews, test_labels = load_data_from_json(CONFIG['test_file'])

    if not test_reviews or not test_labels:
        print("Lỗi: Dữ liệu test rỗng hoặc file không tồn tại.")
        return

    # 2. LOAD TOKENIZER
    print(">>> 2. Đang load tokenizer...")
    if not os.path.exists(CONFIG['tokenizer_path']):
        print(f"Lỗi: Không tìm thấy file tokenizer tại {CONFIG['tokenizer_path']}")
        return

    with open(CONFIG['tokenizer_path'], 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Biến text test thành seq + padding
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(test_reviews),
        maxlen=CONFIG['max_length'],
        padding=CONFIG['padding_type'],
        truncating=CONFIG['trunc_type']
    )

    # 3. LOAD LABEL ENCODER
    print(">>> 3. Đang load label encoder...")
    if not os.path.exists(CONFIG['encoder_path']):
        print(f"Lỗi: Không tìm thấy file label encoder tại {CONFIG['encoder_path']}")
        return

    with open(CONFIG['encoder_path'], 'rb') as handle:
        label_encoder = pickle.load(handle)

    # Biến nhãn text thành số
    y_true = label_encoder.transform(test_labels)
    class_names = label_encoder.classes_

    # 4. LOAD MODEL ĐÃ TRAIN
    print(">>> 4. Đang load model đã train...")
    if not os.path.exists(CONFIG['model_path']):
        print(f"Lỗi: Không tìm thấy file model tại {CONFIG['model_path']}")
        return

    model = tf.keras.models.load_model(CONFIG['model_path'])
    print("    Đã load model thành công.")

    # 5. DỰ ĐOÁN TRÊN TẬP TEST
    print(">>> 5. Đang dự đoán trên tập test...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 6. IN CLASSIFICATION REPORT
    print("\n>>> 6. Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # 7. VẼ VÀ LƯU CONFUSION MATRIX
    print(">>> 7. Vẽ và lưu Confusion Matrix...")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
    else:
        # Vẽ thủ công nếu không có seaborn
        im = ax.imshow(cm)
        ax.figure.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # ghi số lên từng ô
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    # Lưu ảnh
    plt.savefig(CONFIG['cm_image_path'], dpi=300)
    plt.close(fig)

    print(f"    Đã lưu ảnh confusion matrix tại: {CONFIG['cm_image_path']}")


if __name__ == '__main__':
    main()
