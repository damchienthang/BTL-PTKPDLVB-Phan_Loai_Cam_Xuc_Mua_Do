import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


def main():
    data_path = os.path.join("Data", "final_labeled_clean_v3.xlsx")
    df = pd.read_excel(data_path)

    texts = np.array(df['Clean Text'].fillna("đánh giá phim").astype(str).tolist())
    labels = np.array(df['Final Label'].astype(str).tolist())
    comment_ids = np.array(df['Comment ID'].tolist())

    # Chia tập Train (80%) và Test (20%) phân tầng trước khi vector hóa
    train_texts, test_texts, train_labels, test_labels, train_ids, test_ids = train_test_split(
        texts, labels, comment_ids,
        test_size=0.2,
        stratify=labels,
        random_state=18
    )

    print(f"Tổng số mẫu dữ liệu: {len(texts)}")
    print(f"  - Tập Train: {len(train_texts)} mẫu")
    print(f"  - Tập Test : {len(test_texts)} mẫu")

    # Khởi tạo mô hình TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    # CHỈ fit trên tập Train để tránh rò rỉ dữ liệu (Data Leakage)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    # CHỈ transform trên tập Test dựa trên từ điển đã học từ Train
    X_test_tfidf = vectorizer.transform(test_texts)

    vocab = vectorizer.get_feature_names_out()

    print(f"\n[Kết quả Vector hóa TF-IDF]")
    print(f"  * Số lượng đặc trưng từ vựng (chỉ từ Train): {len(vocab)}")
    print(f"  * Kích thước ma trận Train: {X_train_tfidf.shape}")
    print(f"  * Kích thước ma trận Test : {X_test_tfidf.shape}")

    output_dir = os.path.join("Data", "vectorized")
    os.makedirs(output_dir, exist_ok=True)

    deliverable = {
        'X_train': X_train_tfidf,
        'y_train': train_labels,
        'train_texts': train_texts,
        'train_ids': train_ids,
        'X_test': X_test_tfidf,
        'y_test': test_labels,
        'test_texts': test_texts,
        'test_ids': test_ids,
        'vectorizer': vectorizer,
        'feature_names': vocab.tolist()
    }

    output_path = os.path.join(output_dir, "tfidf_train_test.joblib")
    joblib.dump(deliverable, output_path)
    print(f"\nĐã lưu bộ dữ liệu bàn giao tại: {output_path}")


if __name__ == "__main__":
    main()
