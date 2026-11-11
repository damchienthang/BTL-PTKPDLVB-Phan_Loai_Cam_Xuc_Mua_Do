import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    # 1. NẠP DỮ LIỆU
    data_path = os.path.join("Data", "final_labeled_clean.xlsx")
    if not os.path.exists(data_path):
        data_path = os.path.join("Data", "final_labeled_clean_v3.xlsx")

    print(f"[1] Đọc dữ liệu từ: {data_path}")
    df = pd.read_excel(data_path)

    text_col = 'Text_TFIDF' if 'Text_TFIDF' in df.columns else 'Clean Text'
    texts = np.array(df[text_col].fillna("đánh giá phim").astype(str).tolist())
    labels = np.array(df['Final Label'].astype(str).tolist())
    comment_ids = np.array(df['Comment ID'].tolist())

    # 2. PHÂN CHIA TẬP TRAIN (80%) VÀ TEST (20%) CHỐNG RÒ RỈ DỮ LIỆU
    train_texts, test_texts, train_labels, test_labels, train_ids, test_ids = train_test_split(
        texts, labels, comment_ids,
        test_size=0.2,
        stratify=labels,
        random_state=18
    )

    print(f"\n[2] Phân chia Train/Test (Stratified 80/20):")
    print(f"    - Tập Train: {len(train_texts)} mẫu")
    print(f"    - Tập Test : {len(test_texts)} mẫu")

    # 3. VECTOR HÓA BẰNG BAG-OF-WORDS (CountVectorizer)
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=2
    )

    # CHỈ fit trên tập Train để tránh rò rỉ dữ liệu (Anti-Data Leakage)
    X_train_bow = vectorizer.fit_transform(train_texts)
    # CHỈ transform trên tập Test dựa trên từ điển đã học từ Train
    X_test_bow = vectorizer.transform(test_texts)

    vocab = vectorizer.get_feature_names_out()
    sparsity_train = 100.0 * (1.0 - X_train_bow.nnz / (X_train_bow.shape[0] * X_train_bow.shape[1]))

    print(f"\n[3] Kết quả Vector hóa BoW:")
    print(f"    * Số đặc trưng từ vựng học từ Train : {len(vocab):,} n-grams")
    print(f"    * Kích thước ma trận Train           : {X_train_bow.shape}")
    print(f"    * Kích thước ma trận Test            : {X_test_bow.shape}")
    print(f"    * Độ thưa ma trận Train (Sparsity)  : {sparsity_train:.2f}%")

    # 4. ĐÓNG GÓI & LƯU FILE BÀN GIAO
    output_dir = os.path.join("Data", "vectorized")
    os.makedirs(output_dir, exist_ok=True)

    deliverable = {
        'X_train': X_train_bow,
        'y_train': train_labels,
        'train_texts': train_texts,
        'train_ids': train_ids,
        'X_test': X_test_bow,
        'y_test': test_labels,
        'test_texts': test_texts,
        'test_ids': test_ids,
        'vectorizer': vectorizer,
        'feature_names': vocab.tolist()
    }

    output_path = os.path.join(output_dir, "bow_train_test.joblib")
    joblib.dump(deliverable, output_path)
    print(f"\n[4] Đã lưu bộ dữ liệu bàn giao tại: {output_path}")


if __name__ == "__main__":
    main()
