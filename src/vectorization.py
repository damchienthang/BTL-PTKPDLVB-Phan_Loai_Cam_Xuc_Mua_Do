import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def vectorize_dataset(name, file_path, output_prefix):
    print(f"\n{'='*70}")
    print(f" VECTOR HÓA TF-IDF: BỘ DỮ LIỆU {name.upper()}")
    print(f"{'='*70}")

    if not os.path.exists(file_path):
        print(f"[LỖI] Không tìm thấy file: {file_path}")
        return None

    df = pd.read_excel(file_path)
    text_col = 'Text_TFIDF' if 'Text_TFIDF' in df.columns else 'Clean Text'
    texts = np.array(df[text_col].fillna("đánh giá phim").astype(str).tolist())
    labels = np.array(df['Final Label'].astype(str).tolist())
    comment_ids = np.array(df['Comment ID'].tolist())

    print(f"[1] Nạp dữ liệu: {len(texts)} mẫu từ {file_path}")
    print("    Phân bố nhãn:")
    for lbl, count in pd.Series(labels).value_counts().items():
        print(f"      + {lbl:8s}: {count:4d} ({count/len(labels)*100:.2f}%)")

    # Chia tập Train (80%) và Test (20%) phân tầng
    train_texts, test_texts, train_labels, test_labels, train_ids, test_ids = train_test_split(
        texts, labels, comment_ids,
        test_size=0.2,
        stratify=labels,
        random_state=18
    )

    print(f"[2] Phân chia Train/Test: Train={len(train_texts)} | Test={len(test_texts)}")

    # Vector hóa bằng TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    vocab = vectorizer.get_feature_names_out()
    sparsity = 100.0 * (1.0 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]))

    print(f"[3] Kết quả TF-IDF:")
    print(f"    * Số lượng từ vựng n-grams : {len(vocab):,}")
    print(f"    * Kích thước ma trận Train : {X_train.shape}")
    print(f"    * Kích thước ma trận Test  : {X_test.shape}")
    print(f"    * Độ thưa (Sparsity)       : {sparsity:.2f}%")

    output_dir = os.path.join("Data", "vectorized")
    os.makedirs(output_dir, exist_ok=True)

    deliverable = {
        'name': name,
        'X_train': X_train,
        'y_train': train_labels,
        'train_texts': train_texts,
        'train_ids': train_ids,
        'X_test': X_test,
        'y_test': test_labels,
        'test_texts': test_texts,
        'test_ids': test_ids,
        'vectorizer': vectorizer,
        'feature_names': vocab.tolist()
    }

    out_file = os.path.join(output_dir, f"{output_prefix}_tfidf_train_test.joblib")
    joblib.dump(deliverable, out_file)
    print(f"[4] Đã lưu file: {out_file}")

    # Đồng thời lưu file mặc định nếu là TikTok
    if output_prefix == "tiktok":
        default_file = os.path.join(output_dir, "tfidf_train_test.joblib")
        joblib.dump(deliverable, default_file)

    return deliverable


def main():
    # 1. Bộ dữ liệu TikTok (1.307 mẫu)
    vectorize_dataset(
        name="TikTok",
        file_path=os.path.join("Data", "final_labeled_clean.xlsx"),
        output_prefix="tiktok"
    )

    # 2. Bộ dữ liệu Facebook (2.087 mẫu)
    vectorize_dataset(
        name="Facebook",
        file_path=os.path.join("Data", "final_labeled_clean_v3.xlsx"),
        output_prefix="facebook"
    )


if __name__ == "__main__":
    main()
