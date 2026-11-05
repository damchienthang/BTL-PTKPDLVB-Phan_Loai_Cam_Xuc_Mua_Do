"""
Chương trình: Trích chọn đặc trưng & Vector hóa văn bản (TF-IDF)
Dự án: Phân tích cảm xúc bình luận phim "Mưa đỏ" (Đề tài 3A)
Người thực hiện & Thuyết trình: Thắng (Nhiệm vụ 2)

Mục tiêu:
Chuyển đổi dữ liệu văn bản sạch thành ma trận véc-tơ số học (TF-IDF)
để chuyển giao trực tiếp cho Hiếu (Naive Bayes) và Doanh (SVM) huấn luyện.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Đảm bảo in tiếng Việt chuẩn trên terminal Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("=" * 65)
    print(" BÁO CÁO THỰC NGHIỆM: TRÍCH CHỌN ĐẶC TRƯNG & VECTOR HÓA VĂN BẢN")
    print("=" * 65)

    # -------------------------------------------------------------
    # BƯỚC 1: ĐỌC DỮ LIỆU ĐÃ LÀM SẠCH
    # -------------------------------------------------------------
    data_path = os.path.join("Data", "final_labeled_clean_v3.xlsx")

    print(f"\n[Bước 1] Nạp dữ liệu từ: {data_path}")
    df = pd.read_excel(data_path)
    
    # Lấy 2 cột cốt lõi: Văn bản sạch và Nhãn cảm xúc
    texts = df['Clean Text'].fillna("đánh giá phim").astype(str).values
    labels = df['Final Label'].astype(str).values
    comment_ids = df['Comment ID'].values

    print(f"  -> Tổng số bình luận: {len(texts)} mẫu")
    print(f"  -> Phân bố cảm xúc: Positive={sum(labels == 'Positive')}, Neutral={sum(labels == 'Neutral')}, Negative={sum(labels == 'Negative')}")

    # -------------------------------------------------------------
    # BƯỚC 2: VECTOR HÓA BẰNG PHƯƠNG PHÁP TF-IDF
    # -------------------------------------------------------------
    print("\n[Bước 2] Thực hiện Vector hóa bằng mô hình TF-IDF...")
    print("  -> Cấu hình: n-gram=(1, 2) [từ đơn + cụm từ], min_df=2 [lọc từ quá hiếm]")
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    # Học từ vựng và biến đổi toàn bộ bình luận thành ma trận số
    X_tfidf = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()

    # Tính độ thưa (Sparsity) của ma trận
    sparsity = 100.0 * (1.0 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]))

    print(f"\n[KẾT QUẢ VECTOR HÓA]")
    print(f"  * Kích thước ma trận (Số câu x Số đặc trưng): {X_tfidf.shape}")
    print(f"  * Tổng số đặc trưng từ vựng n-gram trích xuất: {len(vocab):,} từ/cụm từ")
    print(f"  * Độ thưa của ma trận đặc trưng: {sparsity:.2f}% (dạng ma trận nén CSR)")

    # -------------------------------------------------------------
    # BƯỚC 3: TOP ĐẶC TRƯNG TIÊU BIỂU
    # -------------------------------------------------------------
    print("\n[Bước 3] Top 15 đặc trưng có trọng số TF-IDF trung bình cao nhất:")
    mean_scores = np.asarray(X_tfidf.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1][:15]
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank:2d}. {vocab[idx]:20s} : {mean_scores[idx]:.4f}")

    # -------------------------------------------------------------
    # BƯỚC 4: VÍ DỤ MINH HỌA TRỰC QUAN
    # -------------------------------------------------------------
    print("\n[Bước 4] Ví dụ trực quan: Cách 1 bình luận chuyển thành véc-tơ số:")
    sample_idx = 0
    sample_text = texts[sample_idx]
    sample_row = X_tfidf[sample_idx].tocoo()
    
    print(f"  - Bình luận mẫu: \"{sample_text}\" (Nhãn: {labels[sample_idx]})")
    print(f"  - Các đặc trưng khác 0 trong véc-tơ của câu này:")
    for col, val in zip(sample_row.col, sample_row.data):
        print(f"      + Từ: '{vocab[col]}' -> Trọng số TF-IDF: {val:.4f}")

    # -------------------------------------------------------------
    # BƯỚC 5: LƯU FILE
    # -------------------------------------------------------------
    output_dir = os.path.join("Data", "vectorized")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Gói dữ liệu hoàn chỉnh
    deliverable = {
        'comment_ids': np.array(comment_ids),
        'texts': np.array(texts, dtype=str),
        'labels': np.array(labels, dtype=str),
        'X_tfidf': X_tfidf,
        'vectorizer': vectorizer,
        'feature_names': vocab.tolist()
    }
    deliverable_path = os.path.join(output_dir, "vectorized_data.joblib")
    joblib.dump(deliverable, deliverable_path)

    # 2. Lưu riêng vectorizer để dùng cho việc phân loại câu mới
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))

    # 3. Xuất bảng ma trận mẫu (20 câu đầu x 10 từ đầu)
    sample_matrix_df = pd.DataFrame(
        X_tfidf[:20, top_indices[:10]].toarray(),
        columns=[vocab[i] for i in top_indices[:10]]
    )
    sample_matrix_df.insert(0, "Label", labels[:20])
    sample_matrix_df.insert(0, "Clean Text", texts[:20])
    sample_matrix_df.to_excel(os.path.join(output_dir, "sample_tfidf_matrix_for_slides.xlsx"), index=False)

    print(f"\n[Bước 5] Hoàn tất bàn giao sản phẩm:")
    print(f"  -> File cho Hiếu & Doanh nạp train : {deliverable_path}")
    print(f"  -> File mô hình Vectorizer          : {os.path.join(output_dir, 'tfidf_vectorizer.joblib')}")
    print(f"  -> File bảng số liệu mẫu chiếu slide: {os.path.join(output_dir, 'sample_tfidf_matrix_for_slides.xlsx')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
