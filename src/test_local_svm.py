"""
Script: test_local_svm.py
Mục đích: Chạy thử nghiệm local 2 mô hình SVM mới được pull về từ Git:
          1. BoW + SVM (models/bow_svm/best_bow_svm.joblib)
          2. TF-IDF + SVM (models/bow_tfidf/best_tfidf_svm.joblib)
Đánh giá trên tập kiểm thử (Test Set) của dự án phim "Trại buôn người".
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=" * 70)
    print(" KIỂM THỬ LOCAL: ĐÁNH GIÁ 2 MÔ HÌNH SVM ĐƯỢC PULL VỀ TỪ GIT")
    print("=" * 70)

    # 1. Đọc dữ liệu kiểm thử
    data_path = os.path.join("Data", "final_labeled_clean_v3.xlsx")
    if not os.path.exists(data_path):
        data_path = "final_labeled_clean_v3.xlsx"
    
    print(f"\n[1] Đang nạp dữ liệu từ: {data_path}")
    df = pd.read_excel(data_path)
    df = df[["Clean Text", "Final Label"]].rename(columns={"Clean Text": "text", "Final Label": "label"})
    df["text"] = df["text"].fillna("đánh giá phim").astype(str)

    # Chia tập dữ liệu đúng như cấu hình trong notebook của Doanh (test_size=0.2, random_state=18, stratify=y)
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=18
    )

    print(f"    - Tập Train: {len(X_train)} mẫu")
    print(f"    - Tập Test : {len(X_test)} mẫu")
    print(f"    - Phân bố nhãn tập Test:\n{y_test.value_counts()}")

    # 2. Đánh giá mô hình 1: BoW + SVM
    print("\n" + "=" * 70)
    print(" [2] ĐÁNH GIÁ MÔ HÌNH 1: Bag-of-Words (BoW) + Linear SVM (OvO)")
    print("=" * 70)
    bow_model_path = os.path.join("models", "bow_svm", "best_bow_svm.joblib")
    bow_model = joblib.load(bow_model_path)
    print(f"    Đã nạp pipeline: {bow_model_path}")
    
    y_pred_bow = bow_model.predict(X_test)
    acc_bow = accuracy_score(y_test, y_pred_bow)
    print(f"    -> ĐỘ CHÍNH XÁC (ACCURACY) TRÊN TẬP TEST: {acc_bow * 100:.2f}%\n")
    print("    Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred_bow))

    # 3. Đánh giá mô hình 2: TF-IDF + SVM
    print("=" * 70)
    print(" [3] ĐÁNH GIÁ MÔ HÌNH 2: TF-IDF + Linear SVM (OvO)")
    print("=" * 70)
    tfidf_model_path = os.path.join("models", "bow_tfidf", "best_tfidf_svm.joblib")
    tfidf_model = joblib.load(tfidf_model_path)
    print(f"    Đã nạp pipeline: {tfidf_model_path}")

    y_pred_tfidf = tfidf_model.predict(X_test)
    acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
    print(f"    -> ĐỘ CHÍNH XÁC (ACCURACY) TRÊN TẬP TEST: {acc_tfidf * 100:.2f}%\n")
    print("    Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred_tfidf))

    # 4. So sánh đối đầu giữa BoW và TF-IDF
    print("=" * 70)
    print(" [4] BẢNG SO SÁNH HIỆU SUẤT TRÊN TẬP TEST")
    print("=" * 70)
    compare_df = pd.DataFrame({
        "Mô hình": ["BoW + LinearSVC (OvO)", "TF-IDF + LinearSVC (OvO)"],
        "Accuracy": [f"{acc_bow*100:.2f}%", f"{acc_tfidf*100:.2f}%"],
        "F1 Negative": [
            f"{classification_report(y_test, y_pred_bow, output_dict=True)['Negative']['f1-score']*100:.2f}%",
            f"{classification_report(y_test, y_pred_tfidf, output_dict=True)['Negative']['f1-score']*100:.2f}%"
        ],
        "F1 Neutral": [
            f"{classification_report(y_test, y_pred_bow, output_dict=True)['Neutral']['f1-score']*100:.2f}%",
            f"{classification_report(y_test, y_pred_tfidf, output_dict=True)['Neutral']['f1-score']*100:.2f}%"
        ],
        "F1 Positive": [
            f"{classification_report(y_test, y_pred_bow, output_dict=True)['Positive']['f1-score']*100:.2f}%",
            f"{classification_report(y_test, y_pred_tfidf, output_dict=True)['Positive']['f1-score']*100:.2f}%"
        ]
    })
    print(compare_df.to_string(index=False))

    # 5. Thử nghiệm dự đoán bình luận mới thực tế
    print("\n" + "=" * 70)
    print(" [5] THỬ NGHIỆM DỰ ĐOÁN TRỰC TIẾP TRÊN BÌNH LUẬN MỚI")
    print("=" * 70)
    test_comments = [
        "Phim Trại buôn người quá xuất sắc, cảnh đánh nhau xem nổi da gà luôn, 10 điểm!",
        "Phim làm vội, kịch bản nhiều sạn, dở tệ không đáng tiền vé xem tí nào",
        "Có ai biết phim này chiếu rạp đến ngày bao nhiêu không mọi người nhỉ?",
        "Tạm ổn, diễn viên đóng tròn vai nhưng cái kết hơi hụt hẫng một chút"
    ]

    preds_bow = bow_model.predict(test_comments)
    preds_tfidf = tfidf_model.predict(test_comments)

    for i, c in enumerate(test_comments, 1):
        print(f"\n{i}. Bình luận: \"{c}\"")
        print(f"   -> BoW   dự đoán: {preds_bow[i-1]}")
        print(f"   -> TF-IDF dự đoán: {preds_tfidf[i-1]}")

    print("\n" + "=" * 70)
    print(" KẾT QUẢ KIỂM THỬ LOCAL HOÀN TẤT THÀNH CÔNG!")
    print("=" * 70)

if __name__ == "__main__":
    main()

