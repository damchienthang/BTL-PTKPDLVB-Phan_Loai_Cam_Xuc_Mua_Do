"""
File: demo_load_data.py
Hướng dẫn dành cho Hiếu (Naive Bayes) và Doanh (SVM):
Nạp dữ liệu đã vector hóa từ Thắng, tự chia tập Train/Test theo ý muốn và huấn luyện.
"""

import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("=" * 60)
    print("HƯỚNG DẪN HIẾU VÀ DOANH SỬ DỤNG DỮ LIỆU ĐÃ VECTOR HÓA")
    print("=" * 60)

    # 1. Nạp file dữ liệu đã vector hóa từ Thắng
    data = joblib.load("Data/vectorized/vectorized_data.joblib")
    X = data['X_tfidf']                    # Ma trận đặc trưng TF-IDF (2087 x 5872)
    y = np.array(data['labels'], dtype=str) # Nhãn cảm xúc (Positive, Neutral, Negative)
    texts = np.array(data['texts'], dtype=str) # Câu văn bản gốc đã làm sạch

    print(f"1. Đã nạp thành công ma trận đặc trưng: {X.shape}")
    print(f"   Tổng số nhãn: {len(y)}")

    # 2. Hiếu và Doanh tự chia tập Train / Test theo tỷ lệ mong muốn (ví dụ 80/20)
    print("\n2. Chia tập Train/Test (80% Train, 20% Test):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   - Tập Train: {X_train.shape[0]} mẫu")
    print(f"   - Tập Test : {X_test.shape[0]} mẫu")

    # 3. Demo nhanh cho HIẾU (Naive Bayes)
    print("\n3. Demo mô hình Naive Bayes (Phần việc của Hiếu):")
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print(f"   -> Accuracy Naive Bayes: {accuracy_score(y_test, y_pred_nb)*100:.2f}%")

    # 4. Demo nhanh cho DOANH (SVM)
    print("\n4. Demo mô hình SVM (Phần việc của Doanh):")
    svm = LinearSVC(C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print(f"   -> Accuracy Linear SVM : {accuracy_score(y_test, y_pred_svm)*100:.2f}%")
    print("\nBáo cáo phân loại mẫu của SVM:")
    print(classification_report(y_test, y_pred_svm))

if __name__ == "__main__":
    main()
