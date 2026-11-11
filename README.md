# NHIỆM VỤ 2: TRÍCH CHỌN ĐẶC TRƯNG & VECTOR HÓA DỮ LIỆU
## Đề tài 3A: Phân tích cảm xúc bình luận phim "Trại buôn người" trên Mạng xã hội
**Môn học:** Phân tích và Khai phá dữ liệu văn bản  

---

## 🎯 Mục tiêu nhiệm vụ
1. Tiếp nhận dữ liệu bình luận phim **"Trại buôn người"** đã làm sạch và gán nhãn ($1.307$ mẫu).
2. Thực hiện phân tầng tập **Train (80%)** và **Test (20%)** trước khi vector hóa nhằm chống rò rỉ dữ liệu (**Data Leakage**).
3. Chuyển đổi văn bản thành ma trận véc-tơ số học bằng 2 phương pháp:
   - **TF-IDF (Term Frequency - Inverse Document Frequency)** kết hợp Unigram & Bigram ($1\text{–}2\text{-gram}$).
   - **Bag-of-Words (BoW / CountVectorizer)** kết hợp Unigram & Bigram ($1\text{–}2\text{-gram}$).
4. Bàn giao gói ma trận đặc trưng Train/Test chuẩn cho các nhánh mô hình học máy (Naive Bayes, SVM).
5. Trực quan hóa đặc trưng bằng biểu đồ phục vụ thuyết trình và báo cáo.

---

## 📁 Cấu trúc thư mục

```text
.
├── Data/
│   ├── final_labeled_clean.xlsx                     # Dữ liệu sạch đầu vào (1.307 mẫu)
│   ├── final_labeled_clean_v3.xlsx                  # Dữ liệu nguồn Facebook (2.087 mẫu)
│   └── vectorized/
│       ├── tfidf_train_test.joblib                  # Gói bàn giao ma trận TF-IDF (Train & Test)
│       └── bow_train_test.joblib                    # Gói bàn giao ma trận BoW (Train & Test)
├── src/
│   ├── vectorization.py                             # Mã nguồn vector hóa TF-IDF (chuẩn Train/Test)
│   ├── vectorization_bow.py                         # Mã nguồn vector hóa BoW (chuẩn Train/Test)
│   ├── plot_features.py                             # Mã nguồn trực quan hóa đặc trưng TF-IDF
│   ├── BoW_SVM.ipynb                                # Huấn luyện mô hình SVM với BoW
│   └── TF_IDF_SVM.ipynb                             # Huấn luyện mô hình SVM với TF-IDF
├── models/                                          # Thư mục chứa mô hình huấn luyện
├── reports/
│   ├── Quy_chuan_Tien_xu_ly_va_Dinh_dang_Du_lieu_NLP.docx # Tài liệu quy chuẩn kỹ thuật tiền xử lý
│   └── figures/
│       ├── top_20_tfidf_overall.png                 # Biểu đồ Top 20 đặc trưng TF-IDF
│       └── top_features_by_sentiment.png            # Biểu đồ Top 15 đặc trưng theo 3 nhãn cảm xúc
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Hướng dẫn chạy chương trình

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Thực hiện Vector hóa văn bản
* **Vector hóa bằng TF-IDF:**
  ```bash
  python src/vectorization.py
  ```
* **Vector hóa bằng Bag-of-Words (BoW):**
  ```bash
  python src/vectorization_bow.py
  ```

### 3. Xuất biểu đồ trực quan hóa đặc trưng
```bash
python src/plot_features.py
```

---

## 📊 Kết quả thực nghiệm

* **Tổng số bình luận:** $1.307$ mẫu (Positive: $853$, Neutral: $260$, Negative: $194$).
* **Phân chia tập dữ liệu:** Train: $1.045$ mẫu ($80\%$), Test: $262$ mẫu ($20\%$).
* **Cấu hình mô hình:** `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True` (học từ vựng chỉ trên tập Train).
* **Số lượng đặc trưng từ vựng:** $2.322$ đặc trưng n-gram.
* **Kích thước ma trận:**
  - Tập Train: `(1045, 2322)`
  - Tập Test: `(262, 2322)`
* **Độ thưa ma trận Train (Sparsity):** $99.36\%$
