# NHIỆM VỤ 2: TRÍCH CHỌN ĐẶC TRƯNG & VECTOR HÓA DỮ LIỆU
## Đề tài 3A: Phân tích cảm xúc bình luận phim "Trại buôn người" trên Mạng xã hội (Facebook & TikTok)
**Môn học:** Phân tích và Khai phá dữ liệu văn bản  

---

## 🎯 Mục tiêu nhiệm vụ
1. Tiếp nhận và xử lý song song cả 2 nguồn dữ liệu bình luận phim **"Trại buôn người"**:
   - **Tập dữ liệu Facebook:** $2.087$ mẫu (`Data/final_labeled_clean_v3.xlsx`).
   - **Tập dữ liệu TikTok:** $1.307$ mẫu (`Data/final_labeled_clean.xlsx`).
2. Thực hiện phân tầng tập **Train (80%)** và **Test (20%)** độc lập cho từng bộ nhằm triệt tiêu lỗi rò rỉ dữ liệu (**Data Leakage**).
3. Chuyển đổi văn bản thành ma trận véc-tơ số học bằng 2 phương pháp:
   - **TF-IDF (Term Frequency - Inverse Document Frequency)** kết hợp Unigram & Bigram ($1\text{–}2\text{-gram}$).
   - **Bag-of-Words (BoW / CountVectorizer)** kết hợp Unigram & Bigram ($1\text{–}2\text{-gram}$).
4. So sánh toàn diện đặc trưng ngôn ngữ, độ dài bình luận và không gian từ vựng giữa **Facebook vs TikTok**.
5. Bàn giao gói ma trận đặc trưng chuẩn hóa và tài liệu quy chuẩn kỹ thuật cho các thành viên nhánh mô hình.

---

## 📁 Cấu trúc thư mục

```text
.
├── Data/
│   ├── final_labeled_clean.xlsx                     # Dữ liệu sạch TikTok (1.307 mẫu)
│   ├── final_labeled_clean_v3.xlsx                  # Dữ liệu sạch Facebook (2.087 mẫu)
│   └── vectorized/
│       ├── tiktok_tfidf_train_test.joblib           # Ma trận TF-IDF TikTok (Train & Test)
│       ├── tiktok_bow_train_test.joblib             # Ma trận BoW TikTok (Train & Test)
│       ├── facebook_tfidf_train_test.joblib         # Ma trận TF-IDF Facebook (Train & Test)
│       └── facebook_bow_train_test.joblib           # Ma trận BoW Facebook (Train & Test)
├── src/
│   ├── vectorization.py                             # Vector hóa TF-IDF cho cả 2 bộ (chuẩn Train/Test)
│   ├── vectorization_bow.py                         # Vector hóa BoW cho cả 2 bộ (chuẩn Train/Test)
│   ├── compare_datasets.py                          # So sánh dữ liệu & đặc trưng Facebook vs TikTok
│   ├── plot_features.py                             # Trực quan hóa đặc trưng TF-IDF
│   ├── BoW_SVM.ipynb                                # Huấn luyện mô hình SVM với BoW
│   └── TF_IDF_SVM.ipynb                             # Huấn luyện mô hình SVM với TF-IDF
├── models/                                          # Thư mục chứa mô hình huấn luyện
├── reports/
│   ├── Quy_chuan_Tien_xu_ly_va_Dinh_dang_Du_lieu_NLP.docx # Tài liệu quy chuẩn kỹ thuật tiền xử lý
│   └── figures/
│       ├── comparison_label_distribution.png        # Biểu đồ so sánh phân bố nhãn FB vs TT
│       ├── comparison_top_features_fb_vs_tt.png     # Biểu đồ so sánh Top đặc trưng TF-IDF FB vs TT
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

### 2. Thực hiện Vector hóa cho cả 2 bộ dữ liệu (Facebook & TikTok)
* **Vector hóa bằng TF-IDF:**
  ```bash
  python src/vectorization.py
  ```
* **Vector hóa bằng Bag-of-Words (BoW):**
  ```bash
  python src/vectorization_bow.py
  ```

### 3. Chạy báo cáo & xuất biểu đồ đối sánh dữ liệu
```bash
python src/compare_datasets.py
```

---

## 📊 Bảng đối sánh dữ liệu: Facebook vs TikTok

| Tiêu chí đối sánh | Facebook (`final_labeled_clean_v3.xlsx`) | TikTok (`final_labeled_clean.xlsx`) | Nhận xét chuyên môn |
| :--- | :---: | :---: | :--- |
| **Quy mô tập dữ liệu** | **2.087 mẫu** | **1.307 mẫu** | Cả 2 đều đủ lớn cho học máy giám sát |
| **Tỷ lệ Positive** | $1.280$ mẫu ($61.33\%$) | $853$ mẫu ($65.26\%$) | Khán giả cả 2 nền tảng đều khen phim là chủ đạo |
| **Tỷ lệ Neutral** | $438$ mẫu ($20.99\%$) | $260$ mẫu ($19.89\%$) | Bình luận hỏi đáp, rủ bạn bè đi xem |
| **Tỷ lệ Negative** | $369$ mẫu ($17.68\%$) | $194$ mẫu ($14.84\%$) | Bình luận chê kịch bản, diễn xuất |
| **Độ dài câu trung bình** | **$18.47$ từ** (Max: $417$ từ) | **$14.04$ từ** (Max: $224$ từ) | Facebook bàn luận sâu hơn, TikTok ngắn gọn & trực diện |
| **Số đặc trưng n-grams** | **$5.872$ n-grams** | **$2.880$ n-grams** | Không gian từ vựng TikTok cô đọng, ít từ hiếm |
| **Từ vựng giao thoa** | \multicolumn{2}{c|}{**$1.540$ n-grams chung ($21.4\%$)**} | Tập trung vào các từ cốt lõi: *hay, phim, xem, đi, coi* |
