# NHIỆM VỤ 2: TRÍCH CHỌN ĐẶC TRƯNG & VECTOR HÓA DỮ LIỆU
## Đề tài 3A: Phân tích cảm xúc bình luận phim "Trại buôn người" (Facebook & TikTok)

---

## 🎯 Mục tiêu
1. Xử lý và vector hóa 2 nguồn dữ liệu: Facebook ($2.087$ mẫu) và TikTok ($1.307$ mẫu).
2. Phân chia Train ($80\%$) và Test ($20\%$) phân tầng trước khi vector hóa để chống rò rỉ dữ liệu.
3. Xuất ma trận đặc trưng TF-IDF và Bag-of-Words (BoW) kết hợp n-gram ($1\text{–}2\text{-gram}$).
4. So sánh dữ liệu và không gian đặc trưng giữa Facebook và TikTok.

---

## 📁 Cấu trúc thư mục

```text
.
├── Data/
│   ├── final_labeled_clean.xlsx                     # Dữ liệu sạch TikTok (1.307 mẫu)
│   ├── final_labeled_clean_v3.xlsx                  # Dữ liệu sạch Facebook (2.087 mẫu)
│   └── vectorized/
│       ├── tiktok_tfidf_train_test.joblib           # TF-IDF TikTok
│       ├── tiktok_bow_train_test.joblib             # BoW TikTok
│       ├── facebook_tfidf_train_test.joblib         # TF-IDF Facebook
│       └── facebook_bow_train_test.joblib           # BoW Facebook
├── src/
│   ├── vectorization.ipynb                          # Notebook luồng thực thi trực quan toàn diện
│   ├── compare_datasets.py                          # Script đối sánh dữ liệu FB vs TikTok
│   └── plot_features.py                             # Script xuất biểu đồ đặc trưng
├── reports/
│   ├── Quy_chuan_Tien_xu_ly_va_Dinh_dang_Du_lieu_NLP.docx # Quy chuẩn tiền xử lý
│   └── figures/                                     # Biểu đồ phân tích & so sánh
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn chạy

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Khởi chạy luồng trực quan trên Notebook (Khuyến nghị)
# Mở và chạy trực tiếp: src/vectorization.ipynb

# 3. Hoặc chạy các script phụ trợ
python src/compare_datasets.py
python src/plot_features.py
```

---

## 📊 Thống kê nhanh

| Chỉ số | Facebook | TikTok |
| :--- | :---: | :---: |
| **Tổng số mẫu** | $2.087$ | $1.307$ |
| **Train / Test (80/20)** | $1.669$ / $418$ | $1.045$ / $262$ |
| **Phân bố nhãn (Pos / Neu / Neg)** | $61.3\%$ / $21.0\%$ / $17.7\%$ | $65.3\%$ / $19.9\%$ / $14.8\%$ |
| **Độ dài câu TB** | $18.5$ từ | $14.0$ từ |
| **Số đặc trưng n-gram (Train)** | $4.745$ | $2.322$ |
| **Đặc trưng chung** | \multicolumn{2}{c|}{$1.540$ n-grams ($21.4\%$)} |
