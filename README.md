# BÀI TẬP LỚN: PHÂN TÍCH VÀ KHAI PHÁ DỮ LIỆU VĂN BẢN (NLP)
## Đề tài: Phân tích cảm xúc bình luận phim "Trại buôn người" (Facebook & TikTok)

---

## 🎯 Giới thiệu tổng quan
Dự án xây dựng một luồng (pipeline) hoàn chỉnh phục vụ nghiên cứu và thực nghiệm xử lý ngôn ngữ tự nhiên (NLP) phân loại sắc thái bình luận tiếng Việt:
1. **Nhiệm vụ 1: Thu thập & Tiền xử lý dữ liệu**
   - Thu thập bình luận từ Facebook ($2.087$ mẫu) và TikTok ($1.307$ mẫu).
   - Gán nhãn tự động bằng Gemini kết hợp thẩm định logic và QC người dùng.
   - Chuẩn hóa văn bản: Unicode NFC, giải mã Teencode, chuẩn hóa emoji, tách từ tiếng Việt (PyVi) tạo ra hai định dạng chuẩn (`Text_PhoBERT` và `Text_TFIDF`).
2. **Nhiệm vụ 2: Trích chọn đặc trưng & Vector hóa dữ liệu**
   - Phân chia Train ($80\%$) và Test ($20\%$) phân tầng nhãn chống rò rỉ dữ liệu (Anti-Data Leakage).
   - Trích xuất ma trận TF-IDF và Bag-of-Words (BoW) kết hợp n-gram ($1\text{–}2\text{-gram}$), lọc từ hiếm (`min_df=2`), `sublinear_tf=True`, chuẩn hóa $L_2$.
   - Đối sánh không gian đặc trưng và phân bố cảm xúc giữa Facebook và TikTok trong notebook [`src/vectorization.ipynb`](src/vectorization.ipynb).
3. **Nhiệm vụ 3: Huấn luyện, đánh giá & So sánh mô hình phân loại**
   - Huấn luyện và tối ưu siêu tham số trên các thuật toán học máy: Naive Bayes (MultinomialNB) và Support Vector Machine (SVM).
   - Đánh giá chéo, kiểm định độ tin cậy thống kê, xuất ma trận nhầm lẫn (Confusion Matrix) và đường cong ROC.

---

## 📁 Cấu trúc thư mục dự án

```text
.
├── Data/                                            # Dữ liệu phục vụ huấn luyện và đánh giá
│   ├── final_labeled_clean.xlsx                     # Dữ liệu sạch TikTok (1.307 mẫu)
│   ├── final_labeled_clean_v3.xlsx                  # Dữ liệu sạch Facebook (2.087 mẫu)
│   ├── merged_fb_tiktok.xlsx                        # Dữ liệu gộp Facebook & TikTok
│   └── vectorized/                                  # Ma trận vector hóa (.joblib)
│       ├── tiktok_tfidf_train_test.joblib           # TF-IDF TikTok
│       ├── tiktok_bow_train_test.joblib             # BoW TikTok
│       ├── facebook_tfidf_train_test.joblib         # TF-IDF Facebook
│       └── facebook_bow_train_test.joblib           # BoW Facebook
├── data/                                            # Cấu trúc thư mục dữ liệu thô và pipeline
│   ├── facebook/ (raw, intermediate, processed)
│   └── tiktok/ (raw)
├── models/                                          # Mô hình đã huấn luyện
│   ├── Naive_Bayes/                                 # Mô hình và thực nghiệm Naive Bayes
│   ├── bow_svm/                                     # Mô hình SVM với Bag-of-Words
│   └── tfidf_svm/                                   # Mô hình SVM với TF-IDF
├── src/                                             # Mã nguồn thực nghiệm
│   ├── vectorization.ipynb                          # Notebook xử lý và vector hóa toàn diện
│   ├── BoW_SVM_V2.ipynb                             # Notebook huấn luyện SVM BoW
│   ├── TF_IDF_SVM_V2.ipynb                          # Notebook huấn luyện SVM TF-IDF
│   └── Model_Comparison.ipynb                       # So sánh hiệu năng các mô hình
├── tools/                                           # Công cụ thu thập dữ liệu (crawlers)
│   ├── crawl_facebook_comments.py
│   └── crawl_tiktok_comments.py
├── reports/                                         # Báo cáo và biểu đồ trực quan hóa
│   ├── figures/                                     # Biểu đồ phân tích dữ liệu & mô hình
│   └── BaoCaoNLP-N07.docx                           # Báo cáo tổng hợp bài tập lớn hoàn chỉnh
├── requirements.txt                                 # Thư viện cần thiết
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & Thực thi

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chạy luồng tiền xử lý và gán nhãn
```bash
# Tiền xử lý dữ liệu Facebook
python process_clean_v3.py

# Tổng hợp và rà soát nhãn
python build_final_labeled_dataset.py
```

### 3. Thực thi Vector hóa dữ liệu
Mở và chạy trực tiếp notebook:
```bash
jupyter notebook src/vectorization.ipynb
```

### 4. Huấn luyện và đánh giá mô hình
- **SVM:** Chạy các notebook trong `src/BoW_SVM_V2.ipynb` và `src/TF_IDF_SVM_V2.ipynb`.
- **So sánh mô hình:** Xem chi tiết tại `src/Model_Comparison.ipynb`.
