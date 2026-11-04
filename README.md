# NHIỆM VỤ 2: TRÍCH CHỌN ĐẶC TRƯNG & VECTOR HÓA DỮ LIỆU
## Đề tài 3A: Phân tích cảm xúc bình luận phim "Mưa đỏ" trên Facebook
**Môn học:** Phân tích và Khai phá dữ liệu văn bản  
**Người thực hiện:** Đàm Chiến Thắng (Thắng) - GitHub: [@damchienthang](https://github.com/damchienthang)

---

## 🎯 Mục tiêu nhiệm vụ
1. Tiếp nhận dữ liệu bình luận đã làm sạch và gán nhãn ($2.087$ mẫu).
2. Chuyển đổi toàn bộ văn bản thành ma trận véc-tơ số học bằng phương pháp **TF-IDF (Term Frequency - Inverse Document Frequency)** kết hợp Unigram & Bigram ($1\text{–}2\text{-gram}$).
3. Bàn giao gói ma trận đặc trưng cho **Hiếu (Naive Bayes)** và **Doanh (SVM)** tự chia tập Train/Test và huấn luyện mô hình.
4. Trực quan hóa phân bố đặc trưng và xuất bảng số liệu mẫu phục vụ thuyết trình báo cáo.

---

## 📁 Cấu trúc thư mục

```text
.
├── Data/
│   ├── final_labeled_clean_v3.xlsx                  # Dữ liệu sạch đầu vào (2.087 mẫu)
│   └── vectorized/
│       ├── vectorized_data.joblib                   # Gói bàn giao ma trận TF-IDF cho Hiếu & Doanh
│       ├── tfidf_vectorizer.joblib                  # Mô hình TF-IDF Vectorizer
│       └── sample_tfidf_matrix_for_slides.xlsx      # Bảng ma trận số mẫu để chiếu Slide
├── src/
│   ├── vectorization.py                             # Mã nguồn chính trích chọn đặc trưng & TF-IDF
│   └── demo_load_data.py                            # Demo hướng dẫn Hiếu & Doanh nạp data và huấn luyện
├── reports/
│   ├── Bao_cao_Nhiem_vu_2_Thang_Trich_chon_dac_trung.docx   # Báo cáo Word chuẩn quy cách nhóm
│   ├── generate_docx_report.py                      # Code sinh báo cáo Word tự động
│   └── figures/
│       ├── top_20_tfidf_overall.png                 # Biểu đồ Top 20 đặc trưng TF-IDF
│       └── top_features_by_sentiment.png            # Biểu đồ Top 15 đặc trưng theo 3 nhãn cảm xúc
├── N7-PTKPDLVB.docx                                 # Kế hoạch & phân công nhiệm vụ nhóm
├── .gitignore
└── README.md
```

---

## 🚀 Hướng dẫn chạy chương trình

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
# Hoặc cài đặt trực tiếp:
pip install scikit-learn pandas numpy joblib python-docx matplotlib seaborn
```

### 2. Thực hiện Vector hóa văn bản
Chạy mã nguồn chính để tạo ma trận đặc trưng TF-IDF:
```bash
python src/vectorization.py
```

### 3. Demo dành cho Hiếu (Naive Bayes) và Doanh (SVM) nạp dữ liệu
```bash
python src/demo_load_data.py
```

Code mẫu để Hiếu và Doanh nạp trong dự án của mình:
```python
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Nạp file từ Thắng
data = joblib.load("Data/vectorized/vectorized_data.joblib")
X = data['X_tfidf']                    # Ma trận đặc trưng (2087 x 5872)
y = np.array(data['labels'], dtype=str) # Nhãn (Positive, Neutral, Negative)

# Tự chia Train / Test theo nhu cầu:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 📊 Kết quả thực nghiệm

* **Tổng số bình luận:** $2.087$ mẫu (Positive: $1.280$, Neutral: $438$, Negative: $369$).
* **Cấu hình mô hình:** `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`.
* **Kích thước ma trận:** **$(2.087 \times 5.872)$** ($5.872$ đặc trưng n-gram).
* **Độ thưa ma trận:** **$99.63\%$** (lưu trữ dưới dạng CSR sparse matrix).
* **Top đặc trưng có trọng số cao nhất:** `hay` ($0.0570$), `phim` ($0.0418$), `xem` ($0.0359$), `không` ($0.0300$), `đi` ($0.0296$), `có`, `tôi`, `coi`, `thấy`, `phim hay`...
