# DỰ ÁN PHÂN TÍCH QUAN ĐIỂM (SENTIMENT ANALYSIS) TỪ MẠNG XÃ HỘI
**Facebook & TikTok Comment Data Processing Pipeline**

---

## 1. Giới thiệu tổng quan

Dự án này xây dựng một luồng (pipeline) hoàn chỉnh phục vụ nghiên cứu và thực nghiệm xử lý ngôn ngữ tự nhiên (NLP) phân loại sắc thái bình luận tiếng Việt (Sentiment Analysis):
1. **Thu thập & Hợp nhất dữ liệu (Data Merging & Ingestion)** từ Facebook và TikTok.
2. **Gán nhãn tự động & Kiểm duyệt chất lượng (LLM Annotation & Review)** bằng mô hình ngôn ngữ lớn (Gemini) kết hợp thẩm định logic.
3. **Tiền xử lý NLP chuyên biệt (Text Preprocessing)**: Chuẩn hóa Unicode NFC, giải mã Teencode/viết tắt, bảo toàn phủ định, chuẩn hóa emoji, tách từ tiếng Việt (PyVi) tạo ra 2 định dạng riêng biệt phục vụ cả mô hình truyền thống (TF-IDF/ML) và mô hình Deep Learning (PhoBERT/Transformer).

> **Lưu ý về Git & Dữ liệu:**
> - Toàn bộ mã nguồn, cấu hình và tài liệu hướng dẫn được quản lý trên GitHub.
> - Các tệp dữ liệu thô (`.csv`, `.xlsx`, `.json`) và dữ liệu đầu ra không đưa lên Git để đảm bảo dung lượng repository nhẹ, sạch và bảo mật dữ liệu. Cấu trúc thư mục được duy trì thông qua các tệp `.gitkeep`.

---

## 2. Cấu trúc thư mục dữ liệu

Khi clone dự án về máy, cấu trúc thư mục dữ liệu đã được dựng sẵn dưới dạng thư mục trống:

```text
nlp/
├── data/
│   ├── facebook/
│   │   ├── raw/                  <-- Nơi chứa dữ liệu Facebook gốc (Excel/JSON crawl về)
│   │   ├── intermediate/         <-- Dữ liệu trung gian (checkpoint LLM, gán nhãn thô, log)
│   │   └── processed/            <-- Nơi lưu kết quả Facebook đã làm sạch & tiền xử lý chuẩn
│   └── tiktok/
│       ├── raw/                  <-- NƠI ĐẶT CÁC FILE CSV GỐC XUẤT TỪ TIKTOK
│       └──                       <-- Nơi lưu file gộp (merged) và file tiền xử lý TikTok
├── tools/                        <-- Công cụ bổ trợ / crawler
├── preprocess_comments.py        <-- Module cốt lõi tiền xử lý văn bản tiếng Việt
├── merge_tiktok_comments.py       <-- Script gộp toàn bộ file CSV TikTok thô
├── preprocess_tiktok_comments.py <-- Script chạy tiền xử lý cho dữ liệu TikTok
├── clean_merge_comments.py       <-- Script gộp dữ liệu thô Facebook
├── process_clean_v3.py           <-- Script chạy tiền xử lý cho dữ liệu Facebook
├── label_all_comments.py         <-- Script gán nhãn tự động qua Gemini API
├── build_final_labeled_dataset.py<-- Script hoàn thiện tập dữ liệu gán nhãn
└── requirements.txt              <-- Danh sách thư viện Python phụ thuộc
```

---

## 3. Hướng dẫn thiết lập môi trường

Yêu cầu: Python 3.10 trở lên trên Windows/Linux/macOS.

```bash
# 1. Tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 4. Hướng dẫn chạy chi tiết từng bước

### PHẦN A: XỬ LÝ DỮ LIỆU TIKTOK (Từ file CSV xuất ra)

#### Bước 1: Đặt dữ liệu gốc vào đúng vị trí
- Sau khi sử dụng tiện ích (extension như *TikTok Comment Exporter*) để tải bình luận từ các video TikTok, bạn sẽ có các file CSV dạng:
  - `TTCommentExporter-7540245360342469895-160-comments.csv`
  - `TTCommentExporter-7541406958004309256-81-comments.csv`
  - ...
- **Vị trí lưu**: Copy toàn bộ các file `.csv` này vào thư mục:
  ```text
  data/tiktok/raw/
  ```

#### Bước 2: Gộp tất cả các file CSV thành 1 bảng tổng hợp
Chạy lệnh sau:
```bash
python merge_tiktok_comments.py
```
- **Chức năng**:
  - Tự động duyệt qua toàn bộ file CSV trong `data/tiktok/raw/`.
  - Chuẩn hóa tên cột, chuyển đổi thời gian đăng bình luận.
  - Tự động loại bỏ các bình luận bị trùng lặp (dựa trên `comment_id` hoặc nội dung).
- **Kết quả đầu ra**:
  - `data/tiktok/tiktok_comments_merged.xlsx` (File Excel tổng hợp).
  - `data/tiktok/tiktok_comments_merged.csv` (File CSV mã hóa UTF-8-sig).

#### Bước 3: Chạy tiền xử lý NLP cho dữ liệu TikTok
Chạy lệnh:
```bash
python preprocess_tiktok_comments.py
```
- **Kết quả đầu ra**:
  - `data/tiktok/tiktok_comments_clean.xlsx`
  - Các cột bổ sung phục vụ huấn luyện mô hình:
    - `Clean Text`: Văn bản đã sửa teencode, chuẩn hóa dấu câu, đưa về chữ thường.
    - `Text_PhoBERT`: Đã tách từ bằng PyVi (`từ_ghép`), giữ nguyên dấu câu và cấu trúc ngữ pháp tự nhiên để đưa vào mô hình Transformer/PhoBERT.
    - `Text_TFIDF`: Đã tách từ (`_`), loại bỏ toàn bộ dấu câu, ký tự đặc biệt và stop words tiếng Việt để đưa vào TF-IDF / Bag of Words / Machine Learning cổ điển (SVM, Naive Bayes, Logistic Regression).

---

### PHẦN B: XỬ LÝ DỮ LIỆU FACEBOOK

#### Bước 1: Đặt dữ liệu thô
- Các bài viết và bình luận crawl từ Facebook dạng file `.xlsx` đặt tại:
  ```text
  data/facebook/raw/simple_post_excel/
  ```

#### Bước 2: Gộp và làm sạch định dạng ban đầu
Chạy lệnh:
```bash
python clean_merge_comments.py
```
- **Kết quả**: Sinh ra file `data/facebook/raw/simple_post_comments_merged.xlsx`.

#### Bước 3: Gán nhãn cảm xúc tự động (LLM) và thẩm định (Tuỳ chọn)
```bash
# Gán nhãn cảm xúc với Gemini API (yêu cầu GEMINI_API_KEY trong file .env)
python label_all_comments.py

# Rà soát, chuẩn hóa nhãn và xử lý các ca nhiễu
python review_annotations_full.py
```
- File trung gian và checkpoint lưu tại `data/facebook/intermediate/`.

#### Bước 4: Chạy tiền xử lý NLP toàn diện
Chạy lệnh:
```bash
python process_clean_v3.py
```
- **Kết quả đầu ra cuối cùng**:
  - Lưu tại: `data/facebook/processed/final_labeled_clean_v3.xlsx`
  - Bao gồm đầy đủ nhãn cảm xúc chuẩn (`Final Label`) cùng 2 định dạng văn bản phục vụ mô hình (`Text_PhoBERT` và `Text_TFIDF`).

---

## 5. Bảng đặc tả các trường dữ liệu đầu ra

| Cột dữ liệu | Mô tả | Ứng dụng phù hợp |
| :--- | :--- | :--- |
| `Comment` / `comment_text` | Văn bản bình luận gốc thu thập từ mạng xã hội | Tham chiếu đối soát |
| `Clean Text` | Đã sửa chính tả, chuẩn hóa Unicode, dịch teencode | Đọc hiểu, hiển thị báo cáo |
| `Text_PhoBERT` | Tách từ bằng dấu gạch dưới (`_`), **giữ nguyên dấu câu và ngữ cảnh** | Huấn luyện mô hình PhoBERT / RoBERTa / Transformer |
| `Text_TFIDF` | Tách từ bằng `_`, **đã lọc sạch dấu câu, ký hiệu và Stopwords** | Trích xuất đặc trưng TF-IDF, CountVectorizer, SVM, Naive Bayes |
| `Final Label` | Nhãn cảm xúc đã được thẩm định (`Positive`, `Negative`, `Neutral`) | Nhãn mục tiêu (Ground Truth) để huấn luyện & đánh giá |

---

## 6. Xử lý sự cố thường gặp (Troubleshooting)

1. **Lỗi `PermissionError: [Errno 13] Permission denied` trên Windows**:
   - Nguyên nhân: Bạn đang mở tệp Excel đầu ra bằng Microsoft Excel khiến hệ điều hành khóa tệp.
   - Khắc phục: Đóng tệp Excel trước khi chạy lại lệnh, hoặc script sẽ tự động tạo tệp dự phòng có đuôi `*_new.xlsx`.
2. **Lỗi hiển thị tiếng Việt trên Terminal Windows**:
   - Script đã được tích hợp `sys.stdout.reconfigure(encoding="utf-8")`. Nếu cần, bạn có thể gõ lệnh `chcp 65001` trên CMD/PowerShell trước khi chạy.

