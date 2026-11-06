# Bản đăng ký bổ sung v3: phần việc còn thiếu để viết báo cáo

Khóa ngày 2026-11-06, **trước khi** chạy bất kỳ bước nào dưới đây. Sai lệch sau thời điểm này ghi ở mục 5.

- Commit gốc: `7a78a4c0152b9581a6296e77c426676d64e4ad86`
- Môi trường: `models/Naive_Bayes/.venv`, Python 3.9.21, khóa tại `requirements_lock_v2.txt`
- Mã nguồn: `models/Naive_Bayes/report_runs/`. Kết quả mới ghi vào `experiments_report/`, riêng phần hoàn tất v2 ghi vào `experiments_v2/` như bản đăng ký v2. Không ghi đè file kết quả nào đã có.

## 1. Vì sao cần chạy thêm

1. Vòng v2 mới dừng ở CV: chưa có `final_candidates_v2.json`, chưa chấm test-v2.
2. BoW mới có điểm CV (E2a). Đề bài yêu cầu so sánh phương pháp trích chọn đặc trưng bằng Accuracy, Precision, Recall, F1, Confusion Matrix, và chưa có metric train/val/test cho BoW.
3. Kế hoạch nhóm (`N7-PTKPDLVB.docx`) giao phần này so sánh NB với SVM kèm ROC/AUC và ghi "cùng tập Train/Test". Thực tế hai bên dùng hai split khác nhau, nên các con số hiện có không so sánh được:

| | Split H (NB) | Split D (SVM, nhánh `origin/doanh/nv3`) |
|---|---|---|
| Cách chia | Train/val/test theo nhóm `Clean Text` | `train_test_split(test_size=0.2, stratify=y, random_state=18)` |
| Kích thước | 1491 / 309 / 287 | 1669 / 418 |

Không tráo chéo được: 365/418 dòng test D nằm trong dev H, 234/287 dòng test H nằm trong train D. Muốn so sánh phải huấn luyện lại trên cùng một split.

## 2. Phần A: hoàn tất v2 đúng bản đăng ký `experiment_plan_v2.md`

- **A1 (`a1_lock_v2.py`, không đọc `test_v2.csv` và không đọc dòng test phim):** tính điểm CV của P1, P2, PE; chọn `R0v2`, `NB_v2`, `Hybrid_v2`, `Bert_v2` như mục 4 và 5 của bản đăng ký v2; khóa vào `experiments_v2/final_candidates_v2.json`.
- **Chọn cấu hình cuối:** giống đợt v1, lấy cấu hình có macro-F1 gộp tốt nhất trên 25 outer fold. Trọng số soft voting chọn trên lưới $\{0, 0.25, 0.5, 0.75, 1\}$ bằng OOF.
- **A2 (`a2_score_v2.py`):** fit trên 1800 dòng dev, chấm test-v2 với họ 3 phép so sánh đã đăng ký (Holm, paired bootstrap 10000 lần theo `tv2_id`, kèm McNemar exact). Sau đó chấm test phim 287 mẫu **chỉ để mô tả**.

## 3. Phần B: NB trên train/val/test với BoW và TF-IDF

Dùng scikit-learn, trên `Clean Text`, split H. Bốn mô hình, đều là MultinomialNB:

| Tên | Đặc trưng | Cách chọn cấu hình |
|---|---|---|
| `BoW_val` | Count | Fit trên train, chọn theo macro-F1 val, lưới 72 cấu hình của `grid_search_val.csv` |
| `TFIDF_val` | TF-IDF sublinear L2 | Như trên. Phải trùng R0 |
| `BoW_cv` | Count | Macro-F1 gộp trên 25 outer fold của dev, lưới G 84 cấu hình |
| `TFIDF_cv` | TF-IDF sublinear L2 | Như trên. Phải trùng R1 đã khóa |

Mỗi mô hình báo cáo: train (resubstitution), val, test khi fit trên train, test khi fit lại trên train+val. Gồm Accuracy, Precision, Recall, F1 macro và theo lớp, Confusion Matrix.

**Công bố:** test phim đã được nhìn ở hai đợt trước, nên mọi điểm test ở phần này là **mô tả**, không dùng để chọn mô hình. CI95 bootstrap theo nhóm 10000 lần, tính cho từng mô hình. So sánh BoW với TF-IDF về mặt suy luận vẫn dựa trên CV (E2a).

## 4. Phần C: so sánh NB với SVM trên cùng split

Đầu vào giống nhau: `Clean Text`, token mặc định của scikit-learn. Không tái sử dụng cấu hình NB cũ, vì chúng được chọn trên dev chứa 365/418 dòng test D.

**Lưới NB (như nhau ở cả hai split):** $n$-gram tối đa $\in\{1,2,3\}$, `min_df` $\in\{1,2,3\}$, $\alpha\in\{0.001,0.003,0.01,0.03,0.1,0.3,1\}$, `fit_prior` $\in\{$True, False$\}$, cân bằng lớp $\in\{$có, không$\}$; riêng TF-IDF thêm `sublinear_tf` $\in\{$True, False$\}$. Tính ngang với các tham số SVM (`C`, `class_weight`, `ngram_range`, `min_df`, `sublinear_tf`).

- **C1, split D (phép so sánh chính):** NB chọn bằng đúng 5 fold của Doanh (`StratifiedKFold(5, shuffle=True, random_state=18)`, trung bình macro-F1 theo fold) chỉ trên 1669 dòng train, rồi fit lại trên 1669 dòng. SVM dùng nguyên `best_bow_svm.joblib`, `best_tfidf_svm.joblib`, phải tái lập đúng macro-F1 0.6148 và 0.6469.
- **Họ so sánh trên test D (Holm, $m=4$):** NB-BoW với SVM-BoW; NB-TF-IDF với SVM-TF-IDF; NB-TF-IDF với NB-BoW; SVM-TF-IDF với SVM-BoW. Chỉ số chính: paired bootstrap macro-F1 theo nhóm `Clean Text`, 10000 lần. Chỉ số phụ: McNemar exact cho accuracy.
- **Độ bền C1:** chấm lại sau khi bỏ các dòng test có text trùng y hệt trong train (19 dòng), vì split D không chia theo nhóm.
- **C2, split H (đối chiếu, mô tả):** NB chọn trên lưới trên, SVM chọn lại bằng đúng quy trình tìm kiếm tuần tự của Doanh (OvO `LinearSVC`: bước 1 `C`, `class_weight`; bước 2 `ngram_range`, `min_df`, `sublinear_tf`). Cả hai chấm macro-F1 gộp trên 25 outer fold của dev, fit trên 1800 dòng, chấm trên test 287 mẫu.
- **ROC/AUC:** One-vs-Rest cho từng lớp và macro-AUC. Điểm số của NB là `predict_proba`, của SVM là `decision_function`.

## 5. Nhật ký sai lệch

Không có thay đổi nào về thiết kế, tiêu chí hoặc họ so sánh đã khai báo. Các mục dưới đây ghi lại những gì xảy ra ngoài kế hoạch.

1. **Lỗi khi chạy `a1_lock_v2.py` (không ảnh hưởng kết quả):** lần chạy đầu dừng ở bước dựng OOF do `load_folds` trả sai chiều ma trận. Chưa có file nào được ghi và chưa đọc tập test. Đã sửa, thêm assert về kích thước rồi chạy lại.
2. **Phân tích thêm, chỉ để mô tả:** so sánh V2 với R0 trên CV (`experiments_report/cv_v2_vs_r0.csv`); baseline luôn đoán một lớp trên từng tập test; đặc trưng tiêu biểu theo lớp, chỉ dùng dev. Không dùng để chọn mô hình.
3. **Chi tiết chưa nêu trong kế hoạch:** CI95 của macro-AUC và $\Delta$ macro-AUC dùng bootstrap 2000 lần theo nhóm, chưa hiệu chỉnh đa so sánh. `Bert_v2` theo quy tắc đã đăng ký là PE, tức trung bình xác suất của P1 và HV2, nên có chứa NB và LogReg; mô hình cuối chạy một seed.
