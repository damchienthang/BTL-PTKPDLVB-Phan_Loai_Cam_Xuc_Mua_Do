# So sánh kết quả: Naive Bayes vs BoW + SVM

Dữ liệu chung: 3314 bình luận (cột `Clean Text`), 3 nhãn — Positive 62.2%, Neutral 20.9%, Negative 16.9%.

## 1. Thiết lập

| | `naive_bayes.ipynb` | `BoW_SVM_V2.ipynb` |
|---|---|---|
| Mô hình | Multinomial NB (tự cài bằng PyTorch) | Linear SVM (One-vs-Rest) |
| Đặc trưng | BoW, n-gram (1,2), `min_df=1` | BoW, n-gram (1,2), `min_df=1`, `binary=False` |
| Chia dữ liệu | Train/Val/Test = 70/15/15 (seed 42) | Train/Test = 80/20 (seed 18) + 5-fold CV |
| Số mẫu test | 498 | 663 |
| Chọn tham số | α trên tập validation | GridSearchCV (C, loss, class_weight, tham số BoW) |
| Xử lý mất cân bằng | Prior đều (`fit_prior=False`) | `class_weight='balanced'` |
| Cấu hình tốt nhất | α = 0.1, prior đều | C = 0.1, `loss='hinge'`, balanced |

## 2. Kết quả trên tập test (macro)

| Metric | Naive Bayes | BoW + SVM | Chênh lệch |
|---|---|---|---|
| Accuracy | 0.7249 | **0.7421** | +1.7 điểm |
| Precision | 0.6585 | **0.6683** | +1.0 điểm |
| Recall | 0.6077 | **0.6532** | +4.6 điểm |
| **Macro-F1** | 0.6258 | **0.6600** | **+3.4 điểm** |

## 3. F1 từng lớp (test)

| Lớp | NB (P / R / F1) | SVM (P / R / F1) |
|---|---|---|
| Negative | 0.568 / 0.500 / 0.532 | 0.580 / 0.518 / **0.547** |
| Neutral | 0.630 / 0.442 / 0.520 | 0.600 / 0.587 / **0.593** |
| Positive | 0.778 / 0.881 / 0.826 | 0.825 / 0.855 / **0.840** |

## 4. Nhận xét

- **SVM tốt hơn ở mọi metric**, rõ nhất ở **Neutral**: recall tăng từ 0.44 lên 0.59 (F1 +7.4 điểm). NB vẫn hay đoán nhầm Neutral sang Positive.
- NB thiên về Positive hơn: recall Positive cao (0.88) nhưng precision thấp (0.78). SVM cân bằng hơn giữa các lớp.
- NB **overfit mạnh**: Train F1 = 0.98 nhưng Test F1 = 0.63. Kết quả của NB trên val (0.629) và test (0.626) khá ổn định.
- Kết quả CV của SVM (Macro-F1 = 0.612) thấp hơn trên test (0.660), nên con số test của SVM có thể hơi lạc quan.
- Điểm mạnh của NB: huấn luyện chỉ là đếm tần suất (rất nhanh), dễ diễn giải. Điểm mạnh của SVM: học được trọng số phân biệt, nắm được các bigram như `không hay` (top đặc trưng của Negative).

## 5. Lưu ý khi so sánh

Hai notebook **dùng cách chia dữ liệu khác nhau** (tỉ lệ, seed, kích thước tập test), nên kết quả không hoàn toàn đối chứng 1–1. Để so sánh công bằng, nên chạy cả hai mô hình trên **cùng một tập train/test**, hoặc so sánh bằng cùng một quy trình 5-fold CV.

**Kết luận:** BoW + Linear SVM (OvR, `class_weight='balanced'`) cho kết quả tốt hơn Naive Bayes (Macro-F1 0.660 so với 0.626), nên chọn SVM làm mô hình chính và giữ NB làm baseline.
