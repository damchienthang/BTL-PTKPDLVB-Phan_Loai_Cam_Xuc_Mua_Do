# Bản đăng ký thí nghiệm v2: dữ liệu giàu hơn và test-v2

Khóa ngày 2026-11-05, **trước khi** chạy thí nghiệm v2 và trước khi có mô hình nào dự đoán trên test-v2. Sai lệch sau thời điểm này ghi ở mục 7.

Người dùng đã phê duyệt toàn bộ hướng đi: dữ liệu ngoài công khai, cài gói, PhoBERT, áp dụng đề xuất sửa nhãn trên train. Bản đăng ký cũ `experiment_plan.md` và kết quả của nó giữ nguyên.

## 1. Dữ liệu

| File | Số dòng | SHA-256 |
|---|---|---|
| `models/Naive_Bayes/datasets/v2/test_v2.csv` | 1311 | `fc6770dea3e7676937b4ae4f6dcb1756485eb0c536f0a1897669331294986b95` |
| `models/Naive_Bayes/datasets/v2/pool_vsmec.csv` | 5294 | `199b251d67408c386cfdc77e66ecdc599ba29098621b11490332b6c367e8d14b` |
| `models/Naive_Bayes/datasets/v2/pool_vsfc.csv` | 11355 | `ba1598cd988925cdbe5e50671c9efd4b45720dcabe80f5da6c7b488a6be30b25` |
| `models/Naive_Bayes/datasets/v2/film_clean_v2.csv` | 2087 | `1b547d75f7c5b7ec974c9e170171369b73209cba477a8a0acabc3aa7bbb643f2` |
| `models/Naive_Bayes/datasets/v2/label_fixes_train.csv` | 8 | `a893da630e968fbe88927e75b07ed4c04dc2bf86164bb73398d4436b8f0e9b55` |

- **test-v2:** UIT-VSMEC, phần `validation` và `test` chính thức. Phân bố: {'Negative': 634, 'Positive': 407, 'Neutral': 270}.
- **Quy đổi nhãn:** Enjoyment thành Positive; Sadness, Fear, Anger, Disgust thành Negative; Other thành Neutral; bỏ Surprise.
- **Pool train:** UIT-VSMEC `train` {'Negative': 2720, 'Positive': 1554, 'Neutral': 1020}; UIT-VSFC `train` {'Positive': 5580, 'Negative': 5324, 'Neutral': 451}. Pool đã loại dòng trùng và gần trùng với test-v2.
- **Văn bản:** `clean_v2` áp dụng giống nhau cho mọi nguồn (`models/Naive_Bayes/build_data_v2.py`). Giữ chữ số, `?`, `!`, emoji, tiểu từ; không xóa stopword; có tách từ `underthesea`.
- `naive_bayes_v2.ipynb` và `phobert_v2.py` **không đọc** `test_v2.csv`.

**Giới hạn đã biết của test-v2:** không phải bình luận phim; nhãn được quy đổi từ cảm xúc; **cùng kho dữ liệu với pool VSMEC**. Mô hình có pool VSMEC gần như chắc chắn được lợi trên test-v2 mà không chứng minh gì về bình luận phim. Kết luận về bình luận phim chỉ dựa vào CV (outer-test chỉ gồm dòng phim).

## 2. Đánh giá

- Giữ nguyên 25 outer fold và inner fold của bản đăng ký cũ, trên 1800 dòng dev phim.
- **Pool luôn nằm trong phần train, không bao giờ ở phần chấm.** Điểm CV chỉ tính trên dòng phim, với nhãn gốc.
- Bootstrap theo group 5000 lần trên dòng phim.
- **Quy tắc chấp nhận:** cận dưới CI95 của $\Delta$ > 0.

## 3. Chuỗi biến thể NB

Mỗi bước một biến, tune nested trên lưới G (84 cấu hình) cộng tham số riêng. Token dùng `\b\w+\b` để giữ chữ số một ký tự.

| ID | Biến thể | So với | Ghi chú |
|---|---|---|---|
| V1 | NB trên `clean_v2` | R1 | Luôn làm nền cho các bước sau vì chỉ `clean_v2` áp dụng được cho dữ liệu ngoài |
| V2 | V1 + class weight balanced | V1 | Giữ nếu được chấp nhận |
| V3a | Nền + pool VSMEC, $\lambda\in\{0.03,0.1,0.3,1\}$ | Nền | |
| V3b | Nền + pool VSFC, cùng $\lambda$ | Nền | |
| V3c | Nền + cả hai pool | Nền | Nếu nhiều biến thể V3 đạt, lấy $\Delta$ lớn nhất |
| V4 | Nền + 8 nhãn sửa | Nền | Chỉ sửa ở phần train của fold, chấm trên nhãn gốc |

$\lambda$ là trọng số mỗi dòng pool, tune nested.

## 4. Mô hình khác

- **LR1:** LogReg trên `clean_v2`. **LR2:** LR1 + pool VSMEC. Lưới: `min_df` $\in\{1,2\}$, `C` $\in\{0.3,1,3,10\}$, `class_weight` $\in\{$None, balanced$\}$, word (1,2), LR2 thêm $\lambda\in\{0.1,0.3,1\}$. Giữ LR2 nếu được chấp nhận so với LR1. sklearn, CPU.
- **HV2:** soft voting NB tốt nhất + LogReg tốt nhất, $w$ tune nested.
- **P1:** PhoBERT `vinai/phobert-base-v2` trên dòng phim. **P2:** P1 + pool VSMEC. Tham số cố định, không tune: 4 epoch, batch 32, lr 2e-5, max_len 128, weight decay 0.01, warmup 10%, loss cân bằng lớp, seed `42+100*r+k`.
- **PE:** trung bình xác suất của PhoBERT tốt hơn và HV2, trọng số cố định 0.5.

Ước lượng thời gian PhoBERT lấy từ smoke test trên một fold phim; điểm của smoke test không dùng để chọn mô hình.

## 5. Chấm

**Ứng viên** khóa trong `experiments_v2/final_candidates_v2.json` trước khi mở test-v2:

- `R0v2`: cấu hình R0 trên `clean_v2`, chỉ dữ liệu phim.
- `NB_v2`: kết quả chuỗi V1–V4.
- `Hybrid_v2`: HV2.
- `Bert_v2`: mô hình có điểm CV cao nhất trong P1, P2, PE.

**Họ so sánh trên test-v2** (Holm, $m=3$, paired bootstrap theo `tv2_id`, 10000 lần, kèm McNemar exact):

1. `NB_v2` so với `R0v2`
2. `Hybrid_v2` so với `NB_v2`
3. `Bert_v2` so với `Hybrid_v2`

**Test cũ (287 mẫu):** sau khi chấm test-v2, chấm các mô hình đã khóa trên test cũ **chỉ để mô tả**, không chọn gì. Đây là lần nhìn thứ ba và được công bố như vậy. Lý do: đây là dữ liệu phim độc lập duy nhất còn lại.

## 6. Không làm

- Không cào Facebook, YouTube, TikTok.
- Không dùng dữ liệu sinh tổng hợp hoặc nhãn LLM trong test-v2.
- Không sửa nhãn ngoài 8 đề xuất đã có; 9 dòng mơ hồ để nhóm quyết.
- Không sửa `Data/final_labeled_clean_v3.xlsx`, `models/Naive_Bayes/datasets/splits/`.

## 7. Nhật ký sai lệch

1. **Vị trí file (không ảnh hưởng kết quả):** theo yêu cầu người dùng, mọi file mới chỉ nằm trong `models/Naive_Bayes/`. `Data/v2/` và `Data/external/` chuyển sang `models/Naive_Bayes/datasets/`. Hash không đổi. Việc chuyển diễn ra khi `phobert_v2.py` đang chạy; script đã nạp dữ liệu vào bộ nhớ từ trước.
2. **Vị trí split (không ảnh hưởng kết quả):** `Data/splits/` được chuyển sang `models/Naive_Bayes/datasets/splits/`, hash không đổi.
3. **Bước chốt và chấm (2026-11-06, khai báo trước khi mở test-v2):** thực hiện bằng `report_runs/a1_lock_v2.py` và `report_runs/a2_score_v2.py` thay vì notebook. Notebook v2 không lưu xác suất OOF nên PE cần dựng lại xác suất của HV2 bằng scikit-learn từ `selected_configs.csv`, có đối chiếu với dự đoán OOF đã lưu. Phần chọn cấu hình cuối theo đúng cách của đợt v1. Chi tiết ở `experiment_plan_v3.md`.
