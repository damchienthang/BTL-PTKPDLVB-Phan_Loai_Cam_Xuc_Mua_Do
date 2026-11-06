# Bản đăng ký thí nghiệm (pre-registration): cải thiện Naive Bayes

Khóa ngày 2026-11-05, **trước khi** chạy bất kỳ thí nghiệm nào. Mọi thay đổi sau thời điểm này phải ghi vào mục "Nhật ký sai lệch" ở cuối file, kèm lý do.

- Commit gốc: `7a78a4c0152b9581a6296e77c426676d64e4ad86`
- Môi trường: `models/Naive_Bayes/.venv` (Python 3.9.21), khóa phiên bản tại `models/Naive_Bayes/requirements_lock.txt`
- Phần cứng: Slurm job trên OSC, NVIDIA A100-PCIE-40GB, 8 CPU (`OMP_NUM_THREADS=1`)

## 1. Dữ liệu bị khóa

| File | SHA-256 |
|---|---|
| `models/Naive_Bayes/datasets/splits/train.csv` | `c04400673765183a0c904d447b611f48b74389c1b68059d517237d4cdab38505` |
| `models/Naive_Bayes/datasets/splits/val.csv` | `155c13a615172a7b06c4f2b647e222736d0354dcb603ad90c0c027ecd67f31c7` |
| `datasets/splits/test.csv` | `6135fb08b390b6743968a1fb1dde8319977e3297bad656f00e1d669f972b9ae2` |
| `Data/final_labeled_clean_v3.xlsx` | `ea6c578763a8396f4c065696e50df0431a0a978cfff40399818e2a7b4ef7a89c` |

Quy tắc:

- Notebook phát triển `naive_bayes_improvement.ipynb` **không đọc** nội dung `test.csv` và lọc xlsx theo ID của dev ngay khi nạp.
- Notebook `naive_bayes_final_test.ipynb` là nơi duy nhất đọc nhãn test, và chỉ chạy sau khi `experiments/final_candidates.json` được khóa.
- Không chạy lại ô 3 của `naive_bayes_pytorch.ipynb` vì ô đó ghi đè split.

**Công bố:** điểm test của 4 mô hình đã có sẵn trong `comparison_val_test.csv` trước bản đăng ký này, nên lần chấm cuối là lần nhìn lại test, không phải lần đầu.

## 2. Thiết kế đánh giá

- **Tập phát triển (dev):** train + val = 1800 dòng.
- **Outer CV:** `StratifiedGroupKFold(5, shuffle=True, random_state=r)` với $r\in\{0,\dots,4\}$, group theo `Clean Text`. Tổng 25 fold.
- **Inner CV:** 3 fold trong outer-train, `random_state=100*r+k`, group theo `Clean Text`. Chọn hyperparameter bằng macro-F1 gộp trên inner OOF. Hòa thì lấy cấu hình đứng trước trong thứ tự lưới.
- **Chống leakage:** vocabulary, IDF, chọn đặc trưng, lọc nhãn, oversample, augmentation đều fit lại chỉ trên phần train của từng fold.
- **Chỉ số chính:** macro-F1. Accuracy chỉ để tham khảo.
- **Điểm một biến thể:** macro-F1 gộp OOF theo từng lần lặp, báo cáo mean ± std qua 5 lần lặp. NB tất định nên std này chỉ phản ánh việc chia fold, không phải bất định tổng quát hóa.
- **So sánh ghép cặp:** bootstrap theo group `Clean Text`, 5000 lần, cùng ma trận trọng số cho mọi cặp. $\Delta$ là trung bình qua 5 lần lặp.

**Quy tắc chấp nhận:** biến thể được chấp nhận khi **cận dưới CI95 của $\Delta$ > 0** so với đối chứng. Nếu không, giữ cấu hình đơn giản hơn. p-value hiệu chỉnh Holm được ghi kèm để tham khảo nhưng không thay đổi quy tắc trên.

## 3. Đối chứng

- **R0:** cấu hình hiện tại cố định: word (1,2), `min_df=2`, TF-IDF sublinear chuẩn L2, MNB, $\alpha=0.01$, `fit_prior=False`.
- **R1:** cùng họ với R0 nhưng tune nested trên lưới G.
- **Lưới G (84 cấu hình):** bậc n-gram tối đa $\in\{1,2\}$; `min_df` $\in\{1,2,3\}$; $\alpha\in\{0.001,0.003,0.01,0.03,0.1,0.3,1\}$; `fit_prior` $\in\{$True, False$\}$.
- E2–E7 so với **R1** để đảm bảo mọi biến thể được tune như nhau trên lưới G, cộng tham số riêng của biến thể.
- Tham chiếu sklearn (CPU): LogReg và LinearSVC nested trên cùng fold. Lưới: bậc n-gram $\in\{1,2\}$, `min_df` $\in\{1,2,3\}$, `class_weight` $\in\{$None, balanced$\}$, `C` $\in\{0.1,0.3,1,3,10,30\}$ cho LogReg và $\{0.03,0.1,0.3,1,3,10\}$ cho LinearSVC. Thêm LogReg C=3 và LinearSVC C=1 cố định để nối với `comparison_val_test.csv`.

## 4. Thí nghiệm

Mỗi dòng đổi một biến so với đối chứng. Giả thuyết bị bác bỏ khi CI95 của $\Delta$ chứa 0 hoặc nằm dưới 0.

| ID | Biến thể | Đối chứng | Giả thuyết |
|---|---|---|---|
| E1 | R1 | R0 | Tune bằng nested CV tốt hơn cấu hình cố định |
| E2a | Trọng số count | R1 | Count tốt hơn TF-IDF L2 cho MNB |
| E2b | Trọng số binary | R1 | Binary tốt hơn TF-IDF L2 cho MNB |
| E3a | ComplementNB | R1 | CNB tốt hơn MNB trên dữ liệu lệch lớp |
| E3b | NB-LR (đặc trưng log-count-ratio binary, LogReg OvR) | R1 | NB-LR tốt hơn MNB |
| E4a | Đầu vào `Clean Text Old` | R1 | Giữ lại `nhưng`, chữ số, token emoji giúp tăng macro-F1 |
| E4b | `Clean Text` + token meta từ RAW (`?`, `!`, điểm cao, điểm thấp) | R1 | Khôi phục tín hiệu bị cleaning xóa giúp tăng macro-F1 |
| E5 | Đánh dấu phạm vi phủ định, cửa sổ $k\in\{1,2,3\}$ tune nested | R1 | Đánh dấu phủ định tốt hơn bigram thuần |
| E6a | Char n-gram `char_wb` (2 đến 4 hoặc 5) | R1 | Char n-gram tốt hơn word n-gram |
| E6b | Hợp word + char n-gram | R1 | Hợp đặc trưng tốt hơn word n-gram |
| E6c | Chọn đặc trưng chi² hoặc MI, $K\in\{500,1000,2000,4000\}$ | R1 | Chọn đặc trưng giảm overfit n-gram hiếm |
| E7a | Class weight balanced | R1 | Tăng macro-F1 |
| E7b | Oversample lớp thiểu số trong train fold | R1 | Tăng macro-F1 |
| E7c | Offset log-prior, $\delta\in\{-1,-0.5,0,0.5,1\}^2$ | R1 | Tăng macro-F1 trên fold ngoài |
| E8a | Soft voting NB + LogReg, trọng số tune nested | NB tốt nhất | Ensemble tốt hơn NB thuần |
| E8b | Stacking NB + LogReg | NB tốt nhất | Ensemble tốt hơn NB thuần |
| E9 | Lọc nhãn nghi nhiễu trong outer-train ($\tau=0.7$), chấm trên nhãn gốc | R1 | Bỏ nhãn nghi nhiễu giúp tăng macro-F1 |
| E10 | Augmentation (random deletion + swap) cho lớp thiểu số trong train fold | R1 | Augmentation giúp tăng macro-F1 |

**Ghép cấu hình:** các thay đổi được chấp nhận ở E2–E7 được thêm lần lượt vào R1 theo thứ tự $\Delta$ giảm dần, giữ lại nếu cận dưới CI95 so với cấu hình hiện tại > 0. Nếu không thay đổi nào được chấp nhận, NB tốt nhất là R1 khi E1 được chấp nhận, ngược lại là R0.

**E9:** chỉ chạy phiên bản "bỏ nhãn". Phiên bản "sửa nhãn" chờ người dùng duyệt `models/Naive_Bayes/datasets/label_audit.csv`. Nhóm mâu thuẫn trong test chỉ được liệt kê, không sửa.

## 5. Chấm test

Tối đa 3 ứng viên được khóa trong `experiments/final_candidates.json` trước khi mở test. Họ so sánh gồm tối đa 3 phép, hiệu chỉnh Holm:

1. NB tốt nhất (fit trên train+val) so với R0 fit lại trên train+val.
2. Mô hình lai tốt nhất (NB-LR hoặc ensemble) so với NB tốt nhất.
3. NB tốt nhất so với LogReg nested fit lại trên train+val.

Nếu NB tốt nhất trùng R0 thì bỏ phép 1 và ghi rõ. Chỉ số chính là paired bootstrap theo group trên macro-F1 (10000 lần). Chỉ số phụ là McNemar exact cho accuracy. R0 fit trên train (0.544) được tái lập để đối chiếu lịch sử, không nằm trong họ so sánh.

## 6. Phân tích khác (chỉ trên dev)

- Phân tích lỗi: đọc ~30 mẫu sai từ OOF, không dùng test.
- Learning curve theo tổng số mẫu và theo từng lớp, trên outer fold.
- Tái tính con số 37/1491 mẫu nghi nhiễu trên train.

## 7. Nhật ký sai lệch

Mọi mục dưới đây được quyết định sau khi đã thấy kết quả trên dev và **trước khi** mở test.

1. **Hạ tầng (không ảnh hưởng kết quả):** venv đặt trong `models/Naive_Bayes/.venv` theo yêu cầu người dùng. NB chạy trên GPU; sklearn chạy trên CPU với 1 luồng BLAS mỗi worker.
2. **E3b* (`E3b_cw`, thêm):** lưới của E3b đã đăng ký không có `class_weight`, trong khi LogReg tham chiếu có, nên so sánh thiếu công bằng. E3b vẫn được báo cáo nguyên bản.
3. **E9b* (`E9b_strict`, thêm):** E9 gắn cờ khoảng 15% outer-train, không phản ánh câu hỏi về ~2.5% nhãn nghi nhiễu. E9 vẫn được báo cáo nguyên bản.
4. **`X_old_bal` (thêm, thăm dò):** ghép E4a + E7a, hai biến thể có $\Delta$ điểm lớn nhất nhưng đều không đạt quy tắc. Đây là bước ngoài quy tắc ghép ở mục 4 và ban đầu được ghi là không đủ điều kiện vào test.
5. **Bảng độ nhạy $\alpha$ (thêm):** chỉ mang tính mô tả, không dùng để chọn tham số.
6. **E8b:** bản đăng ký không nêu cấu hình meta-learner; đã dùng LogReg $C=1$, `class_weight="balanced"`.
7. **Mở rộng họ so sánh trên test (quyết định của người dùng, 2026-11-05, trước khi mở test):** thêm phép thứ 4 `X_old_bal` so với NB tốt nhất, Holm với $m=4$. Lý do: cấu hình này có điểm CV cao nhất trong các biến thể NB; test độc lập với bước chọn trên dev nên vẫn là kiểm chứng hợp lệ nếu khai báo trước khi mở. Điểm CV của nó có thể bị thổi phồng do được chọn sau khi xem $\Delta$ của các biến thể, và báo cáo phải nêu rõ nguồn gốc thăm dò. Việc này thay đổi một cam kết đã ghi trong mục 4.
8. **Vị trí file (không ảnh hưởng kết quả, sau khi đã chấm test):** theo yêu cầu người dùng, báo cáo, hình, kế hoạch dữ liệu và file audit chuyển vào `models/Naive_Bayes/`.
9. **Vị trí split (không ảnh hưởng kết quả):** theo yêu cầu người dùng, `Data/splits/` được chuyển sang `models/Naive_Bayes/datasets/splits/`. Hash ba file không đổi. Mọi đường dẫn trong notebook được sửa tại chỗ, output giữ nguyên.
10. **Thêm lượt nhìn test (2026-11-06, khai báo trước khi chạy):** để có đủ metric BoW và đối chiếu SVM cho báo cáo, các mô hình BoW và SVM được chấm thêm trên test 287 mẫu. Đây là lần nhìn thứ ba, chỉ để mô tả, không chọn mô hình. Chi tiết ở `experiment_plan_v3.md`.
