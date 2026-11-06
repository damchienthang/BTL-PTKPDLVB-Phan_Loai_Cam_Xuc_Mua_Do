# Kế hoạch thu thập và gán nhãn dữ liệu

Tài liệu đi kèm [báo cáo cải thiện Naive Bayes](../reports/Bao_cao_Cai_thien_Naive_Bayes.md). Đây là **kế hoạch và hướng dẫn**. Không có dữ liệu nào được thu thập và không có scraper hay API nào được chạy khi viết tài liệu này.

Nội dung pháp lý chỉ là tổng hợp để nhóm tham khảo, **không phải tư vấn pháp lý**. Điều khoản nền tảng và pháp luật có thể thay đổi; hãy đọc lại bản gốc và hỏi giảng viên hoặc bộ phận pháp chế của trường trước khi làm. Kiểm tra ngày 2026-11-05.

## Mục lục

- [1. Việc cần làm trước khi thu thêm](#1-việc-cần-làm-trước-khi-thu-thêm)
- [2. Cần thêm bao nhiêu mẫu, ở lớp nào](#2-cần-thêm-bao-nhiêu-mẫu-ở-lớp-nào)
- [3. Nguồn dữ liệu khả dĩ](#3-nguồn-dữ-liệu-khả-dĩ)
- [4. Ẩn danh hóa](#4-ẩn-danh-hóa)
- [5. Hướng dẫn gán nhãn](#5-hướng-dẫn-gán-nhãn)
- [6. Làm sạch và khử trùng lặp](#6-làm-sạch-và-khử-trùng-lặp)
- [7. Quy tắc chống contamination](#7-quy-tắc-chống-contamination)
- [8. Augmentation](#8-augmentation)
- [9. Các bước đề xuất](#9-các-bước-đề-xuất)

## 1. Việc cần làm trước khi thu thêm

1. **Xác nhận nguồn dữ liệu hiện có.** Trong {{dev_rows}} bình luận dev, chỉ {{m_md}} nhắc "mưa đỏ", còn {{m_tb}} nhắc "tbn"/"trại buôn" (INFERRED: có thể là phim khác). Tuấn và Thắng cần cho biết các bài đăng nào đã được cào. Kết quả quyết định tên đề tài và nguồn thu thêm.
2. **Xin mã tiền xử lý.** `src/preprocessing.py` nằm trong `.gitignore` và không có trong repo. **UNKNOWN** cho đến khi có file này.
3. **Rà soát việc cào dữ liệu đã làm.** Kế hoạch nhóm ghi dùng công cụ cào Facebook, trong khi điều khoản Meta hạn chế thu thập tự động (mục 3). Nên trao đổi với giảng viên.
4. **Bảo vệ dữ liệu đang có.** `RAW Text` chứa tên thật và `Data/final_labeled_clean_v3.xlsx` đang được git theo dõi. Nếu repo công khai, nên cân nhắc bỏ cột này khỏi bản đẩy lên và làm sạch lịch sử git.
5. **Duyệt `models/Naive_Bayes/datasets/label_audit.csv`** và chốt hướng dẫn gán nhãn ở mục 5.

## 2. Cần thêm bao nhiêu mẫu, ở lớp nào

Số liệu dưới đây là **ước lượng thô (SPECULATIVE)** từ phép ngoại suy $F(n)=a+b\ln n$ trên 4 điểm learning curve. Learning curve thường bão hòa nên phép ngoại suy logarit có thể lạc quan. $n$ là cỡ outer-train, nhỏ hơn dev khoảng 20%.

{{T_EXT}}

*Nguồn: `models/Naive_Bayes/experiments/learning_curve_extrapolation.csv`.*

**Diễn giải:**

- Thêm 200 mẫu **Negative** dự đoán tăng {{g_neg_200}}, Neutral {{g_neu_200}}, Positive {{g_pos_200}}.
- Giảm mẫu Negative làm điểm tụt nhanh nhất, đúng với việc Negative là lớp nhỏ nhất.
- Hiệu ứng của các lớp có thể không cộng dồn.

**Đề xuất (PROPOSED):**

| Đợt | Train | Mục đích |
|---|---|---|
| 1 | +300 Negative, +200 Neutral | Tăng F1 lớp thiểu số |
| 2 | Chỉ làm nếu đợt 1 có lợi: +300 Negative, +300 Neutral | Kiểm tra độ bão hòa |
| Không ưu tiên | Positive | Lợi ích dự đoán nhỏ nhất, hiện đã có nhiều |

**`test-v2`:** bán kính CI95 hiện tại là {{t_ci_hw}} với {{n_test}} mẫu. Giả định bán kính tỉ lệ với $1/\sqrt n$ (INFERRED, chưa tính ảnh hưởng của trùng lặp và phân bố lớp):

| Bán kính CI95 mong muốn | Cỡ `test-v2` xấp xỉ |
|---|---|
| 0.05 | {{n_ci_05}} |
| 0.04 | {{n_ci_04}} |
| 0.03 | {{n_ci_03}} |

Vì macro-F1 phụ thuộc lớp nhỏ nhất, `test-v2` nên có đủ mẫu Neutral và Negative, không chỉ theo tỉ lệ tự nhiên.

## 3. Nguồn dữ liệu khả dĩ

| Nguồn | Điều khoản liên quan | Rủi ro | Khả năng dùng |
|---|---|---|---|
| **Khảo sát có đồng ý** (Google Form cho sinh viên, người xem) | Do nhóm tự đặt, có đồng ý rõ ràng | Thấp về pháp lý; văn phong khác bình luận mạng | **Ưu tiên**, nhất là cho Negative và Neutral |
| **YouTube Data API**, `commentThreads.list` | ToS cấm truy cập bằng phương tiện tự động trừ khi có phép, và cấm thu thập thông tin nhận dạng như username nếu không được phép. API có quota 1 đơn vị mỗi lần gọi. Developer Policies giới hạn lưu trữ dữ liệu (cần đọc lại bản gốc) | Trung bình: bình luận có username, phải bỏ khi lưu | Khả dĩ qua API chính thức, **cần hỏi người phụ trách trước khi gọi** |
| **Facebook** | ToS cấm "access or collect data from our Products using automated means (without our prior permission)". Có Automated Data Collection Terms riêng | Cao nếu cào tự động | **Không đề xuất cào.** Có thể nhờ giảng viên tìm hiểu kênh chính thức |
| **Meta Content Library** | Cần liên kết với tổ chức học thuật hoặc phi lợi nhuận có nghiên cứu là mục đích chính; xét duyệt độc lập; có dữ liệu bình luận; dữ liệu nằm trong môi trường kiểm soát | Thấp về điều khoản; khó tiếp cận với BTL | Chỉ khả thi nếu có giảng viên hoặc trường đứng tên |
| **TikTok** | ToS (khu vực khác, cập nhật 1/12/2025) cấm "use automated scripts to collect information". Research API giới hạn khu vực; nguồn tra cứu không liệt kê Việt Nam | Cao nếu cào | **Không đề xuất** |
| **Trang review phim** (Moveek, IMDb, Letterboxd…) | **UNKNOWN**: chưa kiểm tra điều khoản từng trang | Chưa rõ | Đọc điều khoản trước; nếu không rõ thì không dùng |

**Nguyên tắc:** không vượt đăng nhập, CAPTCHA, giới hạn tốc độ. Không dùng công cụ né điều khoản. Hỏi trước khi gọi bất kỳ API hoặc scraper nào.

**Pháp luật Việt Nam (đọc qua bản đăng lại, chưa đối chiếu Công báo):** Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15, hiệu lực từ 01/01/2026.

- Điều 2 khoản 1: dữ liệu cá nhân là thông tin "xác định hoặc giúp xác định một con người cụ thể"; dữ liệu sau khi khử nhận dạng không còn là dữ liệu cá nhân. Khoản 11 định nghĩa khử nhận dạng.
- Điều 11 khoản 1: dữ liệu thu thập phải có sự đồng ý của chủ thể trước khi thu thập, trừ trường hợp pháp luật có quy định khác.
- Điều 19 khoản 1 liệt kê các trường hợp không cần đồng ý; **nghiên cứu khoa học không có trong danh sách** ở phần đã đọc.
- Điều 14 khoản 6 điểm b: không được tái nhận dạng dữ liệu đã khử nhận dạng, trừ trường hợp pháp luật có quy định khác.

**Chưa đọc nguyên văn Điều 21–39 và chưa đọc nghị định hướng dẫn.** Vì vậy không nên suy ra "bình luận công khai thì được dùng tự do". Đây là lý do khảo sát có đồng ý được ưu tiên.

Bản YouTube ToS đọc được ghi ngày 15/12/2023 và kết quả tìm kiếm cho thấy có thể đã có bản mới hơn với thêm ngoại lệ, nên cần đọc lại bản hiện hành. Phần giới hạn lưu trữ của Developer Policies được tóm tắt tự động, chưa đọc nguyên văn.

**Nguồn đã tra:** [YouTube Terms of Service](https://www.youtube.com/static?template=terms), [commentThreads.list](https://developers.google.com/youtube/v3/docs/commentThreads/list), [YouTube Developer Policies](https://developers.google.com/youtube/terms/developer-policies), [Meta Terms of Service](https://www.facebook.com/terms), [Automated Data Collection Terms](https://www.facebook.com/legal/automated_data_collection_terms), [Meta Content Library](https://transparency.meta.com/researchtools/meta-content-library/), [TikTok Terms of Service](https://www.tiktok.com/legal/page/row/terms-of-service/en), [TikTok Research Tools](https://developers.tiktok.com/products/research-api/). Văn bản luật được đọc qua trang đăng lại [hethongphapluat.com](https://hethongphapluat.com/luat-bao-ve-du-lieu-ca-nhan-2025.html) vì trang thuvienphapluat.vn và hoatieu.vn từ chối truy cập tự động.

## 4. Ẩn danh hóa

Áp dụng **trước khi** lưu vào repo hoặc đưa cho người gán nhãn:

1. Không lưu tên hiển thị, username, ảnh đại diện, link hồ sơ, ID người dùng.
2. Xóa `@mention` và tên được tag đầu câu. Dữ liệu hiện có cho thấy đây là dạng phổ biến nhất, và tên có thể dính chữ ngay sau đó.
3. Thay tên riêng trong nội dung bằng `<TÊN>`, trừ tên phim, tên diễn viên công chúng.
4. Thay ID bài đăng và bình luận bằng mã băm có salt, salt không đưa vào repo.
5. Xóa URL, số điện thoại, email.
6. Giữ `RAW Text` đã ẩn danh ở nơi không công khai.
7. Rà ngẫu nhiên 50 dòng sau khi xử lý.

## 5. Hướng dẫn gán nhãn

### 5.1 Nguyên tắc

Gán theo **thái độ của người viết đối với bộ phim đang xét**. Đọc cả `RAW Text`, không chỉ `Clean Text`, vì cleaning có thể đổi nghĩa ("Hay lắm hả chị" thành `hay lắm chị`).

| Nhãn | Định nghĩa |
|---|---|
| **Positive** | Khen, hài lòng, khuyên xem, bênh vực phim |
| **Negative** | Chê, thất vọng, khuyên không xem |
| **Neutral** | Câu hỏi, thông tin, ý định đi xem, ý kiến cân bằng, không có đối tượng là phim |

### 5.2 Trường hợp biên

Các quy tắc sau là **PROPOSED**, rút ra từ `models/Naive_Bayes/datasets/label_audit.csv`, cần nhóm chốt trước khi gán.

| Trường hợp | Quy tắc | Ví dụ (đã che tên) |
|---|---|---|
| Câu hỏi | Neutral, trừ khi có thái độ rõ | "Hay lắm hả mọi người" |
| Ý định đi xem | Neutral | "Chắc đi coi quá" |
| Mỉa mai | Gán theo ý thật, không theo mặt chữ | "giỏi thiệt chứ, nói phát là răm rắp khỏi cần phiên dịch" |
| Mắng người chê phim | Positive vì bênh phim | "Chê thì lựa phim dở mà coi" |
| Đồng tình, không rõ đối tượng | Neutral | "Bạn nhận xét chính xác" |
| So sánh với phim khác | Xét thái độ với phim đang xét | "Phim kia hay nhưng không bằng phim này" là Positive |
| Khen và chê lẫn lộn | Theo kết luận chung; nếu cân bằng thì Neutral | "Giải trí thì ổn, còn ấn tượng thì không" |
| Điểm số | ≥7/10 Positive, 5–6 Neutral, ≤4 Negative; từ hạ thấp như "chỉ" có thể đổi nhãn | "8 điểm" |
| Tin tức, thông tin | Neutral | "CGV mình mua toàn 115k/vé" |
| Spam, quảng cáo | Đánh dấu loại bỏ, không gán nhãn | |
| Không hiểu nội dung | Đánh dấu "không rõ", đưa vào vòng phân xử | |

### 5.3 Quy trình

1. **Thử nghiệm:** gán 50 mẫu chung, thảo luận, sửa hướng dẫn.
2. **Đo đồng thuận:** ít nhất 2 người gán độc lập trên khoảng 200 mẫu, không xem nhãn của nhau. Mẫu này cũng ước lượng trần điểm do nhiễu nhãn, hiện là UNKNOWN.
3. **Tính Cohen's kappa** (Cohen, 1960):

$$\kappa=\frac{p_o-p_e}{1-p_e}$$

với $p_o$ là tỉ lệ đồng ý quan sát được, $p_e$ là tỉ lệ đồng ý ngẫu nhiên. **Ngưỡng làm việc (PROPOSED):** nếu $\kappa<0.6$, dừng và sửa hướng dẫn trước khi gán tiếp.

4. **Báo cáo** kappa tổng và ma trận nhầm giữa người gán.
5. **Phân xử:** bất đồng do người thứ ba quyết định, ghi lại lý do.
6. **Gán phần còn lại**, trộn 5–10% mẫu đã gán đôi để theo dõi độ lệch.

### 5.4 Dùng LLM để gán nhãn sơ bộ

- Ghi rõ cột `nguon_nhan = "gán nhãn sơ bộ bằng LLM"`, kèm tên model và ngày.
- Không coi là ground truth. Kiểm tra tay trên mẫu con ít nhất 200 dòng và báo cáo tỉ lệ khớp.
- **Không gửi dữ liệu ra dịch vụ ngoài khi chưa được phép** và chưa ẩn danh.
- Không dùng nhãn LLM cho `test-v2`.

## 6. Làm sạch và khử trùng lặp

- **Nhất quán:** dữ liệu mới phải dùng đúng pipeline đã tạo `Clean Text`. Pipeline này là **UNKNOWN** cho đến khi có `preprocessing.py`.
- **Lỗi đã phát hiện:** {{m_ph}} dòng dev có chuỗi giữ chỗ do tên dính chữ; xóa `hả`, `nha` làm câu hỏi thành câu khẳng định; xóa chữ số làm mất điểm số; tách từ dính qua dấu câu; tên riêng còn sót.
- **Sửa pipeline** cần áp dụng cho toàn bộ dữ liệu. Nếu làm vậy thì phải đánh giá trên `test-v2` và nêu rõ, không so trực tiếp với điểm test cũ.
- **Khử trùng lặp:** so với mọi dữ liệu hiện có, **gồm cả test**, bằng text chuẩn hóa (bỏ dấu, bỏ ký tự đặc biệt) và Jaccard trên 5-gram ký tự, ngưỡng 0.8. Mẫu mới trùng với test phải bị loại.

## 7. Quy tắc chống contamination

1. Test hiện tại **giữ nguyên, không thêm, không sửa, không dùng để chọn mẫu**.
2. Dữ liệu mới chỉ vào **train**, hoặc vào tập **`test-v2`** riêng được chọn và khóa trước khi huấn luyện.
3. Chọn `test-v2` trước, loại khỏi nguồn train, rồi mới dùng phần còn lại.
4. Không dùng mô hình đã xem test để chọn mẫu gán nhãn. Nếu dùng active learning thì chỉ dựa trên train và CV.
5. Không sửa nhãn test dựa trên dự đoán mô hình.
6. Lưu dữ liệu mới ở file riêng, không sửa `Data/final_labeled_clean_v3.xlsx` hay `models/Naive_Bayes/datasets/splits/`.
7. Mỗi dòng có: nguồn, mã bài đăng đã băm, ngày thu thập, ngày gán nhãn, người gán, đợt, cột `split_v2`.
8. Sau khi thêm dữ liệu, so sánh với đối chứng trên cùng `test-v2`.

## 8. Augmentation

Đã thử random deletion và random swap cho lớp thiểu số trong train fold (E10, báo cáo mục 5.10): {{e10_n}} bản sao, không tăng có ý nghĩa. Augmentation không thay thế dữ liệu thật và không được áp dụng cho val hay test.

## 9. Các bước đề xuất

1. Xác nhận nguồn dữ liệu và xin `preprocessing.py`.
2. Nhóm duyệt `models/Naive_Bayes/datasets/label_audit.csv`, chốt mục 5.
3. Gán đôi khoảng 200 mẫu hiện có để có kappa. **Đây là thí nghiệm nhỏ có giá trị nhất**, vì nó cho biết trần điểm thực tế.
4. Lập khảo sát có đồng ý và xin ý kiến giảng viên về API.
5. Chốt `test-v2` trước khi thu dữ liệu train mới.
6. Thu đợt 1, chạy lại `naive_bayes_improvement.ipynb`, đánh giá trên `test-v2`.
