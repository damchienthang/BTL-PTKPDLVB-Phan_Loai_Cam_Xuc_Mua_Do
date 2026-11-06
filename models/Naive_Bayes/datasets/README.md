# Dữ liệu dẫn xuất của phần Naive Bayes

Mọi file ở đây do phần Naive Bayes tạo ra. Dữ liệu gốc trong `Data/` không bị sửa.

| Đường dẫn | Nội dung |
|---|---|
| `splits/` | Split train/val/test bị khóa, chuyển từ `Data/splits/`; không được sửa |
| `label_audit.csv` | Đề xuất kiểm toán nhãn, gán nhãn sơ bộ bằng LLM, chưa được người kiểm tra |
| `external/` | Bản sao dữ liệu công khai (xem bên dưới) |
| `v2/test_v2.csv` | test-v2, bị khóa bằng hash trong `v2/MANIFEST.json` |
| `v2/pool_*.csv` | Dữ liệu ngoài chỉ dùng cho train |
| `v2/film_clean_v2.csv` | Dữ liệu phim làm sạch lại từ `RAW Text` bằng `build_data_v2.py` |
| `v2/label_fixes_train.csv` | 8 đề xuất sửa nhãn, chỉ áp dụng ở phần train của fold |

## Nguồn và giấy phép dữ liệu ngoài

Cả hai bộ được tải từ bản sao của `ura-hcmut` trên Hugging Face, gắn nhãn giấy phép **CC BY-NC-SA 4.0**: chỉ dùng phi thương mại, ghi nguồn, chia sẻ tương tự. Bản gốc do nhóm UIT NLP công bố cho mục đích nghiên cứu. Nếu dùng, phải trích dẫn:

- **UIT-VSMEC:** Ho, V., Nguyen, D., Nguyen, D., Pham, L., Nguyen, K., Nguyen, N. Emotion Recognition for Vietnamese Social Media Text. PACLING 2019. Nguồn: <https://huggingface.co/datasets/ura-hcmut/UIT-VSMEC>
- **UIT-VSFC:** Nguyen, K. V., Nguyen, V. D., Nguyen, P. X.-V., Truong, T. T.-H., Nguyen, N. L.-T. UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis. KSE 2018. Nguồn: <https://huggingface.co/datasets/ura-hcmut/UIT-VSFC>

Thông tin trích dẫn lấy từ dataset card. Các bộ này chứa bình luận thật của người dùng; không nên đưa lên repo công khai nếu chưa xem lại điều kiện giấy phép.
