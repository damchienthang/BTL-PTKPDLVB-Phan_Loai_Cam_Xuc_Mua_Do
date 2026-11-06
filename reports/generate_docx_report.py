"""
Script sinh báo cáo Word (.docx) chuẩn theo quy định của nhóm:
- Font: Times New Roman
- Cỡ chữ: 13 cho nội dung
- Line spacing: 1.5 lines
- Margins: Trái 3cm, Phải 2cm, Trên 2cm, Dưới 2cm
- Căn lề: Justify
- Bảng biểu và hình ảnh căn giữa, đánh số thứ tự và chú thích.
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_report():
    doc = Document()

    # 1. Cấu hình Margins: Trái 3cm, Phải 2cm, Trên 2cm, Dưới 2cm
    sections = doc.sections
    for section in sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)

    # Cấu hình Normal style
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Times New Roman'
    font.size = Pt(13)
    font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.5
    style_normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    style_normal.paragraph_format.space_after = Pt(6)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(18)
        run.font.color.rgb = RGBColor(16, 44, 87)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(12)
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.italic = True
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(50, 50, 50)
        p.paragraph_format.space_after = Pt(18)
        return p

    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(15)
        run.font.color.rgb = RGBColor(16, 44, 87)
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13.5)
        run.font.color.rgb = RGBColor(41, 128, 185)
        return p

    def add_heading_3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.italic = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(52, 73, 94)
        return p

    def add_caption(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.italic = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(80, 80, 80)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(12)
        return p

    # TIÊU ĐỀ BÁO CÁO
    add_title("BÁO CÁO NHIỆM VỤ THỰC HIỆN DỰ ÁN NLP (ĐỀ TÀI 3A)\nPHÂN TÍCH CẢM XÚC BÌNH LUẬN PHIM \"TRẠI BUÔN NGƯỜI\"")
    add_subtitle("PHẦN NHIỆM VỤ 2: TRÍCH CHỌN ĐẶC TRƯNG & VECTOR HÓA DỮ LIỆU\nThành viên phụ trách: Thắng")

    # THÔNG TIN CHUNG
    p = doc.add_paragraph()
    r = p.add_run("1. Người thực hiện: ")
    r.bold = True
    p.add_run("Thắng (Nhánh Dữ liệu)\n")
    r2 = p.add_run("2. Nhiệm vụ được giao: ")
    r2.bold = True
    p.add_run("Nhận dữ liệu sạch đã gán nhãn từ Tuấn; xây dựng mã nguồn chuyển đổi văn bản sang không gian vector (Vectorization); thực hiện phân tầng tập Train/Test; xuất gói dữ liệu bàn giao cho Hiếu (Naive Bayes) và Doanh (SVM); viết nội dung lý thuyết Chương 2 về Trích chọn đặc trưng và báo cáo thực nghiệm Mục 3.2.\n")
    r3 = p.add_run("3. Sản phẩm bàn giao: ")
    r3.bold = True
    p.add_run("Mã nguồn trích chọn đặc trưng (vectorization.py), gói dữ liệu vector hóa hoàn chỉnh (vectorized_dataset.joblib, các file ma trận .npz, .npy, file chia train/test .xlsx), biểu đồ phân tích trực quan, và bản báo cáo chi tiết này.")

    # ==========================================
    # PHẦN 1: CHƯƠNG 2 - CƠ SỞ LÝ THUYẾT VỀ TRÍCH CHỌN ĐẶC TRƯNG
    # ==========================================
    add_heading_1("CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VỀ TRÍCH CHỌN ĐẶC TRƯNG")

    add_heading_2("2.1. Đặt vấn đề về biểu diễn văn bản trong học máy")
    p = doc.add_paragraph(
        "Trong lĩnh vực Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP) và Phân tích khai phá dữ liệu văn bản, "
        "các thuật toán học máy phân loại như Naive Bayes, Support Vector Machine (SVM) hay Logistic Regression đều hoạt động "
        "dựa trên các phép toán đại số tuyến tính và số học xác suất. Do đó, máy tính không thể trực tiếp xử lý văn bản thô dưới dạng chuỗi ký tự "
        "mà bắt buộc phải thông qua bước 'Biểu diễn văn bản' (Text Representation) hoặc 'Trích chọn đặc trưng' (Feature Extraction). "
        "Mục tiêu cốt lõi của giai đoạn này là chuyển đổi các chuỗi bình luận tự nhiên của người dùng thành các véc-tơ số học nhiều chiều "
        "sao cho giữ lại tối đa thông tin ngữ nghĩa, cấu trúc từ vựng và sắc thái cảm xúc của câu."
    )

    add_heading_2("2.2. Mô hình Không gian Véc-tơ (Vector Space Model) và Túi từ (Bag-of-Words)")
    p = doc.add_paragraph(
        "Mô hình Không gian Véc-tơ (Vector Space Model - VSM) là một khung lý thuyết kinh điển, trong đó mỗi văn bản được biểu diễn "
        "bằng một điểm hoặc một véc-tơ trong không gian nhiều chiều, với mỗi chiều tương ứng với một từ vựng độc lập trong từ điển chung (Vocabulary). "
        "Mô hình Túi từ (Bag-of-Words - BoW) là dạng triển khai cơ bản nhất của VSM:"
    )
    p_b1 = doc.add_paragraph(style='List Bullet')
    p_b1.add_run("Nguyên lý: ").bold = True
    p_b1.add_run("Xem văn bản như một 'túi' chứa các từ rời rạc, hoàn toàn bỏ qua trật tự xuất hiện và cấu trúc ngữ pháp giữa các từ, chỉ quan tâm đến tần số đếm xuất hiện của mỗi từ.")
    p_b2 = doc.add_paragraph(style='List Bullet')
    p_b2.add_run("Hạn chế: ").bold = True
    p_b2.add_run("Các từ quá phổ biến (như 'phim', 'xem', 'rất', 'là') sẽ có tần số xuất hiện cực lớn, lấn át các từ mang sắc thái cảm xúc đặc trưng hiếm hơn. Hơn nữa, BoW đơn lẻ không bắt được các ngữ cảnh đảo ngữ như 'không hay' hay các từ ghép tiếng Việt.")

    add_heading_2("2.3. Mở rộng N-gram trong phân tích văn bản tiếng Việt")
    p = doc.add_paragraph(
        "Để khắc phục nhược điểm mất thông tin ngữ cảnh của mô hình BoW unigram truyền thống, kỹ thuật N-gram được áp dụng. "
        "Một N-gram là chuỗi liên tiếp gồm N từ (tokens) kề nhau trong văn bản:"
    )
    p_ng1 = doc.add_paragraph(style='List Bullet')
    p_ng1.add_run("Unigram (N=1): ").bold = True
    p_ng1.add_run("Từng từ đơn lẻ (ví dụ: 'phim', 'hay', 'dở', 'xúc_động'). Trong tiếng Việt, các từ ghép đã được nối bằng dấu gạch dưới (như 'thiên_đường_máu', 'phản_diện', 'uy_tín') cũng được xem là một đơn vị từ hoàn chỉnh.")
    p_ng2 = doc.add_paragraph(style='List Bullet')
    p_ng2.add_run("Bigram (N=2): ").bold = True
    p_ng2.add_run("Cặp 2 từ đi liền kề nhau (ví dụ: 'phim hay', 'đáng xem', 'không thích', 'quá đỉnh'). Kỹ thuật Bigram đặc biệt quan trọng trong bài toán phân tích cảm xúc vì giúp phân biệt sự phủ định (phủ định + tính từ tích cực = sắc thái tiêu cực).")
    p = doc.add_paragraph(
        "Việc kết hợp đồng thời Unigram và Bigram (ngram_range=(1, 2)) cho phép mô hình học được cả các từ vựng cốt lõi "
        "lẫn các cụm từ kết hợp mang sắc thái biểu cảm mạnh mẽ trong bình luận của khán giả."
    )

    add_heading_2("2.4. Phương pháp TF-IDF (Term Frequency - Inverse Document Frequency)")
    p = doc.add_paragraph(
        "TF-IDF là một thước đo thống kê định lượng đánh giá mức độ quan trọng của một từ hoặc n-gram đối với một tài liệu cụ thể "
        "trong tương quan với toàn bộ tập văn bản (Corpus). Giá trị TF-IDF được tính bằng tích của hai thành phần:"
    )
    p = doc.add_paragraph()
    r_f = p.add_run("TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)")
    r_f.bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph(
        "Trong đó:\n"
        "1. Term Frequency (TF): Đo tần suất xuất hiện của từ t trong tài liệu d. Nhằm tránh hiện tượng tài liệu dài chứa nhiều từ lặp lại làm bóp méo trọng số, "
        "kỹ thuật Sublinear TF Scaling được áp dụng: TF(t, d) = 1 + ln(f(t, d)) khi f(t, d) > 0, ngược lại bằng 0.\n"
        "2. Inverse Document Frequency (IDF): Đo mức độ 'đặc biệt' hoặc độ hiếm của từ t trên toàn bộ tập tài liệu D gồm N văn bản. "
        "Công thức IDF làm mịn (Smooth IDF) của scikit-learn được định nghĩa:\n"
    )
    p = doc.add_paragraph()
    r_idf = p.add_run("IDF(t) = ln((1 + N) / (1 + DF(t))) + 1")
    r_idf.bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph(
        "Với DF(t) là số lượng văn bản có chứa từ t. Nếu một từ xuất hiện ở hầu hết tất cả các bình luận (như 'phim', 'xem'), "
        "DF(t) tiệm cận N dẫn đến IDF(t) xấp xỉ 1 (rất nhỏ), làm giảm trọng số của từ đó. Ngược lại, nếu một từ cảm xúc đặc thù "
        "(như 'cảm_động', 'xuất_sắc', 'phí_tiền', 'dở_tệ') chỉ xuất hiện ở một số bình luận mang sắc thái rõ ràng, "
        "giá trị IDF(t) sẽ rất cao, giúp thuật toán phân loại dễ dàng nắm bắt."
    )
    p = doc.add_paragraph(
        "Cuối cùng, mỗi véc-tơ tài liệu sau khi nhân TF và IDF được chuẩn hóa theo độ dài Euclid (L2-norm), "
        "đảm bảo các bình luận dài và ngắn đều có chuẩn véc-tơ bằng 1, giúp thuật toán Naive Bayes và SVM "
        "không bị ảnh hưởng bởi độ dài bình luận."
    )

    add_heading_2("2.5. So sánh TF-IDF với Bag-of-Words và lý do lựa chọn cho Đề tài")
    p = doc.add_paragraph(
        "Trong đề tài phân tích cảm xúc bình luận phim 'Trại buôn người', chúng tôi lựa chọn TF-IDF (kết hợp n-gram 1-2) làm phương pháp trích chọn đặc trưng chủ đạo vì các ưu điểm vượt trội sau:\n"
        "1. Triệt tiêu nhiễu hiệu quả: Giảm trọng số của các từ ngữ xuất hiện quá phổ biến nhưng không mang tính khu biệt cảm xúc.\n"
        "2. Tương thích lý tưởng với Naive Bayes và SVM: Với Naive Bayes (MultinomialNB / ComplementNB), ma trận trọng số không âm phản ánh xác suất đặc trưng rất tốt. Với Linear SVM, không gian véc-tơ chuẩn hóa L2 của TF-IDF tạo ra siêu phẳng phân chia (separating hyperplane) tối ưu nhất trong không gian nhiều chiều.\n"
        "3. Tiết kiệm tài nguyên tính toán: So với các mô hình Word Embeddings phức tạp (như Word2Vec hay BERT) đòi hỏi hạ tầng GPU và dữ liệu hàng trăm nghìn mẫu, TF-IDF vận hành nhanh, minh bạch, có thể giải thích được đặc trưng (interpretable) và hoàn toàn phù hợp với quy mô hơn 2.000 mẫu dữ liệu thực tế thu thập từ Facebook."
    )

    # ==========================================
    # PHẦN 2: CHƯƠNG 3 - MỤC 3.2: THỰC NGHIỆM VÀ VECTOR HÓA DỮ LIỆU
    # ==========================================
    add_heading_1("CHƯƠNG 3: THỰC NGHIỆM VÀ KẾT QUẢ")
    add_heading_2("3.2. Thực nghiệm và Vector hóa dữ liệu (Phần thực hiện của Thắng)")

    add_heading_3("3.2.1. Tiếp nhận và mô tả tập dữ liệu sạch")
    p = doc.add_paragraph(
        "Dữ liệu đầu vào tiếp nhận từ phần tiền xử lý của Tuấn là file 'final_labeled_clean_v2.xlsx'. "
        "Bộ dữ liệu bao gồm 2.087 bình luận sạch, không có giá trị khuyết thiếu (null = 0), "
        "với 3 trường dữ liệu chính: 'Comment ID', 'Clean Text' (văn bản đã được chuẩn hóa Unicode, viết thường, "
        "tách từ tiếng Việt nối dấu gạch dưới, mã hóa emoji) và 'Final Label' (nhãn cảm xúc đã được gán nhãn thống nhất)."
    )

    # Bảng phân bố nhãn
    table_label = doc.add_table(rows=4, cols=4)
    table_label.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["STT", "Nhãn cảm xúc (Label)", "Số lượng mẫu", "Tỷ lệ phần trăm (%)"]
    for j, h in enumerate(headers):
        cell = table_label.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_background(cell, "EAECEE")

    data_rows = [
        ["1", "Positive (Tích cực)", "1.280", "61,33%"],
        ["2", "Neutral (Trung tính)", "438", "20,99%"],
        ["3", "Negative (Tiêu cực)", "369", "17,68%"]
    ]
    for i, row in enumerate(data_rows):
        for j, val in enumerate(row):
            cell = table_label.cell(i + 1, j)
            cell.text = val
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_caption("Bảng 3.1: Thống kê phân bố nhãn cảm xúc trên toàn bộ tập dữ liệu")

    add_heading_3("3.2.2. Chiến lược phân chia tập dữ liệu Train / Test phân tầng (Stratified Split)")
    p = doc.add_paragraph(
        "Để đảm bảo tính khoa học và ngăn chặn hiện tượng rò rỉ dữ liệu (Data Leakage) — một lỗi phổ biến khi trích chọn đặc trưng văn bản, "
        "chúng tôi thực hiện quy trình phân chia tập dữ liệu TRƯỚC khi huấn luyện bộ vectorizer:\n"
        "1. Tỷ lệ phân chia: 80% cho tập huấn luyện (Train set) và 20% cho tập kiểm thử (Test set).\n"
        "2. Phân tầng (Stratified): Do tập dữ liệu có sự chênh lệch nhãn (Positive chiếm 61,33%), kỹ thuật Stratified K-Split "
        "được sử dụng để đảm bảo tỷ lệ giữa 3 nhãn cảm xúc ở tập Train và tập Test là hoàn toàn đồng nhất.\n"
        "3. Cố định hạt giống ngẫu nhiên (random_state=42): Đảm bảo việc tái lập kết quả thử nghiệm 100% giữa hai người huấn luyện mô hình là Hiếu (Naive Bayes) và Doanh (SVM)."
    )

    # Bảng chia train test
    table_split = doc.add_table(rows=5, cols=5)
    table_split.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_split = ["Tập dữ liệu", "Negative (0)", "Neutral (1)", "Positive (2)", "Tổng số mẫu"]
    for j, h in enumerate(headers_split):
        cell = table_split.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_background(cell, "EAECEE")

    split_rows = [
        ["Tập Train (80%)", "295", "350", "1.024", "1.669 (80,0%)"],
        ["Tập Test (20%)", "74", "88", "256", "418 (20,0%)"],
        ["Tổng cộng", "369", "438", "1.280", "2.087 (100%)"],
        ["Tỷ lệ thực tế", "17,68%", "20,99%", "61,33%", "100%"]
    ]
    for i, row in enumerate(split_rows):
        for j, val in enumerate(row):
            cell = table_split.cell(i + 1, j)
            cell.text = val
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if i >= 2:
                cell.paragraphs[0].runs[0].bold = True

    add_caption("Bảng 3.2: Thống kê số lượng mẫu phân tầng theo tập Train và Test")

    add_heading_3("3.2.3. Cấu hình tham số của hệ thống Vector hóa TF-IDF")
    p = doc.add_paragraph(
        "Bộ trích chọn đặc trưng TF-IDF được xây dựng trên thư viện scikit-learn với cấu hình tham số được tinh chỉnh tối ưu:"
    )
    p_cf1 = doc.add_paragraph(style='List Bullet')
    p_cf1.add_run("ngram_range=(1, 2): ").bold = True
    p_cf1.add_run("Trích xuất cả unigram và bigram, giúp bắt trọn vẹn ngữ nghĩa từ ghép và cụm từ phủ định.")
    p_cf2 = doc.add_paragraph(style='List Bullet')
    p_cf2.add_run("min_df=2: ").bold = True
    p_cf2.add_run("Chỉ giữ lại các từ/cụm từ xuất hiện ở tối thiểu 2 bình luận trong tập Train. Loại bỏ các từ gõ sai chính tả hy hữu hoặc nhiễu không có giá trị học máy.")
    p_cf3 = doc.add_paragraph(style='List Bullet')
    p_cf3.add_run("sublinear_tf=True: ").bold = True
    p_cf3.add_run("Áp dụng biến đổi logarit 1 + log(tf) cho tần số xuất hiện của từ, giúp bình luận có từ lặp lại nhiều lần không áp đảo điểm số.")
    p_cf4 = doc.add_paragraph(style='List Bullet')
    p_cf4.add_run("Quy trình Fit & Transform nghiêm ngặt: ").bold = True
    p_cf4.add_run("Chỉ thực hiện fit() trên tập Train (1.669 văn bản). Sau đó dùng chính từ vựng và IDF đã học để transform() cho tập Test (418 văn bản). Tuyệt đối không fit lại trên tập Test.")

    add_heading_3("3.2.4. Kết quả không gian véc-tơ và độ thưa của ma trận")
    p = doc.add_paragraph(
        "Sau khi thực hiện trích chọn đặc trưng, các kết quả định lượng thu được như sau:\n"
        "1. Kích thước không gian đặc trưng (Vocabulary Size): Hệ thống đã trích xuất được 5.355 đặc trưng n-gram hợp lệ.\n"
        "2. Kích thước ma trận Train (X_train_tfidf): (1.669 × 5.355), tức 1.669 bình luận được biểu diễn trong không gian 5.355 chiều.\n"
        "3. Kích thước ma trận Test (X_test_tfidf): (418 × 5.355).\n"
        "4. Độ thưa của ma trận (Sparsity): 99,57%. Tức là trong ma trận đặc trưng, có tới 99,57% giá trị bằng 0 (do mỗi bình luận thực tế chỉ chứa từ 10 đến 30 từ trong tổng số 5.355 từ của từ điển). "
        "Vì vậy, chúng tôi lưu trữ dữ liệu dưới định dạng ma trận nén CSR (Compressed Sparse Row) thông qua thư viện scipy.sparse, giúp tiết kiệm bộ nhớ RAM và tăng tốc độ nhân ma trận khi huấn luyện mô hình lên gấp nhiều lần."
    )

    # Chèn biểu đồ 1
    chart1_path = os.path.join("reports", "figures", "top_20_tfidf_overall.png")
    if os.path.exists(chart1_path):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_before = Pt(6)
        doc.add_picture(chart1_path, width=Inches(6.0))
        add_caption("Hình 3.1: Biểu đồ Top 20 đặc trưng có trọng số TF-IDF trung bình cao nhất trên tập Train")

    add_heading_3("3.2.5. Phân tích ngữ nghĩa đặc trưng theo từng nhãn cảm xúc")
    p = doc.add_paragraph(
        "Để đánh giá chất lượng của bộ trích chọn đặc trưng trước khi bàn giao cho nhánh mô hình, "
        "chúng tôi tính toán trọng số TF-IDF trung bình của từng từ vựng tách biệt theo từng nhóm nhãn cảm xúc "
        "(Positive, Neutral, Negative). Kết quả phân tích thể hiện sự tương quan rõ nét giữa đặc trưng và sắc thái:"
    )

    # Chèn biểu đồ 2
    chart2_path = os.path.join("reports", "figures", "top_features_by_sentiment.png")
    if os.path.exists(chart2_path):
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.paragraph_format.space_before = Pt(6)
        doc.add_picture(chart2_path, width=Inches(6.2))
        add_caption("Hình 3.2: Top 15 đặc trưng TF-IDF tiêu biểu nhất theo 3 nhóm nhãn cảm xúc")

    p_an1 = doc.add_paragraph(style='List Bullet')
    p_an1.add_run("Nhóm Positive (Tích cực): ").bold = True
    p_an1.add_run("Chiếm ưu thế tuyệt đối bởi các từ khen ngợi chất lượng phim: 'hay' (trọng số cao nhất 0.0834), 'phim hay' (0.0294), 'xem', 'đi', 'coi', 'luôn', 'lắm', 'quá'. Điều này phản ánh rõ ràng sự hưởng ứng nhiệt tình của khán giả đối với tác phẩm 'Trại buôn người'.")
    p_an2 = doc.add_paragraph(style='List Bullet')
    p_an2.add_run("Nhóm Negative (Tiêu cực): ").bold = True
    p_an2.add_run("Xuất hiện các từ mang sắc thái thất vọng hoặc chê bai: 'dở' (0.0180), 'thiên_đường_máu' (phim cạnh tranh/so sánh tiêu cực), 'làm', 'nó', 'nhiều', 'thấy'. Các đặc trưng này phản ánh những bình luận chê kịch bản hoặc diễn xuất.")
    p_an3 = doc.add_paragraph(style='List Bullet')
    p_an3.add_run("Nhóm Neutral (Trung tính): ").bold = True
    p_an3.add_run("Tập trung vào các từ ngữ thảo luận thông tin, rủ rê đi xem hoặc hỏi đáp: 'có' (0.0335), 'bạn', 'đi', 'người', 'em', 'rồi', 'không'. Đây là các bình luận không thể hiện rõ thái độ khen hay chê.")

    add_heading_3("3.2.6. Quy chuẩn đóng gói và bàn giao sản phẩm cho Hiếu và Doanh")
    p = doc.add_paragraph(
        "Nhằm tạo điều kiện thuận lợi tối đa cho 2 thành viên nhánh mô hình là Hiếu (huấn luyện Naive Bayes) "
        "và Doanh (huấn luyện SVM), toàn bộ sản phẩm của Nhiệm vụ 2 đã được chuẩn hóa và đóng gói tại thư mục Data/vectorized/ bao gồm:\n"
        "1. File gói toàn diện 'vectorized_dataset.joblib': Chứa sẵn toàn bộ các ma trận X_train_tfidf, X_test_tfidf, X_train_bow, X_test_bow, nhãn y_train, y_test, bảng tra cứu nhãn label_mapping và danh sách tên đặc trưng.\n"
        "2. File mô hình Vectorizer 'tfidf_vectorizer.joblib': Cho phép nạp lại để vector hóa trực tiếp các câu bình luận mới trong tương lai.\n"
        "3. Bộ ma trận thưa 'X_train_tfidf.npz', 'X_test_tfidf.npz' và vector nhãn 'y_train.npy', 'y_test.npy': Định dạng chuẩn giúp đọc cực nhanh trên mọi nền tảng.\n"
        "4. Bảng Excel chia tách 'train_split.xlsx' và 'test_split.xlsx': Dành cho việc tra cứu thủ công và kiểm thử chéo văn bản gốc.\n"
        "5. File mã nguồn mẫu 'src/demo_load_data.py': Cung cấp code mẫu nạp dữ liệu chỉ với 1 dòng lệnh, đã được chạy thử nghiệm thành công trên cả 2 mô hình Naive Bayes (Accuracy mẫu ~65.6%) và Linear SVM (Accuracy mẫu ~71.5%)."
    )

    # LỜI KẾT NHIỆM VỤ 2
    add_heading_2("3.2.7. Kết luận và đánh giá hoàn thành nhiệm vụ")
    p = doc.add_paragraph(
        "Nhiệm vụ số 2 của nhóm (Trích chọn đặc trưng và Vector hóa dữ liệu) do Thắng phụ trách đã được hoàn thành 100% "
        "đúng kế hoạch, đáp ứng nghiêm ngặt các quy chuẩn kỹ thuật của bộ môn Phân tích và Khai phá dữ liệu văn bản. "
        "Dữ liệu đầu ra sạch, không gian đặc trưng chất lượng cao, không bị rò rỉ thông tin và sẵn sàng cho giai đoạn 3 (Xây dựng mô hình học máy)."
    )

    output_path = os.path.join("reports", "Bao_cao_Nhiem_vu_2_Thang_Trich_chon_dac_trung.docx")
    doc.save(output_path)
    print(f"Đã xuất báo cáo Word thành công tại: {output_path}")

if __name__ == "__main__":
    create_report()

