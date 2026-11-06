import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_guide_doc(output_path="HUONG_DAN_DU_LIEU.docx"):
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("HƯỚNG DẪN QUẢN LÝ DỮ LIỆU & QUY TRÌNH CHẠY PIPELINE NLP")
    r_title.font.name = 'Calibri'
    r_title.font.size = Pt(20)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79) # Deep Navy

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(20)
    r_sub = p_sub.add_run("Tài liệu kỹ thuật: Nơi lưu trữ dữ liệu, cách vận hành mã nguồn và kết quả đầu ra")
    r_sub.font.italic = True
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Callout Box: Note on Git
    table_note = doc.add_table(rows=1, cols=1)
    table_note.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_note = table_note.cell(0, 0)
    set_cell_background(c_note, "EBF1F5") # Soft blue grey
    set_cell_margins(c_note, top=140, bottom=140, left=200, right=200)
    c_note.width = Inches(6.5)
    p_note = c_note.paragraphs[0]
    p_note.paragraph_format.space_after = Pt(0)
    r_note_b = p_note.add_run("LƯU Ý QUAN TRỌNG VỀ GIT: ")
    r_note_b.bold = True
    r_note_b.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    r_note_t = p_note.add_run(
        "Toàn bộ dữ liệu thô (.csv, .xlsx, .json) và dữ liệu kết quả được cấu hình KHÔNG đưa lên Git (.gitignore) "
        "để giữ mã nguồn nhẹ và sạch. Thư mục data được giữ khung cấu trúc trên Git qua các tệp .gitkeep. "
        "Dữ liệu thực tế trên máy tính của bạn vẫn được bảo toàn nguyên vẹn."
    )
    r_note_t.font.size = Pt(10)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 1. Cấu trúc thư mục
    h1 = doc.add_heading(level=1)
    r = h1.add_run("1. Cấu Trúc Thư Mục Lưu Trữ Dữ Liệu")
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    r.font.size = Pt(14)
    r.bold = True

    p_desc1 = doc.add_paragraph(
        "Dữ liệu của dự án được phân định rành mạch giữa 2 nguồn Facebook và TikTok với các cấp độ xử lý:"
    )
    p_desc1.paragraph_format.space_after = Pt(8)

    # Table for Directories
    dir_table = doc.add_table(rows=1, cols=3)
    dir_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = dir_table.rows[0].cells
    hdr_titles = ["Đường dẫn thư mục", "Mục đích sử dụng", "Tệp tin chứa"]
    for i, title in enumerate(hdr_titles):
        set_cell_background(hdr_cells[i], "1F4E79")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(title)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10.5)

    dir_rows = [
        ("data/tiktok/raw/", "Nơi chứa các file CSV xuất từ TikTok", "TTCommentExporter-*.csv (Tệp dữ liệu thô)"),
        ("data/tiktok/", "Nơi lưu bảng dữ liệu gộp & kết quả làm sạch", "tiktok_comments_merged.xlsx\ntiktok_comments_clean.xlsx"),
        ("data/facebook/raw/", "Nơi chứa dữ liệu Facebook crawl ban đầu", "simple_post_excel/*.xlsx\nsimple_post_comments_merged.xlsx"),
        ("data/facebook/intermediate/", "Dữ liệu trung gian & checkpoint gán nhãn", "all_comments_annotated.xlsx\nall_comments_reviewed_v2.xlsx\ncheckpoint_*.json"),
        ("data/facebook/processed/", "Nơi lưu trữ kết quả cuối cùng hoàn thiện", "final_labeled_clean_v3.xlsx (Dữ liệu chuẩn phục vụ mô hình)"),
    ]

    for row_idx, (path, purpose, files) in enumerate(dir_rows):
        row = dir_table.add_row()
        bg_color = "F7F9FB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate([path, purpose, files]):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            p = cell.paragraphs[0]
            run = p.add_run(text)
            run.font.size = Pt(9.5)
            if col_idx == 0:
                run.bold = True
                run.font.name = "Consolas"
                run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 2. Quy trình xử lý TikTok
    h2 = doc.add_heading(level=1)
    r = h2.add_run("2. Quy Trình Vận Hành Dữ Liệu TikTok (Từ CSV)")
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    r.font.size = Pt(14)
    r.bold = True

    p = doc.add_paragraph()
    r = p.add_run("Bước 2.1: Thu thập & Đặt dữ liệu CSV gốc vào đúng thư mục\n")
    r.bold = True
    r.font.color.rgb = RGBColor(0x2B, 0x57, 0x9A)
    p.add_run(
        "• Sử dụng công cụ xuất bình luận (ví dụ: TikTok Comment Exporter extension) để tải comment từ các video.\n"
        "• Toàn bộ các file CSV vừa tải về (ví dụ: TTCommentExporter-7540245360342469895-160-comments.csv, ...) "
        "hãy copy trực tiếp vào thư mục: data/tiktok/raw/."
    )

    p = doc.add_paragraph()
    r = p.add_run("Bước 2.2: Gộp dữ liệu tự động & Khử trùng lặp (Deduplication)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(0x2B, 0x57, 0x9A)
    p.add_run("Mở terminal hoặc PowerShell tại thư mục gốc của dự án và chạy lệnh:\n")
    r_cmd1 = p.add_run("    python merge_tiktok_comments.py\n")
    r_cmd1.font.name = "Consolas"
    r_cmd1.font.size = Pt(10)
    r_cmd1.bold = True
    r_cmd1.font.color.rgb = RGBColor(0x99, 0x00, 0x00)
    p.add_run(
        "• Hoạt động: Script tự động quét tất cả file CSV trong data/tiktok/raw/, đọc định dạng, chuẩn hóa thời gian, "
        "và lọc trùng hoàn toàn theo ID/nội dung bình luận.\n"
        "• Kết quả sinh ra: data/tiktok/tiktok_comments_merged.xlsx và data/tiktok/tiktok_comments_merged.csv."
    )

    p = doc.add_paragraph()
    r = p.add_run("Bước 2.3: Tiền xử lý NLP chuyên sâu cho TikTok\n")
    r.bold = True
    r.font.color.rgb = RGBColor(0x2B, 0x57, 0x9A)
    p.add_run("Chạy tiếp lệnh tiền xử lý:\n")
    r_cmd2 = p.add_run("    python preprocess_tiktok_comments.py\n")
    r_cmd2.font.name = "Consolas"
    r_cmd2.font.size = Pt(10)
    r_cmd2.bold = True
    r_cmd2.font.color.rgb = RGBColor(0x99, 0x00, 0x00)
    p.add_run(
        "• Hoạt động: Chuẩn hóa ngữ nghĩa, dịch từ điển Teencode/viết tắt/slang, bảo toàn cấu trúc câu phủ định, "
        "chuẩn hóa emoji, và phân tách từ tiếng Việt tự động bằng PyVi.\n"
        "• Kết quả sinh ra: data/tiktok/tiktok_comments_clean.xlsx."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 3. Quy trình xử lý Facebook
    h3 = doc.add_heading(level=1)
    r = h3.add_run("3. Quy Trình Vận Hành Dữ Liệu Facebook")
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    r.font.size = Pt(14)
    r.bold = True

    p = doc.add_paragraph()
    r = p.add_run("Bước 3.1: Đặt dữ liệu thô Facebook\n")
    r.bold = True
    r.font.color.rgb = RGBColor(0x2B, 0x57, 0x9A)
    p.add_run("• Đặt các file crawl thô vào data/facebook/raw/simple_post_excel/.\n")
    p.add_run("• Chạy lệnh gộp: ")
    r_cmd_fb = p.add_run("python clean_merge_comments.py\n")
    r_cmd_fb.font.name = "Consolas"
    r_cmd_fb.bold = True
    p.add_run("• Kết quả sinh ra: data/facebook/raw/simple_post_comments_merged.xlsx.")

    p = doc.add_paragraph()
    r = p.add_run("Bước 3.2: Gán nhãn cảm xúc LLM & Kiểm duyệt (Tuỳ chọn)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(0x2B, 0x57, 0x9A)
    p.add_run("• Chạy gán nhãn: ")
    r_lbl = p.add_run("python label_all_comments.py\n")
    r_lbl.font.name = "Consolas"
    r_lbl.bold = True
    p.add_run("• Chạy rà soát, khử nhiễu: ")
    r_rev = p.add_run("python review_annotations_full.py\n")
    r_rev.font.name = "Consolas"
    r_rev.bold = True
    p.add_run("• Toàn bộ tiến độ lưu theo checkpoint an toàn trong data/facebook/intermediate/.")

    p = doc.add_paragraph()
    r = p.add_run("Bước 3.3: Tiền xử lý NLP tạo tập dữ liệu huấn luyện cuối cùng\n")
    r.bold = True
    r.font.color.rgb = RGBColor(0x2B, 0x57, 0x9A)
    p.add_run("• Chạy lệnh: ")
    r_v3 = p.add_run("python process_clean_v3.py\n")
    r_v3.font.name = "Consolas"
    r_v3.bold = True
    p.add_run(
        "• Kết quả cuối cùng: data/facebook/processed/final_labeled_clean_v3.xlsx "
        "(Tập dữ liệu chuẩn mực nhất đã gồm cả nhãn cảm xúc và các cột tiền xử lý NLP)."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 4. Ý nghĩa các cột dữ liệu đầu ra
    h4 = doc.add_heading(level=1)
    r = h4.add_run("4. Ý Nghĩa Chi Tiết Các Cột Dữ Liệu Đầu Ra")
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    r.font.size = Pt(14)
    r.bold = True

    col_table = doc.add_table(rows=1, cols=3)
    col_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = col_table.rows[0].cells
    for i, title in enumerate(["Tên cột", "Đặc điểm xử lý", "Mô hình / Mục đích phù hợp"]):
        set_cell_background(hdr[i], "1F4E79")
        set_cell_margins(hdr[i], top=120, bottom=120, left=150, right=150)
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(title)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10.5)

    col_rows = [
        (
            "Clean Text",
            "Đã giải mã teencode, chuẩn hóa chính tả, đưa về chữ thường.",
            "Báo cáo người dùng, trực quan hóa từ vựng (Word Cloud), đối soát."
        ),
        (
            "Text_PhoBERT",
            "Tách từ tiếng Việt bằng dấu gạch dưới (ví dụ: 'bộ_phim', 'rất_hay'). GIỮ NGUYÊN dấu câu, từ dừng và cấu trúc cú pháp.",
            "Mô hình Transformer / PhoBERT / RoBERTa (Cơ chế Attention cần ngữ cảnh dấu câu và liên kết ngữ pháp tự nhiên)."
        ),
        (
            "Text_TFIDF",
            "Tách từ tiếng Việt bằng gạch dưới ('_'). LOẠI BỎ sạch toàn bộ dấu câu, ký tự đặc biệt và danh sách từ dừng (Stopwords).",
            "Trích xuất đặc trưng TF-IDF, CountVectorizer, huấn luyện mô hình Machine Learning cổ điển (SVM, Naive Bayes, Logistic Regression, Random Forest)."
        ),
        (
            "Final Label",
            "Nhãn cảm xúc chuẩn hóa: Positive, Negative, Neutral sau khi qua thẩm định logic và cross-check.",
            "Cột nhãn mục tiêu (Ground Truth Label) để huấn luyện và tính toán Accuracy, F1-Score."
        ),
    ]

    for row_idx, (cname, cdesc, cusage) in enumerate(col_rows):
        row = col_table.add_row()
        bg_color = "F7F9FB" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate([cname, cdesc, cusage]):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            p = cell.paragraphs[0]
            run = p.add_run(text)
            run.font.size = Pt(9.5)
            if col_idx == 0:
                run.bold = True
                run.font.name = "Consolas"
                run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    # Save
    doc.save(output_path)
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    create_guide_doc("HUONG_DAN_DU_LIEU.docx")
