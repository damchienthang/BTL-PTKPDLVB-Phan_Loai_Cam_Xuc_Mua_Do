import os
import shutil
import sys
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from preprocess_comments import VietnameseCommentPreprocessor, ENTITY_DICT
_inter_dir = ROOT_DIR / "data" / "facebook" / "intermediate"
TARGET_FILE = _inter_dir / "final_labeled.xlsx"
BACKUP_FILE = _inter_dir / "final_labeled_backup.xlsx"
ANNOTATED_FILE = _inter_dir / "all_comments_annotated.xlsx"


def load_author_names() -> list:
    """
    Tải danh sách tên tác giả gốc để bóc tách tên người dùng ở đầu câu Reply.
    Lọc bỏ các từ trùng với đại từ nhân xưng, từ vựng thông dụng và tên diễn viên.
    """
    if not ANNOTATED_FILE.exists():
        print(f"[Cảnh báo] Không tìm thấy file {ANNOTATED_FILE.name}, bỏ qua bóc tách tên tác giả.")
        return []

    df_ann = pd.read_excel(ANNOTATED_FILE, sheet_name="ALL_ANNOTATED")
    authors = set(df_ann["Tên tác giả"].dropna().astype(str).str.strip())

    # Danh sách từ loại trừ: đại từ xưng hô, động từ/tính từ thông dụng
    stop_words = {
        "Anh", "Em", "Tôi", "Bạn", "Chị", "Ông", "Bà", "Phim", "Hay", "Đẹp",
        "Đoạn", "Xem", "Nay", "Admin", "Ad", "Cậu", "Bác", "Mày", "Tao",
    }
    # Bảo vệ tên diễn viên / phim không bị xóa nhầm
    actor_names = {
        "Hoàng Kim Ngọc", "Quách Ngọc Ngoan", "Quách Ngọc Tuyên",
        "Võ Thành Tâm", "Hải Anh Kore", "Kim Ngọc", "Ngọc Trân",
    }

    valid_authors = [
        a for a in authors
        if a not in stop_words and a not in actor_names and len(a) >= 2
    ]
    # Sắp xếp tên dài trước để ưu tiên khớp chính xác tên đầy đủ
    valid_authors.sort(key=len, reverse=True)
    return valid_authors


def clean_author_prefix(text: str, comment_type: str, valid_authors: list) -> str:
    """
    Cắt bỏ tên người dùng được Reply ở đầu bình luận (nhiễu crawl Facebook).
    """
    if not valid_authors or not text:
        return text

    t = text.strip()
    changed = True
    iterations = 0

    while changed and iterations < 3:
        changed = False
        for a in valid_authors:
            # Khớp tên ở đầu chuỗi và theo sau bởi khoảng trắng hoặc dấu ngắt câu
            if t.startswith(a) and len(t) > len(a) and t[len(a)] in (" ", ":", ",", "\n", "\t"):
                # Với tên chỉ có 1 từ, chỉ bóc tách khi dòng đó là Reply
                if len(a.split()) == 1 and comment_type != "Reply":
                    continue
                rem = t[len(a):].lstrip(":,-–— ").strip()
                if rem:  # Chỉ cắt nếu sau khi cắt câu vẫn còn nội dung
                    t = rem
                    changed = True
                    iterations += 1
                    break
    return t


def main():
    print("=" * 80)
    print("TIỀN XỬ LÝ DỮ LIỆU FILE FINAL_LABELED.XLSX — PHƯƠNG ÁN 2")
    print(" (Xóa tên người dùng ở đầu Reply + Chuẩn hóa thực thể phim/diễn viên)")
    print("=" * 80)

    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {TARGET_FILE}")

    # 1. Đọc dữ liệu từ file final_labeled.xlsx
    df = pd.read_excel(TARGET_FILE, sheet_name="FINAL_DATA")
    print(f"[Input] Đã đọc {len(df):,} bản ghi từ sheet 'FINAL_DATA'")
    print(f"[Input] Các cột hiện tại: {df.columns.tolist()}")

    # 2. Tải danh sách tên tác giả
    valid_authors = load_author_names()
    print(f"[Authors] Đã tải {len(valid_authors):,} tên tác giả hợp lệ để lọc tiền tố Reply")

    # 3. Khởi tạo bộ tiền xử lý với tính năng chuẩn hóa thực thể
    preprocessor = VietnameseCommentPreprocessor(
        lower=True,
        remove_urls=True,
        remove_mentions=True,
        normalize_teencode=True,
        normalize_entities=True,  # Bật chuẩn hóa tên diễn viên/tên phim
        handle_emoji_mode="translate",
        normalize_elongated_chars=True,
        handle_negation=True,
        segment_words=True,
        remove_stopwords=True,  # Bật lọc từ dừng an toàn theo Phương án B
    )

    # 4. Thực hiện tiền xử lý
    print("[Xử lý] Đang tiến hành bóc tách tên riêng Reply & tiền xử lý văn bản...")
    preprocessed_texts = []
    stripped_author_count = 0

    for idx, row in df.iterrows():
        raw = str(row["Text RAW"] or "")
        c_type = str(row.get("Comment Type", "") or "")

        # Bước 1: Xóa tên tác giả ở đầu câu Reply nếu có
        text_without_author = clean_author_prefix(raw, c_type, valid_authors)
        if text_without_author != raw.strip():
            stripped_author_count += 1

        # Bước 2: Chạy pipeline tiền xử lý (chuẩn hóa thực thể, emoji, teencode, word segmentation, negation)
        clean_text = preprocessor.preprocess(text_without_author)
        preprocessed_texts.append(clean_text)

    # 5. Cập nhật/thêm cột "Text Preprocessed" ngay sau "Text RAW"
    col_name = "Text Preprocessed"
    df[col_name] = preprocessed_texts

    ordered_cols = ["Row ID", "Text RAW", col_name, "Comment Type", "Label"]
    df = df[ordered_cols]

    # 6. Ghi lại vào file Excel với định dạng chuẩn
    out_file = TARGET_FILE
    try:
        # Thử mở để kiểm tra quyền ghi
        with open(TARGET_FILE, "r+b"):
            pass
    except (PermissionError, IOError):
        out_file = ROOT_DIR / "final_labeled_no_stopwords.xlsx"
        print(f"[CHÚ Ý] File '{TARGET_FILE.name}' đang được mở bởi ứng dụng Excel!")
        print(f"        Đang xuất ra file song song: '{out_file.name}' để không làm gián đoạn.")

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="FINAL_DATA", index=False)
        ws = writer.sheets["FINAL_DATA"]

        # Cố định dòng tiêu đề và bật lọc
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # Styling
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        border_thin = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9"),
        )

        for col_idx in range(1, len(ordered_cols) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Độ rộng cột
        ws.column_dimensions["A"].width = 12  # Row ID
        ws.column_dimensions["B"].width = 50  # Text RAW
        ws.column_dimensions["C"].width = 50  # Text Preprocessed
        ws.column_dimensions["D"].width = 16  # Comment Type
        ws.column_dimensions["E"].width = 16  # Label

        for row_cells in ws.iter_rows(min_row=2, max_row=len(df) + 1):
            row_cells[0].alignment = Alignment(horizontal="center", vertical="center")  # Row ID
            row_cells[1].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)  # Text RAW
            row_cells[2].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)  # Text Preprocessed
            row_cells[3].alignment = Alignment(horizontal="center", vertical="center")  # Comment Type
            row_cells[4].alignment = Alignment(horizontal="center", vertical="center")  # Label
            for c in row_cells:
                c.border = border_thin

    print(f"\n[Thành công] Đã cập nhật file: {out_file}")
    if out_file != TARGET_FILE:
        print(f"[LƯU Ý] Sau khi bạn đóng file Excel '{TARGET_FILE.name}', bạn có thể chạy lại lệnh để ghi đè vào file gốc.")
    print(f"[Thống kê] Số bình luận được bóc tách tên người dùng Reply: {stripped_author_count:,} dòng")
    print(f"[Xác nhận] Các cột: {ordered_cols}")
    print(f"[Xác nhận] Tổng số dòng: {len(df):,}")

    # 7. Hiển thị mẫu so sánh trực quan
    print("\n" + "=" * 80)
    print("MẪU SO SÁNH TRỰC QUAN TRƯỚC VÀ SAU XỬ LÝ TÊN RIÊNG:")
    print("=" * 80)
    samples_to_show = [2, 3, 9, 10, 18]
    for rid in samples_to_show:
        match_row = df[df["Row ID"] == rid]
        if not match_row.empty:
            r = match_row.iloc[0]
            print(f"\n[ID {r['Row ID']}] - Loại: {r['Comment Type']} - Nhãn: {r['Label']}")
            print(f" - [RAW]         : {r['Text RAW'].strip()}")
            print(f" - [Preprocessed]: {r[col_name]}")


if __name__ == "__main__":
    main()
