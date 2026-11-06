"""
Script gộp tất cả các file CSV bình luận TikTok từ thư mục 'data/tiktok/raw/'.
Hỗ trợ cả file xuất từ Chrome Extension lẫn file cào tự động bằng crawl_tiktok_comments.py.

Thực hiện:
1. Đọc và hợp nhất tất cả các file .csv trong data/tiktok/raw/.
2. Chuẩn hóa tên cột đồng nhất (tiếng Việt & tiếng Anh).
3. Khử trùng lặp theo Comment ID (tránh bị trùng khi cào nhiều lần).
4. Loại bỏ các dòng rác, dòng trống.
5. Xuất ra 2 file:
   - 'data/tiktok/tiktok_comments_merged.csv'
   - 'data/tiktok/tiktok_comments_merged.xlsx'
"""

import re
import sys
from pathlib import Path
import pandas as pd

# UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "tiktok" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "tiktok"
OUTPUT_CSV = OUTPUT_DIR / "tiktok_comments_merged.csv"
OUTPUT_EXCEL = OUTPUT_DIR / "tiktok_comments_merged.xlsx"


def style_worksheet(ws, cols_count: int, row_count: int):
    """Định dạng worksheet Excel chuyên nghiệp nếu có openpyxl."""
    try:
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        border_thin = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9"),
        )

        for col_idx in range(1, cols_count + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for row in ws.iter_rows(min_row=2, max_row=row_count + 1):
            for cell in row:
                cell.border = border_thin
    except Exception:
        pass


def main():
    print("=" * 80)
    print("BẮT ĐẦU GỘP DỮ LIỆU BÌNH LUẬN TIKTOK")
    print("=" * 80)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {RAW_DIR}")

    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong {RAW_DIR}")
        return

    print(f"Tìm thấy {len(csv_files)} file CSV xuất TikTok.")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(f, encoding="utf-8")
            except Exception:
                df = pd.read_csv(f, encoding="cp1258", errors="ignore")

        # Trích xuất video id từ tên file nếu chưa có cột Video ID
        match = re.search(r"TTCommentExporter-(\d+)", f.name)
        video_id = match.group(1) if match else ""
        if "Video ID" not in df.columns and "video_id" not in df.columns:
            df["Video ID"] = str(video_id)
        df["File nguồn"] = f.name
        dfs.append(df)
        print(f"  -> {f.name}: {len(df)} dòng")

    combined = pd.concat(dfs, ignore_index=True)
    total_raw = len(combined)
    print(f"\nTổng số dòng ban đầu trước khi khử trùng lặp: {total_raw:,} dòng")

    # Bản đồ đổi tên cột chuẩn hóa hỗ trợ cả Extension lẫn Scraper tự động
    col_mapping = {
        # Tiếng Việt từ Chrome Extension:
        "ID nhận xét": "Comment ID",
        "ID người dùng": "User ID",
        "Tên người dùng": "Username",
        "Tên nick": "Nickname",
        "Bình luận": "Nội dung",
        "Thời gian bình luận": "Thời gian",
        "Số lượt thích": "Lượt thích",
        "Tác giả đã thích": "Tác giả thích",
        "Số lượng trả lời": "Số phản hồi",
        "Đã ghim lên đầu": "Đã ghim",
        "Trang chủ người dùng": "User URL",
        # Tiếng Anh từ crawl_tiktok_comments.py:
        "comment_id": "Comment ID",
        "video_id": "Video ID",
        "comment_text": "Nội dung",
        "author_username": "Username",
        "author_nickname": "Nickname",
        "likes": "Lượt thích",
        "reply_total": "Số phản hồi",
        "create_time": "Thời gian",
        "reply_to": "Trả lời cho",
        "comment_type": "Loại bình luận",
    }
    combined = combined.rename(columns=col_mapping)

    # Xác định cột Comment ID
    id_col = "Comment ID" if "Comment ID" in combined.columns else combined.columns[0]
    combined[id_col] = combined[id_col].astype(str).str.strip()
    combined = combined.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    print(f"Số dòng sau khi khử trùng lặp theo Comment ID: {len(combined):,} dòng (loại {total_raw - len(combined)} dòng trùng)")

    # Loại bỏ dòng không có nội dung bình luận
    content_col = "Nội dung" if "Nội dung" in combined.columns else combined.columns[1]
    combined = combined.dropna(subset=[content_col]).reset_index(drop=True)
    combined[content_col] = combined[content_col].astype(str).str.strip()
    combined = combined[combined[content_col] != ""].reset_index(drop=True)

    # Thứ tự cột xuất ra
    preferred_cols = [
        "Comment ID",
        "Video ID",
        "User ID",
        "Username",
        "Nickname",
        "Nội dung",
        "Thời gian",
        "Lượt thích",
        "Số phản hồi",
        "Loại bình luận",
        "Trả lời cho",
        "Tác giả thích",
        "Đã ghim",
        "User URL",
        "File nguồn",
    ]
    output_cols = [c for c in preferred_cols if c in combined.columns]
    for c in combined.columns:
        if c not in output_cols:
            output_cols.append(c)
    combined = combined[output_cols]

    # Xuất ra CSV (utf-8-sig để Excel mở không lỗi phông)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[XUẤT FILE CSV] Đã lưu tại: {OUTPUT_CSV}")

    # Xuất Excel (bọc an toàn nếu thiếu openpyxl hoặc file bị khóa)
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name="Du_lieu_TikTok", index=False)
        try:
            import openpyxl
            wb = openpyxl.load_workbook(OUTPUT_EXCEL)
            style_worksheet(wb["Du_lieu_TikTok"], cols_count=len(output_cols), row_count=len(combined))
            wb.save(OUTPUT_EXCEL)
        except Exception:
            pass
        print(f"[XUẤT FILE EXCEL] Đã lưu tại: {OUTPUT_EXCEL}")
    except PermissionError:
        temp_excel = OUTPUT_DIR / "tiktok_comments_merged_new.xlsx"
        print(f"[CẢNH BÁO] File Excel đang mở, lưu tạm vào: {temp_excel}")
        combined.to_excel(temp_excel, sheet_name="Du_lieu_TikTok", index=False, engine="openpyxl")
    except (ImportError, ModuleNotFoundError):
        print("[LƯU Ý] Môi trường chưa cài 'openpyxl', tệp CSV đã sẵn sàng sử dụng.")

    print(f"Tổng số bình luận TikTok hoàn chỉnh: {len(combined):,} dòng.")


if __name__ == "__main__":
    main()
