"""
Script tổng hợp và làm sạch dữ liệu bình luận TikTok từ các file xuất CSV.
Đầu vào: data/tiktok/raw/TTCommentExporter-*.csv
Đầu ra: data/tiktok/tiktok_comments_merged.xlsx và data/tiktok/tiktok_comments_merged.csv
"""

import re
import sys
from pathlib import Path
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
TIKTOK_RAW_DIR = ROOT_DIR / "data" / "tiktok" / "raw"
OUTPUT_EXCEL = ROOT_DIR / "data" / "tiktok" / "tiktok_comments_merged.xlsx"
OUTPUT_CSV = ROOT_DIR / "data" / "tiktok" / "tiktok_comments_merged.csv"


def style_worksheet(ws, cols_count: int, row_count: int):
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


def main():
    print("=" * 80)
    print("BẮT ĐẦU GỘP DỮ LIỆU BÌNH LUẬN TIKTOK")
    print("=" * 80)

    csv_files = sorted(list(TIKTOK_RAW_DIR.glob("TTCommentExporter*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào tại: {TIKTOK_RAW_DIR}")

    print(f"Tìm thấy {len(csv_files)} file CSV xuất TikTok.")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(f, encoding="utf-8")

        # Tách Video ID từ tên file: TTCommentExporter-{VideoID}-{count}-comments.csv
        match = re.search(r"TTCommentExporter-(\d+)", f.name)
        video_id = match.group(1) if match else ""
        df["Video ID"] = str(video_id)
        df["File nguồn"] = f.name
        dfs.append(df)
        print(f"  -> {f.name}: {len(df)} dòng")

    combined = pd.concat(dfs, ignore_index=True)
    total_raw = len(combined)
    print(f"\nTổng số dòng ban đầu trước khi khử trùng lặp: {total_raw:,} dòng")

    # Đổi tên cột chuẩn hóa
    col_mapping = {
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
    }
    combined = combined.rename(columns=col_mapping)

    # Khử trùng lặp theo Comment ID (tránh file tải đúp (1))
    id_col = "Comment ID"
    combined[id_col] = combined[id_col].astype(str)
    combined = combined.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    print(f"Số dòng sau khi khử trùng lặp theo Comment ID: {len(combined):,} dòng (loại {total_raw - len(combined)} dòng trùng)")

    # Loại bỏ dòng không có nội dung bình luận
    combined = combined.dropna(subset=["Nội dung"]).reset_index(drop=True)
    combined["Nội dung"] = combined["Nội dung"].astype(str).str.strip()
    combined = combined[combined["Nội dung"] != ""].reset_index(drop=True)

    # Thứ tự cột xuất ra
    output_cols = [
        "Comment ID",
        "Video ID",
        "User ID",
        "Username",
        "Nickname",
        "Nội dung",
        "Thời gian",
        "Lượt thích",
        "Tác giả thích",
        "Số phản hồi",
        "Đã ghim",
        "User URL",
        "File nguồn",
    ]
    # Giữ lại các cột có trong DataFrame
    output_cols = [c for c in output_cols if c in combined.columns]
    combined = combined[output_cols]

    # Xuất ra CSV và Excel
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[XUẤT FILE CSV] Đã lưu tại: {OUTPUT_CSV}")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Du_lieu_TikTok", index=False)

    wb = openpyxl.load_workbook(OUTPUT_EXCEL)
    style_worksheet(wb["Du_lieu_TikTok"], cols_count=len(output_cols), row_count=len(combined))
    wb.save(OUTPUT_EXCEL)
    print(f"[XUẤT FILE EXCEL] Đã lưu tại: {OUTPUT_EXCEL}")
    print(f"Tổng số bình luận TikTok hoàn chỉnh: {len(combined):,} dòng.")


if __name__ == "__main__":
    main()

