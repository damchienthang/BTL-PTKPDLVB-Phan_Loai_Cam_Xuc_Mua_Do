"""
Script tiền xử lý dữ liệu bình luận TikTok (áp dụng bộ NLP v3 chuẩn hóa)
Đầu vào: data/tiktok/tiktok_comments_merged.xlsx
Đầu ra: data/tiktok/tiktok_comments_clean.xlsx (kèm 2 cột Text_PhoBERT và Text_TFIDF)
"""

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
sys.path.insert(0, str(ROOT_DIR))

from preprocess_comments import VietnameseCommentPreprocessor

INPUT_EXCEL = ROOT_DIR / "data" / "tiktok" / "tiktok_comments_merged.xlsx"
OUTPUT_EXCEL = ROOT_DIR / "data" / "tiktok" / "tiktok_comments_clean.xlsx"


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
    print("BẮT ĐẦU TIỀN XỬ LÝ NLP DỮ LIỆU BÌNH LUẬN TIKTOK")
    print("=" * 80)

    if not INPUT_EXCEL.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {INPUT_EXCEL}. Hãy chạy merge_tiktok_comments.py trước.")

    df = pd.read_excel(INPUT_EXCEL, sheet_name="Du_lieu_TikTok")
    print(f"Đã đọc {len(df):,} bình luận TikTok từ: {INPUT_EXCEL.name}")

    preprocessor = VietnameseCommentPreprocessor()

    print("\n--- Đang tiến hành làm sạch, chuẩn hóa teencode, emoji và tách từ ---")
    clean_texts = []
    phobert_texts = []
    tfidf_texts = []

    for idx, row in df.iterrows():
        raw = str(row.get("Nội dung", "") or "").strip()
        phobert_clean = preprocessor.preprocess_phobert(raw)
        tfidf_clean = preprocessor.preprocess_tfidf(raw)

        phobert_texts.append(phobert_clean)
        tfidf_texts.append(tfidf_clean)
        clean_texts.append(tfidf_clean)

    df["Clean Text"] = clean_texts
    df["Text_PhoBERT"] = phobert_texts
    df["Text_TFIDF"] = tfidf_texts

    # Khử trùng lặp nội dung sạch nếu có
    total_before = len(df)
    df_clean = df.drop_duplicates(subset=["Clean Text"], keep="first").reset_index(drop=True)
    dups_removed = total_before - len(df_clean)
    if dups_removed > 0:
        print(f"  -> Đã loại bỏ {dups_removed} bình luận bị trùng lặp nội dung sạch.")

    # Loại bỏ câu rỗng sau khi làm sạch
    df_clean = df_clean[df_clean["Clean Text"] != ""].reset_index(drop=True)

    print(f"\n--- Xuất file kết quả tiền xử lý: {OUTPUT_EXCEL.name} ---")
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            df_clean.to_excel(writer, sheet_name="TIKTOK_CLEAN", index=False)
        wb = openpyxl.load_workbook(OUTPUT_EXCEL)
        style_worksheet(wb["TIKTOK_CLEAN"], cols_count=len(df_clean.columns), row_count=len(df_clean))
        wb.save(OUTPUT_EXCEL)
        print(f"[THÀNH CÔNG] Đã lưu file Excel tại: {OUTPUT_EXCEL}")
    except PermissionError:
        backup = ROOT_DIR / "data" / "tiktok" / "tiktok_comments_clean_new.xlsx"
        print(f"[CẢNH BÁO] File đang mở trong Excel! Lưu tạm vào: {backup.name}")
        with pd.ExcelWriter(backup, engine="openpyxl") as writer:
            df_clean.to_excel(writer, sheet_name="TIKTOK_CLEAN", index=False)
        wb = openpyxl.load_workbook(backup)
        style_worksheet(wb["TIKTOK_CLEAN"], cols_count=len(df_clean.columns), row_count=len(df_clean))
        wb.save(backup)
        print(f"[THÀNH CÔNG] Đã lưu tại: {backup}")

    print(f"Tổng số bình luận TikTok sau tiền xử lý: {len(df_clean):,} dòng sẵn sàng đưa vào gán nhãn/huấn luyện.")


if __name__ == "__main__":
    main()

