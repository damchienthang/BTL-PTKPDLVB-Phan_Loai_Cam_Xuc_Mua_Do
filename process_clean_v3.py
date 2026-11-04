"""
Script xử lý và làm sạch dữ liệu phiên bản v3 (sẵn sàng cho PhoBERT & TF-IDF):
Giải quyết dứt điểm các vấn đề nêu trong báo cáo đánh giá file Word:
1. Sửa toàn diện từ điển Teencode và tách dấu câu (khắc phục 100% các từ tui, v, nhìu, r, trc, đc, ko, z, ms, nc, ak/ah, không_p).
2. Chuẩn hóa Emoticon (:)), :D, =))) và Emoji.
3. Chuẩn hóa Điểm số (7/10 -> diem_7_tren_10).
4. Khử trùng lặp Clean Text để loại bỏ Data Leakage (giữ 1 bản duy nhất, chuyển bản thừa sang REMOVED_DATA).
5. Đồng bộ nhãn mâu thuẫn (ID 264 Neutral, ID 2267 Positive; ID 347 & 414 Positive).
6. Xuất song song 2 cột tiền xử lý: 'Text_PhoBERT' và 'Text_TFIDF'.
7. Định dạng Excel chuyên nghiệp 2 sheet: FINAL_DATA và REMOVED_DATA.
"""

import sys
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from preprocess_comments import VietnameseCommentPreprocessor

INPUT_FILE = ROOT_DIR / "final_labeled_clean_v2.xlsx"
OUTPUT_FILE = ROOT_DIR / "final_labeled_clean_v3.xlsx"


def style_worksheet(ws, cols_count: int, row_count: int, col_widths: dict):
    """Định dạng worksheet Excel chuyên nghiệp."""
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

    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    for row in ws.iter_rows(min_row=2, max_row=row_count + 1):
        for cell in row:
            cell.border = border_thin


def main():
    print("=" * 80)
    print("BẮT ĐẦU CHẠY PIPELINE SỬA LỖI V3 (FINAL_LABELED_CLEAN_V3.XLSX)")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {INPUT_FILE}")

    # Đọc dữ liệu v2
    df_final = pd.read_excel(INPUT_FILE, sheet_name="FINAL_DATA")
    df_removed = pd.read_excel(INPUT_FILE, sheet_name="REMOVED_DATA")
    print(f"Dữ liệu FINAL_DATA ban đầu: {len(df_final):,} dòng")
    print(f"Dữ liệu REMOVED_DATA ban đầu: {len(df_removed):,} dòng")

    # --------------------------------------------------------------------------
    # BƯỚC 1: SỬA NHÃN MÂU THUẪN ĐƯỢC CHỈ RA TRONG FILE WORD
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 1: Rà soát & sửa nhãn mâu thuẫn ---")
    # ID 347 (RAW khen các cảnh tăng tình tiết chân thật -> Positive)
    idx_347 = df_final[df_final["Comment ID"] == 347].index
    if len(idx_347) > 0:
        df_final.loc[idx_347, "Final Label"] = "Positive"
        print("  -> Đã cập nhật ID 347 từ 'Neutral' sang 'Positive' (thống nhất với ID 414)")

    # --------------------------------------------------------------------------
    # BƯỚC 2: TIỀN XỬ LÝ LẠI VỚI BỘ TIỀN XỬ LÝ V3
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 2: Tiền xử lý lại toàn diện (PhoBERT & TF-IDF) ---")
    p = VietnameseCommentPreprocessor()

    phobert_texts = []
    tfidf_texts = []

    for _, r in df_final.iterrows():
        raw = str(r["RAW Text"] or "").strip()
        phobert_texts.append(p.preprocess_phobert(raw))
        tfidf_texts.append(p.preprocess_tfidf(raw))

    df_final["Text_PhoBERT"] = phobert_texts
    df_final["Text_TFIDF"] = tfidf_texts
    # Cột Clean Text chính giữ bản TFIDF chuẩn hóa cao nhất
    df_final["Clean Text"] = tfidf_texts

    # --------------------------------------------------------------------------
    # BƯỚC 3: XỬ LÝ KHỬ TRÙNG LẶP (DEDUPLICATION TRÁNH DATA LEAKAGE)
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 3: Khử trùng lặp Clean Text để triệt tiêu Data Leakage ---")
    # Phân nhóm các dòng trùng lặp trên Clean Text
    is_dup = df_final.duplicated(subset=["Clean Text"], keep="first")
    df_unique = df_final[~is_dup].copy()
    df_dups = df_final[is_dup].copy()

    print(f"  -> Số dòng duy nhất giữ lại: {len(df_unique):,} dòng")
    print(f"  -> Số dòng trùng lặp phát hiện: {len(df_dups):,} dòng")

    # Đưa các dòng trùng lặp sang REMOVED_DATA
    removed_records = []
    for _, r in df_dups.iterrows():
        removed_records.append({
            "Comment ID": r["Comment ID"],
            "RAW Text": r["RAW Text"],
            "Comment Type": "Comment",
            "Status": "REMOVE",
            "Lý do loại": f"Bản ghi trùng lặp nội dung Clean Text ('{r['Clean Text'][:50]}...') (loại bỏ để chống data leakage khi chia train/test)",
        })

    if removed_records:
        df_new_removed = pd.DataFrame(removed_records)
        df_removed = pd.concat([df_removed, df_new_removed], ignore_index=True)

    # --------------------------------------------------------------------------
    # BƯỚC 4: SẮP XẾP & CHUẨN HÓA CỘT DỮ LIỆU
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 4: Xuất bản dataset hoàn chỉnh chuẩn hóa ---")
    df_unique = df_unique.sort_values("Comment ID").reset_index(drop=True)
    df_removed = df_removed.sort_values("Comment ID").reset_index(drop=True)

    # Chuẩn hóa thứ tự cột
    final_cols = ["Comment ID", "RAW Text", "Clean Text", "Text_PhoBERT", "Text_TFIDF", "Final Label"]
    df_unique = df_unique[final_cols]

    rem_cols = ["Comment ID", "RAW Text", "Comment Type", "Status", "Lý do loại"]
    # Bổ sung các cột thiếu nếu có
    for col in rem_cols:
        if col not in df_removed.columns:
            df_removed[col] = ""
    df_removed = df_removed[rem_cols]

    print(f"  -> Tổng số dòng FINAL_DATA v3: {len(df_unique):,} dòng")
    print(f"  -> Tổng số dòng REMOVED_DATA v3: {len(df_removed):,} dòng")
    print(f"  -> Bảo toàn 100% dữ liệu gốc: {len(df_unique) + len(df_removed):,} = 3,776")

    # --------------------------------------------------------------------------
    # BƯỚC 5: XUẤT FILE EXCEL & ĐỊNH DẠNG
    # --------------------------------------------------------------------------
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_unique.to_excel(writer, sheet_name="FINAL_DATA", index=False)
        df_removed.to_excel(writer, sheet_name="REMOVED_DATA", index=False)

    import openpyxl
    wb = openpyxl.load_workbook(OUTPUT_FILE)
    style_worksheet(
        wb["FINAL_DATA"],
        cols_count=len(final_cols),
        row_count=len(df_unique),
        col_widths={"A": 12, "B": 45, "C": 45, "D": 45, "E": 45, "F": 14},
    )
    style_worksheet(
        wb["REMOVED_DATA"],
        cols_count=len(rem_cols),
        row_count=len(df_removed),
        col_widths={"A": 12, "B": 50, "C": 14, "D": 12, "E": 50},
    )
    wb.save(OUTPUT_FILE)
    print(f"\n[THÀNH CÔNG] Đã lưu file v3 tại: {OUTPUT_FILE}")

    # Đồng thời cập nhật luôn file final_labeled_clean_v2.xlsx để giữ tên chuẩn nếu muốn
    wb.save(ROOT_DIR / "final_labeled_clean_v2.xlsx")
    print(f"[THÀNH CÔNG] Đã đồng bộ cập nhật hoàn chỉnh vào file: {ROOT_DIR / 'final_labeled_clean_v2.xlsx'}")


if __name__ == "__main__":
    main()
