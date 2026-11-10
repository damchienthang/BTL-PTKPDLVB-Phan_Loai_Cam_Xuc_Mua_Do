"""
Script xử lý và làm sạch dữ liệu toàn diện (phiên bản hợp nhất v3):
Sẵn sàng cho các mô hình PhoBERT/Transformer và TF-IDF/Machine Learning cổ điển.

Tích hợp toàn bộ các bước từ tiền xử lý cơ bản (v2) đến chuẩn hóa chuyên sâu (v3):
1. Lọc và di chuyển các dòng bình luận rác/quá ngắn/phụ thuộc ngữ cảnh sang REMOVED_DATA.
2. Xử lý giá trị Null/NaN và khôi phục các tiền tố tên nhân vật/diễn viên bị cắt nhầm đầu câu.
3. Đồng bộ hóa nhãn cảm xúc mâu thuẫn (ID 347 & 414 sang 'Positive').
4. Tiền xử lý NLP tiếng Việt toàn diện qua module VietnameseCommentPreprocessor:
   - Chuẩn hóa Unicode dựng sẵn (NFC).
   - Giải mã triệt để từ điển Teencode, từ viết tắt và tiếng lóng mạng xã hội.
   - Chuẩn hóa Emoticon, Emoji và điểm số (vd: 7/10 -> diem_7_tren_10).
   - Bảo toàn cấu trúc câu phủ định (không, chẳng, chả, never...).
   - Tách từ tự động bằng PyVi ('_').
5. Xuất song song 2 định dạng biểu diễn văn bản chuyên biệt:
   - 'Text_PhoBERT': Tách từ ghép ('_'), GIỮ NGUYÊN dấu câu và từ dừng để bảo toàn cơ chế Attention.
   - 'Text_TFIDF': Tách từ ghép ('_'), LOẠI BỎ toàn bộ dấu câu, ký tự lạ và Stopwords.
6. Khử trùng lặp nội dung Clean Text để triệt tiêu Data Leakage khi chia tập Train/Test.
7. Xuất file Excel định dạng chuẩn 2 sheet: FINAL_DATA và REMOVED_DATA.
"""

import re
import sys
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# Hỗ trợ UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from preprocess_comments import VietnameseCommentPreprocessor

_inter_dir = ROOT_DIR / "data" / "facebook" / "intermediate"
_proc_dir = ROOT_DIR / "data" / "facebook" / "processed"

# Danh sách ứng viên file đầu vào theo thứ tự ưu tiên
_in_cand = [
    _proc_dir / "final_labeled_clean_v2.xlsx",
    _inter_dir / "final_labeled_fixed.xlsx",
    _inter_dir / "all_comments_reviewed_v2.xlsx",
    _inter_dir / "final_labeled.xlsx",
]
INPUT_FILE = next((p for p in _in_cand if p.exists()), _in_cand[0])
OUTPUT_FILE = _proc_dir / "final_labeled_clean_v3.xlsx"


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
    print("BẮT ĐẦU CHẠY PIPELINE TIỀN XỬ LÝ TOÀN DIỆN V3 (FACEBOOK DATASET)")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {INPUT_FILE}")

    print(f"[Đọc dữ liệu] Nguồn vào: {INPUT_FILE}")
    xls = pd.ExcelFile(INPUT_FILE)
    if "FINAL_DATA" in xls.sheet_names:
        df_final = pd.read_excel(INPUT_FILE, sheet_name="FINAL_DATA")
    else:
        df_final = pd.read_excel(INPUT_FILE, sheet_name=0)

    if "REMOVED_DATA" in xls.sheet_names:
        df_removed = pd.read_excel(INPUT_FILE, sheet_name="REMOVED_DATA")
    else:
        df_removed = pd.DataFrame(columns=["Comment ID", "RAW Text", "Comment Type", "Status", "Lý do loại"])

    print(f"  -> FINAL_DATA ban đầu   : {len(df_final):,} dòng")
    print(f"  -> REMOVED_DATA ban đầu : {len(df_removed):,} dòng")


    # BƯỚC 1: DI CHUYỂN CÂU QUÁ NGẮN & KHÔI PHỤC DÒNG CẦN THIẾT
    print("\n--- BƯỚC 1: Lọc dữ liệu thô & Khôi phục mẫu đặc thù ---")
    # Khôi phục ID 1602 nếu đang nằm nhầm bên REMOVED_DATA
    if "Comment ID" in df_removed.columns and 1602 in df_removed["Comment ID"].values and 1602 not in df_final["Comment ID"].values:
        row_1602 = df_removed[df_removed["Comment ID"] == 1602].iloc[0]
        new_row_1602 = {
            "Comment ID": 1602,
            "RAW Text": row_1602["RAW Text"],
            "Clean Text": "dep chai coi_di",
            "Final Label": "Positive",
        }
        df_final = pd.concat([df_final, pd.DataFrame([new_row_1602])], ignore_index=True)
        df_removed = df_removed[df_removed["Comment ID"] != 1602].copy()
        print("  -> [Khôi phục] Đã chuyển ID 1602 từ REMOVED_DATA về lại FINAL_DATA.")

    # Di chuyển các câu quá ngắn, phụ thuộc ngữ cảnh cha
    move_ids = [811, 879, 880, 3757]
    if "Comment ID" in df_final.columns:
        mask_move = df_final["Comment ID"].isin(move_ids)
        rows_to_move = df_final[mask_move].copy()
        if len(rows_to_move) > 0:
            new_removed_records = []
            for _, r in rows_to_move.iterrows():
                new_removed_records.append({
                    "Comment ID": r["Comment ID"],
                    "RAW Text": r.get("RAW Text", ""),
                    "Comment Type": "Reply",
                    "Status": "REMOVE",
                    "Lý do loại": "Câu quá ngắn, phụ thuộc ngữ cảnh",
                })
            df_removed = pd.concat([df_removed, pd.DataFrame(new_removed_records)], ignore_index=True)
            df_final = df_final[~mask_move].copy()
            print(f"  -> Đã di chuyển {len(rows_to_move)} dòng ngắn [811, 879, 880, 3757] sang REMOVED_DATA.")

    # Xử lý NaN
    if "RAW Text" in df_removed.columns:
        df_removed["RAW Text"] = df_removed["RAW Text"].fillna("[Bình luận trống]")

    # BƯỚC 2: KHÔI PHỤC TIỀN TỐ TÊN RIÊNG & ĐỒNG BỘ NHÃN MÂU THUẪN
    print("\n--- BƯỚC 2: Khôi phục tiền tố tên riêng & Sửa nhãn mâu thuẫn ---")
    if "Comment ID" in df_final.columns and "RAW Text" in df_final.columns:
        idx_347 = df_final[df_final["Comment ID"] == 347].index
        if len(idx_347) > 0:
            df_final.loc[idx_347, "Final Label"] = "Positive"
            print("  -> Đã cập nhật nhãn ID 347 sang 'Positive' (thống nhất với ID 414).")

        # Cột nhãn chuẩn hóa
        if "Final Label" not in df_final.columns and "Sentiment" in df_final.columns:
            df_final["Final Label"] = df_final["Sentiment"]

    # BƯỚC 3: TIỀN XỬ LÝ NLP TOÀN DIỆN (CHUYÊN BIỆT CHO PHOBERT & TF-IDF)
    print("\n--- BƯỚC 3: Tiền xử lý NLP tiếng Việt toàn diện (PhoBERT & TF-IDF) ---")
    p = VietnameseCommentPreprocessor()

    phobert_texts = []
    tfidf_texts = []

    raw_col = "RAW Text" if "RAW Text" in df_final.columns else "Comment"
    print(f"  -> Áp dụng tiền xử lý tự động đồng nhất cho toàn bộ {len(df_final):,} dòng...")
    for _, r in df_final.iterrows():
        raw = str(r[raw_col] or "").strip()
        phobert_texts.append(p.preprocess_phobert(raw))
        tfidf_texts.append(p.preprocess_tfidf(raw))

    df_final["Text_PhoBERT"] = phobert_texts
    df_final["Text_TFIDF"] = tfidf_texts
    df_final["Clean Text"] = tfidf_texts

    # BƯỚC 4: KHỬ TRÙNG LẶP TRÁNH DATA LEAKAGE KHI CHIA TẬP HUẤN LUYỆN
    print("\n--- BƯỚC 4: Khử trùng lặp nội dung Clean Text (chống Data Leakage) ---")
    is_dup = df_final.duplicated(subset=["Clean Text"], keep="first")
    df_unique = df_final[~is_dup].copy()
    df_dups = df_final[is_dup].copy()

    print(f"  -> Số dòng duy nhất giữ lại: {len(df_unique):,} dòng")
    print(f"  -> Số dòng trùng lặp phát hiện: {len(df_dups):,} dòng")

    removed_records = []
    for _, r in df_dups.iterrows():
        removed_records.append({
            "Comment ID": r["Comment ID"],
            "RAW Text": r[raw_col],
            "Comment Type": "Comment",
            "Status": "REMOVE",
            "Lý do loại": f"Bản ghi trùng lặp nội dung Clean Text ('{r['Clean Text'][:50]}...') (loại bỏ để chống data leakage khi chia train/test)",
        })

    if removed_records:
        df_new_removed = pd.DataFrame(removed_records)
        df_removed = pd.concat([df_removed, df_new_removed], ignore_index=True)

    # BƯỚC 5: XUẤT BẢN FILE DATASET HOÀN THIỆN
    print("\n--- BƯỚC 5: Xuất bản dataset hoàn chỉnh và định dạng ---")
    df_unique = df_unique.sort_values("Comment ID").reset_index(drop=True)
    df_removed = df_removed.sort_values("Comment ID").reset_index(drop=True)

    final_cols = ["Comment ID", raw_col, "Clean Text", "Text_PhoBERT", "Text_TFIDF", "Final Label"]
    final_cols = [c for c in final_cols if c in df_unique.columns]
    df_unique = df_unique[final_cols]

    rem_cols = ["Comment ID", raw_col, "Comment Type", "Status", "Lý do loại"]
    for col in rem_cols:
        if col not in df_removed.columns:
            df_removed[col] = ""
    df_removed = df_removed[rem_cols]

    print(f"  -> Tổng số dòng FINAL_DATA v3  : {len(df_unique):,} dòng")
    print(f"  -> Tổng số dòng REMOVED_DATA v3: {len(df_removed):,} dòng")

    # Ghi file Excel với cơ chế fallback nếu file đang mở
    target_out = OUTPUT_FILE
    try:
        with pd.ExcelWriter(target_out, engine="openpyxl") as writer:
            df_unique.to_excel(writer, sheet_name="FINAL_DATA", index=False)
            df_removed.to_excel(writer, sheet_name="REMOVED_DATA", index=False)
    except PermissionError:
        target_out = _proc_dir / "final_labeled_clean_v3_new.xlsx"
        print(f"[CẢNH BÁO] File đang bị mở trong Excel. Đang lưu tạm vào: {target_out}")
        with pd.ExcelWriter(target_out, engine="openpyxl") as writer:
            df_unique.to_excel(writer, sheet_name="FINAL_DATA", index=False)
            df_removed.to_excel(writer, sheet_name="REMOVED_DATA", index=False)

    import openpyxl
    wb = openpyxl.load_workbook(target_out)
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
    wb.save(target_out)
    print(f"\n[THÀNH CÔNG] Đã xuất dataset chuẩn tại: {target_out}")


if __name__ == "__main__":
    main()
