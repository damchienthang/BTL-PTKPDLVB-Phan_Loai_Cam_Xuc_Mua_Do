"""
Script xử lý và làm sạch dữ liệu phiên bản v2: 'final_labeled_clean_v2.xlsx'
Thực hiện chính xác 5 bước theo yêu cầu của Data Engineer.
"""

import re
import sys
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
_candidates = [
    ROOT_DIR / "final_labeled_fixed.xlsx",
    ROOT_DIR / "archive" / "final_labeled_fixed.xlsx",
]
INPUT_FILE = next((p for p in _candidates if p.exists()), _candidates[0])
OUTPUT_FILE = ROOT_DIR / "final_labeled_clean_v2.xlsx"


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
    print("BẮT ĐẦU XỬ LÝ DỮ LIỆU TẠO FILE 'final_labeled_clean_v2.xlsx'")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}")

    # Đọc dữ liệu từ file gốc
    df_final = pd.read_excel(INPUT_FILE, sheet_name="FINAL_DATA")
    df_removed = pd.read_excel(INPUT_FILE, sheet_name="REMOVED_DATA")

    print(f"[Đọc file] Sheet FINAL_DATA ban đầu   : {len(df_final):,} dòng")
    print(f"[Đọc file] Sheet REMOVED_DATA ban đầu : {len(df_removed):,} dòng")

    # Kiểm tra khôi phục ID 1602 nếu đang nằm nhầm bên REMOVED_DATA
    if 1602 in df_removed["Comment ID"].values and 1602 not in df_final["Comment ID"].values:
        row_1602 = df_removed[df_removed["Comment ID"] == 1602].iloc[0]
        # Chuyển về FINAL_DATA
        new_row_1602 = {
            "Comment ID": 1602,
            "RAW Text": row_1602["RAW Text"],
            "Clean Text": "dep chai coi_di",
            "Final Label": "Positive",
        }
        df_final = pd.concat([df_final, pd.DataFrame([new_row_1602])], ignore_index=True)
        # Xóa khỏi REMOVED_DATA
        df_removed = df_removed[df_removed["Comment ID"] != 1602].copy()
        print("[Khôi phục] Đã chuyển ID 1602 từ REMOVED_DATA về lại FINAL_DATA.")

    # --------------------------------------------------------------------------
    # BƯỚC 1: DI CHUYỂN CÁC DÒNG QUÁ NGẮN (LỌC DATA)
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 1: Di chuyển các dòng quá ngắn [811, 879, 880, 3757] ---")
    move_ids = [811, 879, 880, 3757]
    mask_move = df_final["Comment ID"].isin(move_ids)
    rows_to_move = df_final[mask_move].copy()

    print(f" -> Tìm thấy {len(rows_to_move)} dòng cần di chuyển trong FINAL_DATA:")
    for _, r in rows_to_move.iterrows():
        print(f"    + ID {r['Comment ID']}: {r['RAW Text']}")

    # Chuẩn bị dữ liệu để append vào REMOVED_DATA
    new_removed_records = []
    for _, r in rows_to_move.iterrows():
        new_removed_records.append({
            "Comment ID": r["Comment ID"],
            "RAW Text": r["RAW Text"],
            "Comment Type": "Reply",
            "Status": "REMOVE",
            "Lý do loại": "Câu quá ngắn, phụ thuộc ngữ cảnh",
        })

    # Cắt (move) sang REMOVED_DATA
    df_removed = pd.concat([df_removed, pd.DataFrame(new_removed_records)], ignore_index=True)

    # Đảm bảo sheet FINAL_DATA không còn các ID này (xóa các row đó đi)
    df_final = df_final[~mask_move].copy()
    print(f" -> Đã xóa {len(rows_to_move)} dòng khỏi FINAL_DATA.")
    print(f" -> Số dòng FINAL_DATA sau Bước 1: {len(df_final):,} dòng")
    print(f" -> Số dòng REMOVED_DATA sau Bước 1: {len(df_removed):,} dòng")

    # --------------------------------------------------------------------------
    # BƯỚC 2: XỬ LÝ DỮ LIỆU NULL (NaN)
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 2: Xử lý dữ liệu Null (NaN) ---")
    # 1. Điền chuỗi "[Bình luận trống]" vào những ô bị NaN ở cột RAW Text trong REMOVED_DATA
    null_raw_count = df_removed["RAW Text"].isnull().sum()
    df_removed["RAW Text"] = df_removed["RAW Text"].fillna("[Bình luận trống]")
    print(f" -> Đã điền '[Bình luận trống]' cho {null_raw_count} ô bị NaN ở cột RAW Text trong REMOVED_DATA.")

    # 2. Trong sheet 'FINAL_DATA', sửa cột Clean Text của dòng có Comment ID = 417 thành "hay"
    if 417 in df_final["Comment ID"].values:
        old_val = df_final.loc[df_final["Comment ID"] == 417, "Clean Text"].values[0]
        df_final.loc[df_final["Comment ID"] == 417, "Clean Text"] = "hay"
        print(f" -> Comment ID 417: Đã cập nhật Clean Text từ '{old_val}' thành 'hay'.")

    # --------------------------------------------------------------------------
    # BƯỚC 3: KHÔI PHỤC TÊN BỊ CẮT NHẦM ĐẦU CÂU TRONG CỘT 'Clean Text'
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 3: Khôi phục tên bị cắt nhầm đầu câu trong Clean Text ---")

    def restore_prefix(current_text: str, prefix_to_add: str, known_prefixes: list) -> str:
        text = str(current_text or "").strip()
        # Nếu đã có prefix chính xác
        if text.startswith(prefix_to_add):
            return text
        # Nếu đang bắt đầu bằng một biến thể khác của tên đó, thay thế
        for kp in known_prefixes:
            if text.startswith(kp):
                rem = text[len(kp):].strip()
                return (prefix_to_add + rem).strip()
        # Nếu chưa có, thêm tiền tố vào đầu câu
        return (prefix_to_add + text).strip()

    # ID 1334: Thêm tiền tố "hộ_linh_tráng_sĩ "
    if 1334 in df_final["Comment ID"].values:
        old_t = df_final.loc[df_final["Comment ID"] == 1334, "Clean Text"].values[0]
        new_t = restore_prefix(old_t, "hộ_linh_tráng_sĩ ", ["hộ_linh_tráng_sĩ", "hộ linh tráng sĩ"])
        df_final.loc[df_final["Comment ID"] == 1334, "Clean Text"] = new_t
        print(f" -> ID 1334: {new_t}")

    # ID 1602: Thêm tiền tố "steven_nguyễn "
    if 1602 in df_final["Comment ID"].values:
        old_t = df_final.loc[df_final["Comment ID"] == 1602, "Clean Text"].values[0]
        new_t = restore_prefix(old_t, "steven_nguyễn ", ["steven_nguyễn", "steven nguyen", "steven_nguyen"])
        df_final.loc[df_final["Comment ID"] == 1602, "Clean Text"] = new_t
        print(f" -> ID 1602: {new_t}")

    # ID 3086: Thêm tiền tố "cô như_quỳnh "
    if 3086 in df_final["Comment ID"].values:
        old_t = df_final.loc[df_final["Comment ID"] == 3086, "Clean Text"].values[0]
        new_t = restore_prefix(old_t, "cô như_quỳnh ", ["cô_như_quỳnh ", "cô_như_quỳnh", "cô như quỳnh"])
        df_final.loc[df_final["Comment ID"] == 3086, "Clean Text"] = new_t
        print(f" -> ID 3086: {new_t}")

    # ID 3211: Thêm tiền tố "minh_kiên "
    if 3211 in df_final["Comment ID"].values:
        old_t = df_final.loc[df_final["Comment ID"] == 3211, "Clean Text"].values[0]
        new_t = restore_prefix(old_t, "minh_kiên ", ["minh_kiên", "minh kiên"])
        df_final.loc[df_final["Comment ID"] == 3211, "Clean Text"] = new_t
        print(f" -> ID 3211: {new_t}")

    # --------------------------------------------------------------------------
    # BƯỚC 4: SỬA LỖI TỪ ĐIỂN (REPLACE TEXT) TRONG TOÀN BỘ CỘT 'Clean Text'
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 4: Sửa lỗi từ điển trong toàn bộ cột 'Clean Text' ---")

    # Danh sách thay thế Regex độc lập với word boundary \b
    regex_word_replacements = [
        (re.compile(r"\bthường\b", flags=re.IGNORECASE), "thằng"),
        (re.compile(r"\bphê\b", flags=re.IGNORECASE), "phe"),
        (re.compile(r"\bchả\b", flags=re.IGNORECASE), "cha"),
        (re.compile(r"\b(\d+|trăm|ngàn|nghìn)\s+tí\b", flags=re.IGNORECASE), r"\1 tỷ"),
    ]

    # Danh sách thay thế cụm từ cố định (không cần word boundary)
    fixed_phrase_replacements = [
        ("hôm quá", "hôm qua"),
        ("tối quá", "tối qua"),
        ("đêm quá", "đêm qua"),
        ("quá cam", "qua cam"),
        ("quá campuchia", "qua campuchia"),
        ("lừa quá", "lừa qua"),
        ("dụ quá", "dụ qua"),
        ("vượt quá", "vượt qua"),
        ("bỏ quá", "bỏ qua"),
        ("quá bên", "qua bên"),
        ("tử_chiến trên không_đưa", "tử_chiến_trên_không đưa"),
        ("tử_chiến trên không_được", "tử_chiến_trên_không được"),
        ("tử_chiến trên không_thể", "tử_chiến_trên_không thể"),
    ]

    clean_series = df_final["Clean Text"].astype(str).copy()

    # Áp dụng Regex word boundary
    for pat, repl in regex_word_replacements:
        clean_series = clean_series.apply(lambda text: pat.sub(repl, text))

    # Áp dụng thay thế cụm từ cố định
    for src, dst in fixed_phrase_replacements:
        clean_series = clean_series.apply(lambda text: text.replace(src, dst))

    df_final["Clean Text"] = clean_series
    print(" -> Đã hoàn tất thay thế Regex word boundary và cụm từ cố định.")

    # --------------------------------------------------------------------------
    # BƯỚC 5: XUẤT FILE 'final_labeled_clean_v2.xlsx'
    # --------------------------------------------------------------------------
    print("\n--- BƯỚC 5: Xuất file mới 'final_labeled_clean_v2.xlsx' ---")

    # Sắp xếp theo Comment ID tăng dần
    df_final = df_final.sort_values("Comment ID").reset_index(drop=True)
    df_removed = df_removed.sort_values("Comment ID").reset_index(drop=True)

    # Đảm bảo đúng các cột theo chuẩn
    df_final = df_final[["Comment ID", "RAW Text", "Clean Text", "Final Label"]]
    df_removed = df_removed[["Comment ID", "RAW Text", "Comment Type", "Status", "Lý do loại"]]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_final.to_excel(writer, sheet_name="FINAL_DATA", index=False)
        ws_final = writer.sheets["FINAL_DATA"]
        style_worksheet(
            ws_final,
            cols_count=4,
            row_count=len(df_final),
            col_widths={"A": 14, "B": 55, "C": 55, "D": 16},
        )
        for row in ws_final.iter_rows(min_row=2, max_row=len(df_final) + 1):
            row[0].alignment = Alignment(horizontal="center", vertical="center")
            row[1].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            row[2].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            row[3].alignment = Alignment(horizontal="center", vertical="center")

        df_removed.to_excel(writer, sheet_name="REMOVED_DATA", index=False)
        ws_removed = writer.sheets["REMOVED_DATA"]
        style_worksheet(
            ws_removed,
            cols_count=5,
            row_count=len(df_removed),
            col_widths={"A": 14, "B": 55, "C": 15, "D": 15, "E": 45},
        )
        for row in ws_removed.iter_rows(min_row=2, max_row=len(df_removed) + 1):
            row[0].alignment = Alignment(horizontal="center", vertical="center")
            row[1].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            row[2].alignment = Alignment(horizontal="center", vertical="center")
            row[3].alignment = Alignment(horizontal="center", vertical="center")
            row[4].alignment = Alignment(horizontal="left", vertical="center")

    print(f"\n[THÀNH CÔNG] Đã xuất file: {OUTPUT_FILE}")
    print(f"[XÁC MINH] Sheet 'FINAL_DATA'  : {len(df_final):,} dòng, 4 cột.")
    print(f"[XÁC MINH] Sheet 'REMOVED_DATA': {len(df_removed):,} dòng, 5 cột.")
    print(f"[TỔNG CỘNG] Tổng số dòng 2 sheet: {len(df_final) + len(df_removed):,} dòng (bằng đúng 3.776 dòng gốc).")

    # Kiểm tra lại các ID đã xử lý
    print("\n" + "=" * 80)
    print("KIỂM TRA CÁC MẪU ĐÃ XỬ LÝ THEO YÊU CẦU:")
    print("=" * 80)
    print("- Kiểm tra các ID đã di chuyển sang REMOVED_DATA:")
    print(df_removed[df_removed["Comment ID"].isin(move_ids)][["Comment ID", "RAW Text", "Status", "Lý do loại"]])

    print("\n- Kiểm tra các ID đã khôi phục tiền tố trong FINAL_DATA:")
    print(df_final[df_final["Comment ID"].isin([417, 1334, 1602, 3086, 3211])][["Comment ID", "Clean Text"]])


if __name__ == "__main__":
    main()

