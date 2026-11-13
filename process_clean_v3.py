"""
Script xử lý và làm sạch dữ liệu toàn diện (Clean Architecture v3):
Sẵn sàng cho các mô hình PhoBERT/Transformer và TF-IDF/Machine Learning cổ điển.

Các bước thực thi:
1. Đọc dữ liệu linh hoạt (hỗ trợ cả Facebook và TikTok, tự nhận diện sheet & cột).
2. Tách dữ liệu: đưa các câu KEEP sang FINAL_DATA, các câu REMOVE/REVIEW sang REMOVED_DATA.
3. Tiền xử lý NLP tiếng Việt toàn diện qua module VietnameseCommentPreprocessor:
   - Chuẩn hóa Unicode dựng sẵn (NFC).
   - Giải mã triệt để từ điển Teencode, từ viết tắt và tiếng lóng.
   - Loại bỏ hoàn toàn Emoji và Emoticon.
   - Chuẩn hóa điểm số (vd: 7/10 -> diem_7_tren_10).
   - Tách từ tự động bằng PyVi (nối '_').
   - Xuất song song 2 định dạng: 'Text_PhoBERT' và 'Text_TFIDF'.
4. Khử trùng lặp nội dung Clean Text để triệt tiêu Data Leakage khi chia tập Train/Test.
5. Xuất file Excel định dạng chuẩn 2 sheet: FINAL_DATA và REMOVED_DATA kèm styling chuyên nghiệp.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
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



TIKTOK_INTERMEDIATE_DIR = ROOT_DIR / "data" / "tiktok" / "intermediate"
TIKTOK_PROCESSED_DIR = ROOT_DIR / "data" / "tiktok" / "processed"
FACEBOOK_INTERMEDIATE_DIR = ROOT_DIR / "data" / "facebook" / "intermediate"
FACEBOOK_PROCESSED_DIR = ROOT_DIR / "data" / "facebook" / "processed"

# Cấu hình nguồn dữ liệu riêng biệt cho từng nền tảng
PLATFORM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tiktok": {
        "name": "TIKTOK",
        "candidates": (
            TIKTOK_INTERMEDIATE_DIR / "tiktok_comments_reviewed_v2.xlsx",
            TIKTOK_PROCESSED_DIR / "final_labeled_clean_v2.xlsx",
            TIKTOK_INTERMEDIATE_DIR / "final_labeled_fixed.xlsx",
            TIKTOK_INTERMEDIATE_DIR / "Danh_sach_kiem_tra_nhan.xlsx",
            ROOT_DIR / "data" / "tiktok" / "tiktok_comments_annotated.xlsx",
        ),
        "output_file": TIKTOK_PROCESSED_DIR / "final_labeled_clean_v3.xlsx",
    },
    "facebook": {
        "name": "FACEBOOK",
        "candidates": (
            FACEBOOK_PROCESSED_DIR / "final_labeled_clean_v2.xlsx",
            FACEBOOK_INTERMEDIATE_DIR / "final_labeled.xlsx",
            FACEBOOK_INTERMEDIATE_DIR / "all_comments_reviewed_v2.xlsx",
            FACEBOOK_INTERMEDIATE_DIR / "final_labeled_fixed.xlsx",
        ),
        "output_file": FACEBOOK_PROCESSED_DIR / "final_labeled_clean_v3.xlsx",
    },
}

# Danh sách tên cột ứng viên cho từng thuộc tính
RAW_TEXT_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "RAW Text", "Nội dung RAW", "Nội dung", "Comment", "Text RAW", "comment_text",
)

ID_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "Comment ID", "Row ID", "ID", "comment_id",
)

LABEL_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "Final Label", "Final Sentiment", "Sentiment", "Nhãn", "sentiment", "Label",
)

FINAL_DATA_COLUMNS: Tuple[str, ...] = (
    "Comment ID", "RAW Text", "Clean Text", "Text_PhoBERT", "Text_TFIDF", "Final Label",
)

REMOVED_DATA_COLUMNS: Tuple[str, ...] = (
    "Comment ID", "RAW Text", "Comment Type", "Status", "Lý do loại",
)

# Cấu hình giao diện Excel
EXCEL_HEADER_BG_COLOR = "1F4E79"
EXCEL_HEADER_FONT_COLOR = "FFFFFF"
EXCEL_BORDER_COLOR = "D9D9D9"

FINAL_COL_WIDTHS: Dict[str, int] = {
    "A": 12, "B": 45, "C": 45, "D": 45, "E": 45, "F": 14,
}
REMOVED_COL_WIDTHS: Dict[str, int] = {
    "A": 12, "B": 50, "C": 14, "D": 12, "E": 50,
}



def find_matched_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    """Tìm cột đầu tiên khớp với danh sách ứng viên (Guard clause - Early return)."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def extract_raw_datasets(input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Đọc và phân tách dữ liệu ban đầu thành 2 tập: FINAL_DATA (KEEP) và REMOVED_DATA.
    Flattened logic loại bỏ if/else lồng nhau.
    """
    xls = pd.ExcelFile(input_path)

    if "FINAL_DATA" in xls.sheet_names:
        df_final = pd.read_excel(input_path, sheet_name="FINAL_DATA")
    elif "ALL_REVIEWED" in xls.sheet_names:
        df_final = pd.read_excel(input_path, sheet_name="ALL_REVIEWED")
    else:
        df_final = pd.read_excel(input_path, sheet_name=0)

    # Nếu nguồn vào đã có sheet REMOVED_DATA riêng
    if "REMOVED_DATA" in xls.sheet_names:
        df_removed = pd.read_excel(input_path, sheet_name="REMOVED_DATA")
        return df_final, df_removed

    # Nếu nguồn vào có cột trạng thái (Final Status hoặc filter_status)
    status_col = find_matched_column(df_final, ("Final Status", "filter_status", "Status"))
    if status_col:
        mask_keep = df_final[status_col].astype(str).str.upper() == "KEEP"
        df_removed = df_final[~mask_keep].copy()
        df_final = df_final[mask_keep].copy()
        return df_final, df_removed

    # Trường hợp mặc định
    df_removed = pd.DataFrame(columns=list(REMOVED_DATA_COLUMNS))
    return df_final, df_removed


def standardize_dataframe_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Chuẩn hóa tên cột ID, RAW Text và Nhãn nhãn theo schema chuẩn."""
    # 1. Chuẩn hóa ID
    id_col = find_matched_column(df, ID_COLUMN_CANDIDATES)
    if not id_col:
        df["Comment ID"] = range(1, len(df) + 1)
    elif id_col != "Comment ID":
        df["Comment ID"] = df[id_col]

    # 2. Chuẩn hóa nhãn cảm xúc
    label_col = find_matched_column(df, LABEL_COLUMN_CANDIDATES)
    if label_col and label_col != "Final Label":
        df["Final Label"] = df[label_col]

    # 3. Chuẩn hóa nội dung thô
    raw_col = find_matched_column(df, RAW_TEXT_COLUMN_CANDIDATES)
    if not raw_col:
        raise KeyError(f"Không tìm thấy cột nội dung bình luận trong: {df.columns.tolist()}")

    df["RAW Text"] = df[raw_col].fillna("").astype(str)
    return df, "RAW Text"


def style_worksheet(ws, cols_count: int, row_count: int, col_widths: Dict[str, int]) -> None:
    """Áp dụng định dạng chuyên nghiệp cho worksheet Excel."""
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    header_fill = PatternFill(
        start_color=EXCEL_HEADER_BG_COLOR, end_color=EXCEL_HEADER_BG_COLOR, fill_type="solid"
    )
    header_font = Font(name="Calibri", size=11, bold=True, color=EXCEL_HEADER_FONT_COLOR)
    border_thin = Border(
        left=Side(style="thin", color=EXCEL_BORDER_COLOR),
        right=Side(style="thin", color=EXCEL_BORDER_COLOR),
        top=Side(style="thin", color=EXCEL_BORDER_COLOR),
        bottom=Side(style="thin", color=EXCEL_BORDER_COLOR),
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


def run_cleaning_pipeline(
    input_file: Path,
    output_file: Path,
) -> None:
    """Thực thi pipeline làm sạch và tiền xử lý NLP toàn diện."""
    print("=" * 80)
    print("BẮT ĐẦU CHẠY PIPELINE TIỀN XỬ LÝ TOÀN DIỆN V3 (CLEAN ARCHITECTURE)")
    print("=" * 80)

    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {input_file}")

    print(f"[Đọc dữ liệu] Nguồn vào: {input_file}")
    df_final, df_removed = extract_raw_datasets(input_file)
    print(f"  -> FINAL_DATA ban đầu   : {len(df_final):,} dòng")
    print(f"  -> REMOVED_DATA ban đầu : {len(df_removed):,} dòng")

    # 1. Chuẩn hóa Schema
    df_final, raw_col = standardize_dataframe_schema(df_final)

    # 2. Tiền xử lý NLP
    print("\n--- Tiền xử lý NLP tiếng Việt toàn diện (PhoBERT & TF-IDF) ---")
    preprocessor = VietnameseCommentPreprocessor()

    raw_texts = df_final[raw_col].tolist()
    phobert_texts: List[str] = [preprocessor.preprocess_phobert(t) for t in raw_texts]
    tfidf_texts: List[str] = [preprocessor.preprocess_tfidf(t) for t in raw_texts]

    df_final["Text_PhoBERT"] = phobert_texts
    df_final["Text_TFIDF"] = tfidf_texts
    df_final["Clean Text"] = tfidf_texts

    # 3. Khử trùng lặp tránh Data Leakage
    print("\n--- Khử trùng lặp nội dung Clean Text (chống Data Leakage) ---")
    is_duplicate = df_final.duplicated(subset=["Clean Text"], keep="first")
    df_unique = df_final[~is_duplicate].copy()
    df_duplicates = df_final[is_duplicate].copy()

    print(f"  -> Số dòng duy nhất giữ lại: {len(df_unique):,} dòng")
    print(f"  -> Số dòng trùng lặp phát hiện: {len(df_duplicates):,} dòng")

    if not df_duplicates.empty:
        leakage_records = [
            {
                "Comment ID": r["Comment ID"],
                "RAW Text": r[raw_col],
                "Comment Type": "Comment",
                "Status": "REMOVE",
                "Lý do loại": f"Trùng lặp Clean Text ('{r['Clean Text'][:50]}...') (loại bỏ chống data leakage)",
            }
            for _, r in df_duplicates.iterrows()
        ]
        df_removed = pd.concat([df_removed, pd.DataFrame(leakage_records)], ignore_index=True)

    # 4. Sắp xếp và định dạng cột cuối cùng
    df_unique = df_unique.sort_values("Comment ID").reset_index(drop=True)
    df_removed = df_removed.sort_values("Comment ID").reset_index(drop=True)

    active_final_cols = [c for c in FINAL_DATA_COLUMNS if c in df_unique.columns]
    df_unique = df_unique[active_final_cols]

    for col in REMOVED_DATA_COLUMNS:
        if col not in df_removed.columns:
            df_removed[col] = ""
    df_removed = df_removed[list(REMOVED_DATA_COLUMNS)]

    print(f"  -> Tổng số dòng FINAL_DATA v3  : {len(df_unique):,} dòng")
    print(f"  -> Tổng số dòng REMOVED_DATA v3: {len(df_removed):,} dòng")

    # 5. Xuất file Excel kèm cơ chế Fallback
    target_output = output_file
    try:
        with pd.ExcelWriter(target_output, engine="openpyxl") as writer:
            df_unique.to_excel(writer, sheet_name="FINAL_DATA", index=False)
            df_removed.to_excel(writer, sheet_name="REMOVED_DATA", index=False)
    except PermissionError:
        target_output = output_file.parent / f"{output_file.stem}_new.xlsx"
        print(f"[CẢNH BÁO] File bị khóa do mở trong Excel. Lưu tạm vào: {target_output}")
        with pd.ExcelWriter(target_output, engine="openpyxl") as writer:
            df_unique.to_excel(writer, sheet_name="FINAL_DATA", index=False)
            df_removed.to_excel(writer, sheet_name="REMOVED_DATA", index=False)

    wb = openpyxl.load_workbook(target_output)
    style_worksheet(wb["FINAL_DATA"], len(active_final_cols), len(df_unique), FINAL_COL_WIDTHS)
    style_worksheet(wb["REMOVED_DATA"], len(REMOVED_DATA_COLUMNS), len(df_removed), REMOVED_COL_WIDTHS)
    wb.save(target_output)

    print(f"\n[THÀNH CÔNG] Đã xuất dataset chuẩn tại: {target_output}")


def parse_arguments() -> argparse.Namespace:
    """Xử lý các tham số dòng lệnh để chạy riêng biệt từng nền tảng."""
    parser = argparse.ArgumentParser(
        description="Pipeline tiền xử lý NLP bình luận mạng xã hội (Facebook & TikTok)"
    )
    parser.add_argument(
        "--source",
        choices=["facebook", "tiktok"],
        default="tiktok",
        help="Chọn nền tảng dữ liệu cần xử lý (mặc định: tiktok)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Đường dẫn file đầu vào tùy chỉnh (nếu muốn chạy file chỉ định)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Đường dẫn file đầu ra tùy chỉnh",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    platform_cfg = PLATFORM_CONFIGS[args.source]
    platform_name = platform_cfg["name"]

    print(f"\n[KHỞI ĐỘNG] Chế độ tiền xử lý nền tảng: {platform_name}")

    # Xác định file đầu vào
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = next((p for p in platform_cfg["candidates"] if p.exists()), None)

    if not input_file or not input_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file nguồn khả dụng cho {platform_name}! "
            f"Đã tìm kiếm: {[str(p.name) for p in platform_cfg['candidates']]}"
        )

    # Xác định file đầu ra
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = platform_cfg["output_file"]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    run_cleaning_pipeline(input_file=input_file, output_file=output_file)


if __name__ == "__main__":
    main()
