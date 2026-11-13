"""
Xây dựng tập dữ liệu gán nhãn cuối cùng (Final Labeled Dataset)
Hỗ trợ cả Facebook và TikTok bằng tham số --source facebook | tiktok.

Nguyên tắc hợp nhất:
1. Đọc file kết quả LLM đã qua rà soát (ALL_REVIEWED).
2. Đọc file kiểm tra chất lượng của con người (QC_CHECKLIST).
3. Ghi đè quyết định của con người lên kết quả LLM nếu có.
4. Xuất file final_labeled.xlsx chuẩn hóa theo schema thống nhất:
   ['Row ID', 'Comment ID', 'Nội dung RAW', 'Loại', 'Trạng thái', 'Nhãn']
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# Thiết lập UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent

PLATFORM_BUILD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "facebook": {
        "inter_dir": ROOT_DIR / "data" / "facebook" / "intermediate",
        "file_llm_candidates": (
            ROOT_DIR / "data" / "facebook" / "intermediate" / "all_comments_reviewed_v2.xlsx",
            ROOT_DIR / "all_comments_reviewed_v2.xlsx",
        ),
        "file_qc_candidates": (
            ROOT_DIR / "data" / "facebook" / "intermediate" / "Danh_sach_kiem_tra_nhan.xlsx",
            ROOT_DIR / "Danh_sach_kiem_tra_nhan.xlsx",
        ),
        "output_file": ROOT_DIR / "data" / "facebook" / "intermediate" / "final_labeled.xlsx",
    },
    "tiktok": {
        "inter_dir": ROOT_DIR / "data" / "tiktok" / "intermediate",
        "file_llm_candidates": (
            ROOT_DIR / "data" / "tiktok" / "intermediate" / "tiktok_comments_reviewed_v2.xlsx",
            ROOT_DIR / "data" / "tiktok" / "tiktok_comments_annotated.xlsx",
        ),
        "file_qc_candidates": (
            ROOT_DIR / "data" / "tiktok" / "intermediate" / "Danh_sach_kiem_tra_nhan.xlsx",
            ROOT_DIR / "data" / "tiktok" / "Danh_sach_kiem_tra_nhan.xlsx",
        ),
        "output_file": ROOT_DIR / "data" / "tiktok" / "intermediate" / "final_labeled.xlsx",
    },
}

FINAL_SCHEMA_COLUMNS = [
    "Row ID",
    "Comment ID",
    "Nội dung RAW",
    "Loại",
    "Trạng thái",
    "Nhãn",
]


def style_worksheet(ws, cols_count: int, row_count: int) -> None:
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

    for row in ws.iter_rows(min_row=2, max_row=row_count + 1):
        for cell in row:
            cell.border = border_thin


def run_build_final(source: str, input_llm_path: Optional[Path] = None, input_qc_path: Optional[Path] = None) -> None:
    print("=" * 80)
    print(f"XÂY DỰNG DATASET CUỐI CÙNG (FINAL LABELED) — NỀN TẢNG: {source.upper()}")
    print("=" * 80)

    cfg = PLATFORM_BUILD_CONFIGS[source]
    file_llm = input_llm_path or next((p for p in cfg["file_llm_candidates"] if p.exists()), None)
    file_qc = input_qc_path or next((p for p in cfg["file_qc_candidates"] if p.exists()), None)
    output_file = cfg["output_file"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not file_llm:
        raise FileNotFoundError(f"Không tìm thấy file kết quả LLM cho {source.upper()}!")
    print(f"[Xác nhận] File LLM : {file_llm.name}")

    # Đọc dữ liệu LLM
    try:
        df_llm = pd.read_excel(file_llm, sheet_name="ALL_REVIEWED")
    except Exception:
        df_llm = pd.read_excel(file_llm, sheet_name=0)

    # Chuẩn hóa tên cột file LLM
    for c_alias in ["Nội dung RAW", "Nội dung", "RAW Text", "comment_text"]:
        if c_alias in df_llm.columns:
            df_llm["Nội dung RAW"] = df_llm[c_alias]
            break

    for s_alias in ["Final Status", "filter_status", "Status", "Trạng thái"]:
        if s_alias in df_llm.columns:
            df_llm["Final Status"] = df_llm[s_alias]
            break

    for l_alias in ["Final Sentiment", "sentiment", "Sentiment", "Nhãn", "Final Label"]:
        if l_alias in df_llm.columns:
            df_llm["Final Sentiment"] = df_llm[l_alias]
            break

    row_id_col = "Row ID" if "Row ID" in df_llm.columns else "Row ID gốc" if "Row ID gốc" in df_llm.columns else None
    if not row_id_col:
        df_llm["Row ID"] = range(1, len(df_llm) + 1)
        row_id_col = "Row ID"

    if "Comment ID" not in df_llm.columns:
        df_llm["Comment ID"] = df_llm[row_id_col]

    if "Loại" not in df_llm.columns:
        df_llm["Loại"] = df_llm.get("Loại bình luận", "Comment")

    total_llm = len(df_llm)
    print(f"- Tổng số bản ghi từ LLM: {total_llm:,} dòng")

    # Đọc file QC (nếu có)
    qc_map = {}
    if file_qc and file_qc.exists():
        print(f"[Xác nhận] File QC con người: {file_qc.name}")
        try:
            df_qc = pd.read_excel(file_qc, sheet_name="QC_CHECKLIST")
        except Exception:
            df_qc = pd.read_excel(file_qc, sheet_name=0)

        qc_id_col = "Row ID gốc" if "Row ID gốc" in df_qc.columns else "Row ID" if "Row ID" in df_qc.columns else df_qc.columns[0]
        # Khử trùng
        df_qc = df_qc.drop_duplicates(subset=[qc_id_col], keep="last")

        for _, r in df_qc.iterrows():
            rid = int(r[qc_id_col])
            raw_st = r.get("Status người")
            raw_lb = r.get("Nhãn người")

            st = str(raw_st).strip().upper() if pd.notna(raw_st) and str(raw_st).strip() else None
            lb = str(raw_lb).strip().capitalize() if pd.notna(raw_lb) and str(raw_lb).strip() else None

            if st or lb:
                qc_map[rid] = {"status": st, "label": lb}
        print(f"- Số quyết định từ con người được nạp: {len(qc_map):,} dòng")
    else:
        print("- Không có file QC hoặc bỏ qua (sử dụng 100% quyết định từ LLM).")

    # Hợp nhất quyết định
    final_records = []
    override_count = 0

    for _, row in df_llm.iterrows():
        rid = int(row[row_id_col])
        cid = row["Comment ID"]
        text = row["Nội dung RAW"]
        c_type = row["Loại"]

        llm_status = str(row.get("Final Status", "")).strip().upper()
        llm_label = str(row.get("Final Sentiment", "")).strip().capitalize()
        if llm_label in ["Nan", "None", ""]:
            llm_label = None

        final_st = llm_status
        final_lb = llm_label

        # Override nếu có quyết định con người
        if rid in qc_map:
            override = qc_map[rid]
            if override["status"]:
                final_st = override["status"]
            if override["label"]:
                final_lb = override["label"]
            override_count += 1

        final_records.append({
            "Row ID": rid,
            "Comment ID": cid,
            "Nội dung RAW": text,
            "Loại": c_type,
            "Trạng thái": final_st,
            "Nhãn": final_lb if final_st == "KEEP" else "",
        })

    df_final = pd.DataFrame(final_records)
    print(f"\nĐã tổng hợp thành công {len(df_final):,} dòng (Ghi đè bởi con người: {override_count:,} dòng).")
    print(f"Phân bố trạng thái: {df_final['Trạng thái'].value_counts().to_dict()}")

    # Xuất file Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_final.to_excel(writer, sheet_name="FINAL_LABELED", index=False)

    import openpyxl
    wb = openpyxl.load_workbook(output_file)
    style_worksheet(wb["FINAL_LABELED"], len(FINAL_SCHEMA_COLUMNS), len(df_final))
    wb.save(output_file)
    print(f"[THÀNH CÔNG] Đã lưu file kết quả tại: {output_file}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tổng hợp nhãn cuối cùng từ LLM và Human QC.")
    parser.add_argument(
        "--source",
        choices=["facebook", "tiktok"],
        default="facebook",
        help="Chọn nền tảng: facebook hoặc tiktok (mặc định: facebook)",
    )
    args = parser.parse_args()
    run_build_final(args.source)


if __name__ == "__main__":
    main()
