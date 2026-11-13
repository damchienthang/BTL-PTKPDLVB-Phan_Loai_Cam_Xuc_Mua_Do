"""
Script trích xuất danh sách các bình luận cần con người kiểm tra (Human QC Review).
Hỗ trợ cả Facebook và TikTok bằng tham số --source facebook | tiktok.

Tính năng:
1. Tự động đọc file review v2 (hoặc file gán nhãn v1 nếu chưa có v2).
2. Lọc ra các dòng mà AI phân vân, độ tin cậy thấp, hoặc trạng thái REVIEW.
3. Xuất ra file 'Danh_sach_kiem_tra_nhan.xlsx' chuẩn 3 sheet:
   - QC_CHECKLIST: Bảng công việc kèm cột tô màu xanh lá để người điền (Status người, Nhãn người).
   - HUONG_DAN: Hướng dẫn chi tiết cách duyệt và các quy chuẩn nhãn.
   - THONG_KE: Báo cáo số liệu tổng quan về các ca cần rà soát.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# Thiết lập UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent

PLATFORM_QC_CONFIGS: Dict[str, Dict[str, Any]] = {
    "facebook": {
        "name": "FACEBOOK",
        "candidates": (
            ROOT_DIR / "data" / "facebook" / "intermediate" / "all_comments_reviewed_v2.xlsx",
            ROOT_DIR / "data" / "facebook" / "intermediate" / "all_comments_annotated.xlsx",
        ),
        "output_file": ROOT_DIR / "data" / "facebook" / "intermediate" / "Danh_sach_kiem_tra_nhan.xlsx",
    },
    "tiktok": {
        "name": "TIKTOK",
        "candidates": (
            ROOT_DIR / "data" / "tiktok" / "intermediate" / "tiktok_comments_reviewed_v2.xlsx",
            ROOT_DIR / "data" / "tiktok" / "tiktok_comments_annotated.xlsx",
        ),
        "output_file": ROOT_DIR / "data" / "tiktok" / "intermediate" / "Danh_sach_kiem_tra_nhan.xlsx",
    },
}

QC_COLUMNS = [
    "Ưu tiên",
    "Row ID gốc",
    "Nội dung RAW",
    "Loại",
    "Status LLM",
    "Nhãn LLM",
    "Lý do cần kiểm tra",
    "Status người",
    "Nhãn người",
    "Ghi chú",
]


def extract_qc_records_from_reviewed(df_reviewed: pd.DataFrame, df_human_sheet: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Trích xuất từ file kết quả review v2 (all_comments_reviewed_v2.xlsx)."""
    qc_rows = []

    # 1. Nếu file đã có sẵn sheet HUMAN_REVIEW
    if df_human_sheet is not None and not df_human_sheet.empty:
        for _, r in df_human_sheet.iterrows():
            rid = int(r["Row ID"])
            raw = str(r.get("Nội dung RAW", "") or "").strip()
            c_type = str(r.get("Loại", "Comment"))
            st_llm = str(r.get("Proposed Status") or r.get("Final Status") or "")
            lb_llm = str(r.get("Proposed Sentiment") or r.get("Final Sentiment") or "")
            reason = str(r.get("Reason", ""))
            conf = str(r.get("Confidence", "")).upper()

            prio = "P0" if st_llm == "REVIEW" else "P1" if conf == "LOW" else "P2"
            qc_rows.append({
                "Ưu tiên": prio,
                "Row ID gốc": rid,
                "Nội dung RAW": raw,
                "Loại": c_type,
                "Status LLM": st_llm,
                "Nhãn LLM": lb_llm if st_llm == "KEEP" else "",
                "Lý do cần kiểm tra": reason,
                "Status người": "",
                "Nhãn người": "",
                "Ghi chú": "",
            })
    else:
        # 2. Quét từ bảng ALL_REVIEWED
        for _, r in df_reviewed.iterrows():
            rid = int(r.get("Row ID", r.name + 2))
            raw = str(r.get("Nội dung RAW") or r.get("Nội dung") or "").strip()
            c_type = str(r.get("Loại", "Comment"))
            st_llm = str(r.get("Final Status", "")).strip().upper()
            lb_llm = str(r.get("Final Sentiment", "")).strip().capitalize()
            conf = str(r.get("Confidence", "")).strip().upper()
            reason = str(r.get("Reason", ""))

            if st_llm == "REVIEW":
                prio = "P0"
                note = f"LLM chưa kết luận: {reason}"
            elif conf == "LOW":
                prio = "P1"
                note = f"Độ tin cậy thấp (LOW): {reason}"
            else:
                continue

            qc_rows.append({
                "Ưu tiên": prio,
                "Row ID gốc": rid,
                "Nội dung RAW": raw,
                "Loại": c_type,
                "Status LLM": st_llm,
                "Nhãn LLM": lb_llm if st_llm == "KEEP" else "",
                "Lý do cần kiểm tra": note,
                "Status người": "",
                "Nhãn người": "",
                "Ghi chú": "",
            })

    df_qc = pd.DataFrame(qc_rows)
    if not df_qc.empty:
        df_qc = df_qc.sort_values(by=["Ưu tiên", "Row ID gốc"]).reset_index(drop=True)
    return df_qc


def extract_qc_records_from_annotated(df_anno: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất từ file gán nhãn lần 1 (all_comments_annotated.xlsx) nếu chưa có v2."""
    qc_rows = []
    for idx, r in df_anno.iterrows():
        rid = idx + 2
        raw = str(r.get("Nội dung") or r.get("Nội dung RAW") or "").strip()
        c_type = str(r.get("Loại", "Comment"))
        st_llm = str(r.get("filter_status", "")).strip().upper()
        lb_llm = str(r.get("sentiment", "")).strip().capitalize()
        conf = str(r.get("confidence_level", "")).strip().upper()
        rev_req = bool(r.get("review_required", False))
        reason = str(r.get("reason", ""))

        if st_llm == "REVIEW":
            prio = "P0"
            note = f"AI chưa thể kết luận: {reason}"
        elif conf == "LOW":
            prio = "P1"
            note = f"Độ tin cậy thấp: {reason}"
        elif rev_req or conf == "MEDIUM":
            prio = "P2"
            note = f"AI đề xuất người rà soát: {reason}"
        else:
            continue

        qc_rows.append({
            "Ưu tiên": prio,
            "Row ID gốc": rid,
            "Nội dung RAW": raw,
            "Loại": c_type,
            "Status LLM": st_llm,
            "Nhãn LLM": lb_llm if st_llm == "KEEP" else "",
            "Lý do cần kiểm tra": note,
            "Status người": "",
            "Nhãn người": "",
            "Ghi chú": "",
        })

    df_qc = pd.DataFrame(qc_rows)
    if not df_qc.empty:
        df_qc = df_qc.sort_values(by=["Ưu tiên", "Row ID gốc"]).reset_index(drop=True)
    return df_qc


def export_qc_excel(df_qc: pd.DataFrame, output_path: Path, platform_name: str, total_source_rows: int) -> None:
    """Xuất file Excel chuyên nghiệp gồm 3 sheet: QC_CHECKLIST, HUONG_DAN, THONG_KE."""
    p0_count = (df_qc["Ưu tiên"] == "P0").sum() if not df_qc.empty else 0
    p1_count = (df_qc["Ưu tiên"] == "P1").sum() if not df_qc.empty else 0
    p2_count = (df_qc["Ưu tiên"] == "P2").sum() if not df_qc.empty else 0

    df_stat = pd.DataFrame([
        {"CHỈ SỐ": f"Tổng số bình luận {platform_name}", "SỐ LƯỢNG": total_source_rows},
        {"CHỈ SỐ": "Số bình luận cần người kiểm tra (Tổng)", "SỐ LƯỢNG": len(df_qc)},
        {"CHỈ SỐ": "- Ưu tiên P0 (AI chưa kết luận - REVIEW)", "SỐ LƯỢNG": p0_count},
        {"CHỈ SỐ": "- Ưu tiên P1 (Độ tin cậy LOW)", "SỐ LƯỢNG": p1_count},
        {"CHỈ SỐ": "- Ưu tiên P2 (Độ tin cậy MEDIUM / Cần rà soát)", "SỐ LƯỢNG": p2_count},
    ])

    huong_dan_rows = [
        ["HƯỚNG DẪN KIỂM TRA CHẤT LƯỢNG NHÃN (QC CHECKLIST)", ""],
        ["1. Mục tiêu", "Rà soát và đưa ra quyết định cuối cùng cho các bình luận mà AI phân vân."],
        ["2. Cột 'Status người' (Cột H)", "Điền 'KEEP' nếu liên quan đến phim; điền 'REMOVE' nếu là rác/spam/tag/ngoài lề."],
        ["3. Cột 'Nhãn người' (Cột I)", "Nếu Status là KEEP, điền 1 trong 3 nhãn: Positive, Negative, Neutral. Nếu REMOVE thì để trống."],
        ["4. Ưu tiên xử lý", "Làm từ P0 (bắt buộc) -> P1 (độ tin cậy thấp) -> P2."],
        ["5. Tích hợp sau khi check", "Lưu file này lại và chạy 'py build_final_labeled_dataset.py --source ...' để tự động ghi đè nhãn."],
    ]
    df_hd = pd.DataFrame(huong_dan_rows, columns=["HẠNG MỤC", "HƯỚNG DẪN CHI TIẾT"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_qc.to_excel(writer, sheet_name="QC_CHECKLIST", index=False)
        df_hd.to_excel(writer, sheet_name="HUONG_DAN", index=False)
        df_stat.to_excel(writer, sheet_name="THONG_KE", index=False)

        wb = writer.book
        ws_qc = wb["QC_CHECKLIST"]
        ws_qc.freeze_panes = "A2"
        ws_qc.auto_filter.ref = ws_qc.dimensions

        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        border_thin = Border(
            left=Side(style="thin", color="D3D3D3"),
            right=Side(style="thin", color="D3D3D3"),
            top=Side(style="thin", color="D3D3D3"),
            bottom=Side(style="thin", color="D3D3D3"),
        )
        p0_fill = PatternFill(start_color="FFD9D9", end_color="FFD9D9", fill_type="solid")
        p1_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        human_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")

        # Header styling
        for c_idx in range(1, len(QC_COLUMNS) + 1):
            cell = ws_qc.cell(row=1, column=c_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        ws_qc.row_dimensions[1].height = 28

        # Data styling
        for r_idx in range(2, len(df_qc) + 2):
            prio_val = ws_qc.cell(row=r_idx, column=1).value
            for c_idx in range(1, len(QC_COLUMNS) + 1):
                c = ws_qc.cell(row=r_idx, column=c_idx)
                c.font = Font(name="Segoe UI", size=10)
                c.border = border_thin

                if c_idx == 1:
                    if prio_val == "P0":
                        c.fill = p0_fill
                    elif prio_val == "P1":
                        c.fill = p1_fill
                    c.alignment = Alignment(horizontal="center", vertical="center")
                elif c_idx in [8, 9]:  # Status người, Nhãn người
                    c.fill = human_fill
                    c.alignment = Alignment(horizontal="center", vertical="center")
                elif c_idx in [2, 4, 5, 6]:
                    c.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

            ws_qc.row_dimensions[r_idx].height = 24

        col_widths = {
            1: 10,  # Ưu tiên
            2: 12,  # Row ID gốc
            3: 50,  # Nội dung RAW
            4: 12,  # Loại
            5: 14,  # Status LLM
            6: 14,  # Nhãn LLM
            7: 45,  # Lý do cần kiểm tra
            8: 16,  # Status người
            9: 16,  # Nhãn người
            10: 25, # Ghi chú
        }
        for col_i, width in col_widths.items():
            col_letter = get_column_letter(col_i)
            ws_qc.column_dimensions[col_letter].width = width

    print(f"[THÀNH CÔNG] Đã xuất file: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích xuất danh sách kiểm tra nhãn cho con người (QC Checklist).")
    parser.add_argument(
        "--source",
        choices=["facebook", "tiktok"],
        default="tiktok",
        help="Chọn nền tảng dữ liệu (mặc định: tiktok)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="File đầu vào tùy chỉnh",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="File đầu ra tùy chỉnh",
    )
    args = parser.parse_args()

    cfg = PLATFORM_QC_CONFIGS[args.source]
    platform_name = cfg["name"]

    print("=" * 80)
    print(f"TRÍCH XUẤT DANH SÁCH KIỂM TRA NHÃN (QC CHECKLIST) — {platform_name}")
    print("=" * 80)

    # Tìm file nguồn
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = next((p for p in cfg["candidates"] if p.exists()), None)

    if not input_path or not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file kết quả gán nhãn hoặc review cho {platform_name}!")

    print(f"[Đọc dữ liệu] Nguồn vào: {input_path.name}")
    xls = pd.ExcelFile(input_path)

    df_human_sheet = pd.read_excel(input_path, sheet_name="HUMAN_REVIEW") if "HUMAN_REVIEW" in xls.sheet_names else None

    if "ALL_REVIEWED" in xls.sheet_names:
        df_base = pd.read_excel(input_path, sheet_name="ALL_REVIEWED")
        df_qc = extract_qc_records_from_reviewed(df_base, df_human_sheet)
    elif "ALL_ANNOTATED" in xls.sheet_names:
        df_base = pd.read_excel(input_path, sheet_name="ALL_ANNOTATED")
        df_qc = extract_qc_records_from_annotated(df_base)
    else:
        df_base = pd.read_excel(input_path, sheet_name=0)
        df_qc = extract_qc_records_from_reviewed(df_base, df_human_sheet)

    total_rows = len(df_base)
    output_path = Path(args.output) if args.output else cfg["output_file"]

    print(f"- Tổng số dòng trong file gốc: {total_rows:,} dòng")
    print(f"- Tổng số dòng cần con người kiểm tra: {len(df_qc):,} dòng ({len(df_qc)/total_rows*100:.1f}%)")

    export_qc_excel(df_qc, output_path, platform_name, total_rows)


if __name__ == "__main__":
    main()

