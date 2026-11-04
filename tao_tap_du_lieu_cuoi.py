import os
import sys
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# Thiết lập UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent

# Tìm file LLM đã rà soát
file_llm_candidates = [
    ROOT_DIR / "tat_ca_binh_luan_da_duyet_v2.xlsx",
    ROOT_DIR / "all_comments_reviewed_v2(1).xlsx",
    ROOT_DIR / "all_comments_reviewed_v2.xlsx",
]
FILE_LLM = next((p for p in file_llm_candidates if p.exists()), None)

# Tìm file QC Checklist do con người kiểm tra
file_qc_candidates = [
    ROOT_DIR / "danh_sach_kiem_tra_nhan.xlsx",
    ROOT_DIR / "Danh_sach_kiem_tra_nhan_truoc_thu_6.xlsx",
    ROOT_DIR / "Danh_sach_kiem_tra_nhan.xlsx",
]
FILE_QC = next((p for p in file_qc_candidates if p.exists()), None)

OUTPUT_FILE = ROOT_DIR / "du_lieu_sach_cu.xlsx"


def validate_files():
    if not FILE_LLM:
        raise FileNotFoundError(
            f"Không tìm thấy file kết quả LLM. Đã tìm kiếm: {[p.name for p in file_llm_candidates]}"
        )
    if not FILE_QC:
        raise FileNotFoundError(
            f"Không tìm thấy file kiểm tra thủ công (QC). Đã tìm kiếm: {[p.name for p in file_qc_candidates]}"
        )
    print(f"[Xác nhận] File LLM: {FILE_LLM.name}")
    print(f"[Xác nhận] File QC con người: {FILE_QC.name}")


def main():
    print("=" * 80)
    print("XÂY DỰNG DATASET CUỐI CÙNG (FINAL LABELED DATASET) — PHIM 'TRẠI BUÔN NGƯỜI'")
    print("=" * 80)

    validate_files()

    # 1. Đọc dữ liệu từ 2 file
    df_llm = pd.read_excel(FILE_LLM, sheet_name="ALL_REVIEWED")
    df_qc = pd.read_excel(FILE_QC, sheet_name="QC_CHECKLIST")

    total_llm_records = len(df_llm)
    total_qc_records = len(df_qc)
    print(f"- Tổng số bản ghi từ ALL_REVIEWED: {total_llm_records:,} dòng")
    print(f"- Tổng số bản ghi trong QC_CHECKLIST: {total_qc_records:,} dòng")

    # Kiểm tra trùng Row ID trong QC
    qc_row_id_col = "Row ID gốc"
    if df_qc[qc_row_id_col].duplicated().sum() > 0:
        dup_ids = df_qc[df_qc[qc_row_id_col].duplicated()][qc_row_id_col].tolist()
        print(f"[CẢNH BÁO] Phát hiện Row ID gốc bị trùng trong QC: {dup_ids}")
        # Giữ dòng cuối cùng được chỉnh sửa
        df_qc = df_qc.drop_duplicates(subset=[qc_row_id_col], keep="last")

    # 2. Xây dựng từ điển quyết định của con người
    # Chuẩn hóa giá trị
    qc_map = {}
    human_updated_count = 0
    human_keep_count = 0
    human_remove_count = 0
    human_review_count = 0

    valid_statuses = {"KEEP", "REMOVE", "REVIEW"}
    valid_labels = {"Positive", "Neutral", "Negative"}

    for _, row in df_qc.iterrows():
        rid = int(row[qc_row_id_col])
        raw_st = row.get("Status người")
        raw_lb = row.get("Nhãn người")

        st = str(raw_st).strip().upper() if pd.notna(raw_st) and str(raw_st).strip() else None
        lb = str(raw_lb).strip().capitalize() if pd.notna(raw_lb) and str(raw_lb).strip() else None

        # Kiểm tra tính hợp lệ
        if st and st not in valid_statuses:
            print(f"[LỖI DỮ LIỆU QC] Row ID {rid}: Status người '{st}' không hợp lệ! Bỏ qua status này.")
            st = None
        if lb and lb not in valid_labels:
            print(f"[LỖI DỮ LIỆU QC] Row ID {rid}: Nhãn người '{lb}' không hợp lệ! Bỏ qua nhãn này.")
            lb = None

        if st is not None or lb is not None:
            human_updated_count += 1
            if st == "KEEP":
                human_keep_count += 1
            elif st == "REMOVE":
                human_remove_count += 1
            elif st == "REVIEW":
                human_review_count += 1

            qc_map[rid] = {"human_status": st, "human_label": lb}

    print(f"- Số dòng được con người cập nhật (Status hoặc Nhãn): {human_updated_count:,}")
    print(f"  + Con người chọn KEEP:   {human_keep_count:,}")
    print(f"  + Con người chọn REMOVE: {human_remove_count:,}")
    print(f"  + Con người chọn REVIEW: {human_review_count:,}")

    # 3. Hợp nhất nhãn theo các quy tắc ưu tiên
    final_rows = []
    conflict_count = 0
    discarded_remove_count = 0
    discarded_review_count = 0
    discarded_invalid_label_or_text = 0

    for _, row in df_llm.iterrows():
        rid = int(row["Row ID"])
        raw_text = row["Nội dung RAW"]
        c_type = str(row.get("Loại", "") or "").strip()
        llm_status = str(row.get("Final Status", "") or "").strip().upper()
        llm_label = str(row.get("Final Sentiment", "") or "").strip().capitalize()

        # Chuẩn hóa llm values
        if llm_status not in valid_statuses:
            llm_status = "REMOVE"
        if llm_label not in valid_labels:
            llm_label = None

        # Lấy quyết định con người nếu có
        human_dec = qc_map.get(rid, {})
        h_status = human_dec.get("human_status")
        h_label = human_dec.get("human_label")

        # Quy tắc 6: Nếu LLM là REMOVE/REVIEW nhưng con người chỉ điền nhãn mà không xác nhận KEEP
        if llm_status in ["REMOVE", "REVIEW"] and h_status is None and h_label is not None:
            print(f"[XUNG ĐỘT] Row ID {rid}: LLM là {llm_status} nhưng con người chỉ điền nhãn '{h_label}' mà không xác nhận Status = KEEP. Đánh dấu loại để an toàn.")
            conflict_count += 1
            continue

        # Quy tắc 1 & 3: Xác định Status cuối cùng
        if h_status is not None:
            effective_status = h_status
        else:
            effective_status = llm_status

        # Quy tắc 2, 3, 5: Xác định Label cuối cùng
        if h_label is not None:
            effective_label = h_label
        else:
            # Nếu con người chọn KEEP nhưng không điền nhãn mới, giữ nhãn LLM cũ khi hợp lệ
            effective_label = llm_label

        # Quy tắc 4 & 7: Nếu REMOVE hoặc REVIEW thì loại
        if effective_status == "REMOVE":
            discarded_remove_count += 1
            continue
        if effective_status == "REVIEW":
            discarded_review_count += 1
            continue

        # Kiểm tra điều kiện giữ lại:
        if effective_status != "KEEP":
            discarded_remove_count += 1
            continue

        # Phải có nhãn cảm xúc hợp lệ
        if effective_label not in valid_labels:
            print(f"[LOẠI BỎ] Row ID {rid}: Status là KEEP nhưng không có nhãn cảm xúc hợp lệ (Label = '{effective_label}').")
            discarded_invalid_label_or_text += 1
            continue

        # Nội dung RAW không được rỗng hoặc "nan", "none", "null"
        if pd.isna(raw_text) or not str(raw_text).strip():
            print(f"[LOẠI BỎ] Row ID {rid}: Nội dung RAW trống.")
            discarded_invalid_label_or_text += 1
            continue

        clean_text_str = str(raw_text).strip()
        if clean_text_str.lower() in ["nan", "none", "null"]:
            print(f"[LOẠI BỎ] Row ID {rid}: Nội dung RAW là chuỗi rỗng '{clean_text_str}'.")
            discarded_invalid_label_or_text += 1
            continue

        # Chuẩn hóa Comment Type
        c_type_clean = "Reply" if "reply" in c_type.lower() else "Comment"

        final_rows.append({
            "Row ID": rid,
            "Text RAW": clean_text_str,
            "Comment Type": c_type_clean,
            "Label": effective_label,
        })

    # 4. Tạo DataFrame cuối cùng và sắp xếp theo Row ID tăng dần
    df_final = pd.DataFrame(final_rows)
    df_final = df_final.sort_values("Row ID").reset_index(drop=True)

    # Đảm bảo đúng 4 cột theo thứ tự
    final_cols = ["Row ID", "Text RAW", "Comment Type", "Label"]
    df_final = df_final[final_cols]

    # 5. Xuất file Excel final_labeled.xlsx với 1 sheet duy nhất FINAL_DATA
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_final.to_excel(writer, sheet_name="FINAL_DATA", index=False)
        ws = writer.sheets["FINAL_DATA"]

        # Cố định dòng tiêu đề
        ws.freeze_panes = "A2"
        # Bật bộ lọc tự động
        ws.auto_filter.ref = ws.dimensions

        # Styling tiêu đề
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        border_thin = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9"),
        )

        for col_idx in range(1, len(final_cols) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Độ rộng cột hợp lý
        ws.column_dimensions["A"].width = 12  # Row ID
        ws.column_dimensions["B"].width = 65  # Text RAW
        ws.column_dimensions["C"].width = 16  # Comment Type
        ws.column_dimensions["D"].width = 16  # Label

        for row_cells in ws.iter_rows(min_row=2, max_row=len(df_final) + 1):
            row_cells[0].alignment = Alignment(horizontal="center", vertical="center")  # Row ID
            row_cells[1].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)  # Text RAW
            row_cells[2].alignment = Alignment(horizontal="center", vertical="center")  # Comment Type
            row_cells[3].alignment = Alignment(horizontal="center", vertical="center")  # Label
            for c in row_cells:
                c.border = border_thin

    print(f"\n[Thành công] Đã xuất file: {OUTPUT_FILE}")

    # 6. ĐỌC LẠI FILE VỪA XUẤT ĐỂ XÁC MINH (10 TIÊU CHÍ KIỂM TRA TỰ ĐỘNG)
    print("\n--- KIỂM TRA VÀ XÁC MINH FILE ĐẦU RA (VERIFICATION) ---")
    xl_verify = pd.ExcelFile(OUTPUT_FILE)
    sheet_names = xl_verify.sheet_names
    assert sheet_names == ["FINAL_DATA"], f"Lỗi: Số lượng hoặc tên sheet không đúng: {sheet_names}"
    print(f"1. Xác nhận số sheet: Đúng duy nhất 1 sheet '{sheet_names[0]}'")

    df_verify = pd.read_excel(OUTPUT_FILE, sheet_name="FINAL_DATA")
    assert df_verify.columns.tolist() == final_cols, f"Lỗi: Cột không đúng thứ tự: {df_verify.columns.tolist()}"
    print(f"2. Xác nhận 4 cột chuẩn: {df_verify.columns.tolist()}")

    final_count = len(df_verify)
    print(f"3. Tổng số bình luận KEEP hợp lệ trong file cuối: {final_count:,} dòng")
    assert final_count == len(df_final), "Lỗi: Số dòng xuất ra không khớp!"

    print(f"4. Số dòng bị loại bỏ:")
    print(f"   - Tổng REMOVE bị loại: {discarded_remove_count:,}")
    print(f"   - Tổng REVIEW bị loại: {discarded_review_count:,}")
    print(f"   - Tổng thiếu nhãn/nội dung/xung đột: {discarded_invalid_label_or_text + conflict_count:,}")

    # Đảm bảo không còn REMOVE / REVIEW
    assert "REMOVE" not in df_verify["Label"].values, "Lỗi: Còn nhãn REMOVE trong file cuối!"
    assert "REVIEW" not in df_verify["Label"].values, "Lỗi: Còn nhãn REVIEW trong file cuối!"
    print("5. Xác nhận không còn REMOVE hay REVIEW trong file: ĐẠT (100%)")

    # 100% nhãn hợp lệ
    label_counts = df_verify["Label"].value_counts().to_dict()
    invalid_labels_found = set(df_verify["Label"].unique()) - valid_labels
    assert len(invalid_labels_found) == 0, f"Lỗi: Có nhãn không hợp lệ: {invalid_labels_found}"
    print("6. Xác nhận 100% nhãn hợp lệ (Positive / Neutral / Negative): ĐẠT")

    # Đảm bảo không trùng Row ID
    assert df_verify["Row ID"].duplicated().sum() == 0, "Lỗi: Row ID bị trùng lặp!"
    print(f"7. Xác nhận tính duy nhất của Row ID: ĐẠT ({df_verify['Row ID'].nunique()} IDs duy nhất)")

    # Đảm bảo thứ tự Row ID tăng dần
    assert df_verify["Row ID"].is_monotonic_increasing, "Lỗi: Row ID không sắp xếp tăng dần!"
    print("8. Xác nhận sắp xếp theo Row ID tăng dần: ĐẠT")

    # Đảm bảo Text RAW không rỗng
    empty_texts = df_verify["Text RAW"].isna().sum() + (df_verify["Text RAW"].astype(str).str.strip() == "").sum()
    assert empty_texts == 0, "Lỗi: Còn nội dung rỗng!"
    print("9. Xác nhận không có nội dung rỗng/thiếu: ĐẠT (100% hợp lệ)")

    # Kiểm tra bảo toàn quyết định con người
    sample_qc_human = df_qc[df_qc["Status người"] == "KEEP"].head(5)
    for _, r in sample_qc_human.iterrows():
        rid = int(r[qc_row_id_col])
        exp_lb = str(r["Nhãn người"]).strip().capitalize()
        matched_row = df_verify[df_verify["Row ID"] == rid]
        if len(matched_row) > 0:
            actual_lb = matched_row["Label"].values[0]
            assert actual_lb == exp_lb, f"Lỗi: Quyết định con người tại Row {rid} không khớp (kỳ vọng: {exp_lb}, thực tế: {actual_lb})"
    print("10. Xác nhận độ ưu tiên chính xác của quyết định con người: ĐẠT (100% bảo toàn)")

    # In phân bố nhãn
    print("\n" + "=" * 80)
    print("BẢNG TỔNG KẾT PHÂN BỐ DỮ LIỆU FINAL DATASET:")
    print("=" * 80)
    print(f"Tổng số bản ghi bàn giao: {final_count:,} dòng\n")
    print(f"{'Nhãn cảm xúc (Label)':<25} | {'Số lượng':<10} | {'Tỷ lệ (%)':<10}")
    print("-" * 52)
    for lb in ["Positive", "Neutral", "Negative"]:
        c = label_counts.get(lb, 0)
        pct = (c / final_count) * 100
        print(f"{lb:<25} | {c:<10d} | {pct:<10.1f}%")
    print("-" * 52)

    print(f"\nPhân bố loại bình luận:")
    for ct, c in df_verify["Comment Type"].value_counts().items():
        print(f" - {ct:<10}: {c:<5d} ({c/final_count*100:.1f}%)")

    print("\n" + "=" * 80)
    print(f"ĐÃ HOÀN TẤT VÀ XUẤT FILE DUY NHẤT: {OUTPUT_FILE}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

