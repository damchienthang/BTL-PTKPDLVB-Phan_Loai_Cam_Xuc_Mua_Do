import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from pydantic import BaseModel, Field

# Thêm hỗ trợ utf-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

INPUT_FILE = ROOT_DIR / "all_comments_annotated.xlsx"
OUTPUT_FILE = ROOT_DIR / "all_comments_reviewed_v2.xlsx"
CHECKPOINT_FULL = ROOT_DIR / "checkpoint_full_reviewed_v2.json"
CHECKPOINT_SAMPLE = ROOT_DIR / "checkpoint_review_sample_100.json"
TARGET_MOVIE = "Trại buôn người"
BATCH_SIZE = 30


# ==============================================================================
# 1. Pydantic Schema cho Structured Output
# ==============================================================================
class ReviewedItem(BaseModel):
    row_id: int = Field(description="Số dòng Excel gốc (Row ID) của bình luận")
    filter_status: Literal["KEEP", "REMOVE", "REVIEW"] = Field(
        description="KEEP: liên quan phim; REMOVE: rác/spam/tag/phụ thuộc hoàn toàn bình luận cha/ngoài lề; REVIEW: mơ hồ, trái chiều, không rõ đối tượng"
    )
    sentiment: Optional[Literal["Positive", "Negative", "Neutral"]] = Field(
        default=None,
        description="CHỈ gán khi filter_status là KEEP. Nếu REMOVE hoặc REVIEW thì BẮT BUỘC là null",
    )
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Độ tin cậy của quyết định rà soát"
    )
    reason: str = Field(
        description="Giải thích ngắn gọn bằng tiếng Việt lý do giữ/loại/chọn cảm xúc và lý do sửa nhãn cũ (nếu có)"
    )


class BatchReviewResponse(BaseModel):
    annotations: List[ReviewedItem]


# ==============================================================================
# 2. Quản lý Pool API Key & Tự Động Đảo Key
# ==============================================================================
class KeyManager:
    def __init__(self, keys: List[str]):
        seen = set()
        clean_keys = []
        for k in keys:
            k = k.strip()
            if k and k not in seen:
                clean_keys.append(k)
                seen.add(k)
        self.keys = clean_keys
        self.current_idx = 0
        self.exhausted_keys = set()
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self.keys:
            raise ValueError("Không tìm thấy API key nào!")
        current_key = self.keys[self.current_idx]
        from google import genai

        self._client = genai.Client(api_key=current_key)
        masked = current_key[:8] + "..." + current_key[-4:]
        print(f" -> Đang dùng API Key [{self.current_idx + 1}/{len(self.keys)}]: {masked}")

    @property
    def client(self):
        return self._client

    def rotate_key(self) -> bool:
        self.exhausted_keys.add(self.current_idx)
        print(f"\n[QUOTA EXHAUSTED] Key [{self.current_idx + 1}/{len(self.keys)}] đã hết hạn ngạch!")
        for i in range(len(self.keys)):
            if i not in self.exhausted_keys:
                self.current_idx = i
                self._init_client()
                return True
        print("\n[TẤT CẢ KEY ĐÃ HẾT QUOTA] Toàn bộ danh sách API key đều đã chạm giới hạn!")
        return False


# ==============================================================================
# 3. Prompt Rà Soát Chất Lượng Dữ Liệu
# ==============================================================================
def build_review_prompt(batch_items: List[Dict[str, Any]]) -> str:
    prompt = f"""Bạn là chuyên gia NLP Senior và Data Annotation Auditor kiểm soát chất lượng dữ liệu cho bài toán phân loại cảm xúc bình luận Facebook về bộ phim "{TARGET_MOVIE}".

BẠN ĐANG RÀ SOÁT LẠI NHÃN CŨ ĐỂ PHÁT HIỆN VÀ SỬA LỖI THEO CÁC NGUYÊN TẮC BẮT BUỘC SAU:

### NGUYÊN TẮC CỐT LÕI:
"Giữ bình luận có thể xác định được đối tượng là bộ phim mục tiêu và xác định được cảm xúc từ CHÍNH NỘI DUNG BÌNH LUẬN ĐỘC LẬP, KHÔNG CẦN SUY RA CẢM XÚC TỪ BÌNH LUẬN CHA."
- Được phép hiểu "phim này", "bộ phim này", "phim" là nói về bộ phim mục tiêu trừ khi có dấu hiệu đang nói về phim khác.
- Không bắt buộc bình luận phải nhắc đúng tên "{TARGET_MOVIE}".

### 1. QUY TẮC SÀNG LỌC (filter_status):
* KEEP:
  - Có nội dung thực sự liên quan đến phim mục tiêu.
  - Nội dung TỰ THỂ HIỆN ĐƯỢC cảm xúc hoặc thông tin khách quan về phim khi đọc độc lập.
  - Đánh giá diễn xuất, kịch bản, âm thanh, bối cảnh, cảm xúc khi xem ("xem mà khóc", "ám ảnh", "sợ thót tim" -> KEEP, Positive).
  - So sánh phim mục tiêu với phim khác nhưng có nhận xét phim mục tiêu ("phim này không hay bằng Mưa đỏ" -> KEEP, Negative).
* REMOVE:
  - Spam, quảng cáo, bán hàng, link rác, chỉ tag tên (@Tên).
  - Nội dung vô nghĩa, chỉ emoji không thể diễn giải.
  - Trò chuyện riêng tư giữa người dùng không liên quan đến phim.
  - Chỉ bàn về bộ phim khác.
  - Chỉ nhận xét ngoại hình, đời tư, ngoài đời của diễn viên mà KHÔNG đánh giá diễn xuất hay bộ phim.
  - [CỰC KỲ QUAN TRỌNG ĐỐI VỚI REPLY]: Loại BỎ TẤT CẢ các Reply phụ thuộc hoàn toàn vào bình luận cha để hiểu ý kiến:
    + "Chuẩn luôn" -> REMOVE
    + "Tôi cũng nghĩ vậy" -> REMOVE
    + "Không đồng ý" -> REMOVE
    + "2 em mà?" -> REMOVE
    + "Chốt kèo" -> REMOVE
    + "Ukm" / "Dạ" / "Đúng rồi" / "ok" -> REMOVE
  - [LƯU Ý ĐẶC BIỆT VỀ NHÂN VẬT]: KHÔNG ĐƯỢC tự động gán Negative chỉ vì bình luận nói một nhân vật trong phim là người ác, đê tiện, độc ác hay đáng sợ (ví dụ: "lão này đóng phản diện đạt ghê, nhìn ác vãi" -> là khen diễn xuất -> KEEP, Positive; "ông trùm ác dã man" -> nhận xét nhân vật chứ KHÔNG chê phim).
* REVIEW:
  - Không rõ đang đánh giá phim hay đánh giá một người ngoài đời.
  - Bình luận chưa đủ ngữ cảnh để xác định đối tượng.
  - Vừa khen vừa chê cân bằng, không có thái độ nổi trội.
  - Mỉa mai quá phức tạp hoặc lỗi diễn đạt gây khó hiểu.
  * Khi chọn REVIEW, sentiment BẮT BUỘC PHẢI LÀ null!

### 2. QUY TẮC GÁN NHÃN CẢM XÚC (sentiment - CHỈ DÀNH CHO KEEP):
- Positive: Khen ngợi phim, diễn xuất, kịch bản, hình ảnh, sự hài lòng, xúc động. Cảm xúc sợ hãi/khóc do phim mang lại là khen phim chạm cảm xúc -> Positive.
- Negative: Chê bai phim dở, kịch bản tệ, diễn xuất đơ, thất vọng, khuyên không xem. So sánh thấy phim mục tiêu kém hơn phim khác -> Negative.
- Neutral: Liên quan đến phim nhưng chỉ hỏi hoặc cung cấp thông tin khách quan (hỏi rạp chiếu, giờ chiếu, bao nhiêu phút, diễn viên nào đóng) mà KHÔNG thể hiện thái độ đánh giá.
* TUYỆT ĐỐI KHÔNG gán Neutral cho bình luận thiếu ngữ cảnh, vô nghĩa hoặc không liên quan!

---
DANH SÁCH BÌNH LUẬN CẦN RÀ SOÁT:
"""
    for item in batch_items:
        r_id = item["row_id"]
        text = str(item["Nội dung"] or "")
        c_type = item["Loại"]
        old_status = item.get("filter_status", "")
        old_sent = item.get("sentiment", "")
        prompt += f"\n- [Row ID: {r_id}] | Loại: {c_type} | Nhãn cũ: {old_status} - {old_sent}\n"
        prompt += f"  Nội dung: \"{text}\"\n"

    prompt += "\nTrả về JSON tuân thủ schema BatchReviewResponse."
    return prompt


# ==============================================================================
# 4. Gọi API với Retry và Key Rotation
# ==============================================================================
def review_batch_with_retry(
    key_manager: KeyManager,
    model_name: str,
    batch_items: List[Dict[str, Any]],
    max_retries: int = 5,
):
    from google.genai import types

    prompt = build_review_prompt(batch_items)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BatchReviewResponse,
        temperature=0.1,
    )

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            response = key_manager.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            res_dict = json.loads(response.text)
            batch_res = BatchReviewResponse.model_validate(res_dict)

            # Kiểm tra đủ row_id
            returned_ids = {item.row_id for item in batch_res.annotations}
            expected_ids = {item["row_id"] for item in batch_items}
            if not expected_ids.issubset(returned_ids):
                missing = expected_ids - returned_ids
                print(f"Warning: Batch thiếu Row ID: {missing}. Retry lần {attempt}...")
                time.sleep(2 * attempt)
                continue

            tokens = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "candidates_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
            } if getattr(response, "usage_metadata", None) else {}

            return batch_res.annotations, tokens

        except Exception as e:
            err_str = str(e)
            print(f"\nLỗi khi gọi API (lần {attempt}/{max_retries}): {err_str[:120]}")

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str:
                has_next = key_manager.rotate_key()
                if has_next:
                    print("-> Đã chuyển sang key mới, thử lại batch ngay lập tức...")
                    attempt = 0
                    time.sleep(1)
                    continue
                else:
                    return None, {}

            if attempt == max_retries:
                print(f"Đã thử {max_retries} lần không thành công cho batch này.")
                return None, {}

            wait_time = 3 * attempt
            print(f"Chờ {wait_time}s trước khi retry...")
            time.sleep(wait_time)

    return None, {}


# ==============================================================================
# 5. Xuất File Excel all_comments_reviewed_v2.xlsx
# ==============================================================================
def export_reviewed_excel(df_all: pd.DataFrame, review_dict: Dict[int, Dict[str, Any]], output_path: Path):
    print("\n--- Đang tạo và định dạng file Excel all_comments_reviewed_v2.xlsx ---")

    all_reviewed_rows = []
    change_log_rows = []
    human_review_rows = []

    for _, row in df_all.iterrows():
        r_id = int(row["row_id"])
        raw_text = str(row.get("Nội dung", "") or "")
        c_type = str(row.get("Loại", "") or "")

        old_status = str(row.get("filter_status", "") or "")
        old_sent = str(row.get("sentiment", "") or "")
        if old_sent in ["nan", "None", "NULL"]:
            old_sent = ""

        rev = review_dict.get(r_id, {})
        new_status = rev.get("filter_status", old_status)
        new_sent = rev.get("sentiment") or ""
        confidence = rev.get("confidence_level", "MEDIUM")
        reason = rev.get("reason", "")

        # Ràng buộc: Nếu REMOVE hoặc REVIEW thì sentiment phải rỗng
        if new_status in ["REMOVE", "REVIEW"]:
            new_sent = ""

        # Sheet 1: ALL_REVIEWED
        all_reviewed_rows.append({
            "Row ID": r_id,
            "Nội dung RAW": raw_text,
            "Loại": c_type,
            "Final Status": new_status,
            "Final Sentiment": new_sent,
            "Confidence": confidence,
            "Reason": reason,
        })

        # Kiểm tra thay đổi để ghi vào Sheet 3: CHANGE_LOG
        is_status_changed = old_status != new_status
        is_sent_changed = (old_status == "KEEP" and new_status == "KEEP" and old_sent != new_sent)

        if is_status_changed or is_sent_changed:
            change_type = ""
            if is_status_changed:
                change_type = f"{old_status} -> {new_status}"
            elif is_sent_changed:
                change_type = f"Sentiment: {old_sent} -> {new_sent}"

            change_log_rows.append({
                "Row ID": r_id,
                "Nội dung RAW": raw_text,
                "Loại": c_type,
                "Old Status": old_status,
                "New Status": new_status,
                "Old Sentiment": old_sent,
                "New Sentiment": new_sent,
                "Change Type": change_type,
                "Reason": reason,
            })

        # Sheet 2: HUMAN_REVIEW (những dòng cần con người quyết định)
        if new_status == "REVIEW" or confidence == "LOW":
            human_review_rows.append({
                "Row ID": r_id,
                "Nội dung RAW": raw_text,
                "Loại": c_type,
                "Old Status": old_status,
                "Proposed Status": new_status,
                "Old Sentiment": old_sent,
                "Proposed Sentiment": new_sent,
                "Confidence": confidence,
                "Reason": reason,
                "Human Decision": "",
                "Human Label": "",
            })

    df_all_reviewed = pd.DataFrame(all_reviewed_rows)
    df_change_log = pd.DataFrame(change_log_rows) if change_log_rows else pd.DataFrame(columns=[
        "Row ID", "Nội dung RAW", "Loại", "Old Status", "New Status", "Old Sentiment", "New Sentiment", "Change Type", "Reason"
    ])
    df_human_review = pd.DataFrame(human_review_rows) if human_review_rows else pd.DataFrame(columns=[
        "Row ID", "Nội dung RAW", "Loại", "Old Status", "Proposed Status", "Old Sentiment", "Proposed Sentiment", "Confidence", "Reason", "Human Decision", "Human Label"
    ])

    # Sheet 4: SUMMARY (Thống kê tổng hợp)
    summary_data = [
        {"Chỉ số": "Tổng số bình luận trong toàn bộ dataset", "Số lượng": len(df_all_reviewed), "Tỷ lệ": "100.0%"},
        {"Chỉ số": "Số dòng đã được rà soát thực tế", "Số lượng": len(review_dict), "Tỷ lệ": f"{len(review_dict)/len(df_all_reviewed)*100:.1f}%"},
        {"Chỉ số": "Số dòng giữ lại (KEEP)", "Số lượng": (df_all_reviewed["Final Status"] == "KEEP").sum(), "Tỷ lệ": f"{(df_all_reviewed['Final Status'] == 'KEEP').sum()/len(df_all_reviewed)*100:.1f}%"},
        {"Chỉ số": "Số dòng loại bỏ (REMOVE)", "Số lượng": (df_all_reviewed["Final Status"] == "REMOVE").sum(), "Tỷ lệ": f"{(df_all_reviewed['Final Status'] == 'REMOVE').sum()/len(df_all_reviewed)*100:.1f}%"},
        {"Chỉ số": "Số dòng cần người xem lại (REVIEW)", "Số lượng": (df_all_reviewed["Final Status"] == "REVIEW").sum(), "Tỷ lệ": f"{(df_all_reviewed['Final Status'] == 'REVIEW').sum()/len(df_all_reviewed)*100:.1f}%"},
        {"Chỉ số": "--- Cảm xúc cho các dòng KEEP ---", "Số lượng": "", "Tỷ lệ": ""},
        {"Chỉ số": "+ Positive", "Số lượng": (df_all_reviewed["Final Sentiment"] == "Positive").sum(), "Tỷ lệ": f"{(df_all_reviewed['Final Sentiment'] == 'Positive').sum()/max(1, (df_all_reviewed['Final Status'] == 'KEEP').sum())*100:.1f}%"},
        {"Chỉ số": "+ Neutral", "Số lượng": (df_all_reviewed["Final Sentiment"] == "Neutral").sum(), "Tỷ lệ": f"{(df_all_reviewed['Final Sentiment'] == 'Neutral').sum()/max(1, (df_all_reviewed['Final Status'] == 'KEEP').sum())*100:.1f}%"},
        {"Chỉ số": "+ Negative", "Số lượng": (df_all_reviewed["Final Sentiment"] == "Negative").sum(), "Tỷ lệ": f"{(df_all_reviewed['Final Sentiment'] == 'Negative').sum()/max(1, (df_all_reviewed['Final Status'] == 'KEEP').sum())*100:.1f}%"},
        {"Chỉ số": "--- Biến động so với nhãn cũ ---", "Số lượng": "", "Tỷ lệ": ""},
        {"Chỉ số": "Tổng số dòng đã được điều chỉnh nhãn", "Số lượng": len(df_change_log), "Tỷ lệ": f"{len(df_change_log)/len(df_all_reviewed)*100:.1f}%"},
        {"Chỉ số": "+ Sửa KEEP -> REMOVE (loại bỏ ngữ cảnh rác)", "Số lượng": sum(1 for r in change_log_rows if r["Change Type"] == "KEEP -> REMOVE"), "Tỷ lệ": ""},
        {"Chỉ số": "+ Sửa REMOVE -> KEEP (cứu bình luận bị loại nhầm)", "Số lượng": sum(1 for r in change_log_rows if r["Change Type"] == "REMOVE -> KEEP"), "Tỷ lệ": ""},
        {"Chỉ số": "+ Chuyển sang REVIEW", "Số lượng": sum(1 for r in change_log_rows if "REVIEW" in r["New Status"]), "Tỷ lệ": ""},
        {"Chỉ số": "+ Đổi nhãn cảm xúc (Sentiment Change)", "Số lượng": sum(1 for r in change_log_rows if "Sentiment:" in r["Change Type"]), "Tỷ lệ": ""},
    ]
    df_summary = pd.DataFrame(summary_data)

    # Styling Excel bằng openpyxl
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    border_thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sheets_data = {
            "ALL_REVIEWED": df_all_reviewed,
            "HUMAN_REVIEW": df_human_review,
            "CHANGE_LOG": df_change_log,
            "SUMMARY": df_summary,
        }

        for sheet_name, sheet_df in sheets_data.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for col_idx in range(1, len(sheet_df.columns) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

            for col_cells in ws.columns:
                col_letter = col_cells[0].column_letter
                col_name = str(col_cells[0].value or "")
                if col_name in ["Nội dung RAW", "Reason", "Chỉ số"]:
                    ws.column_dimensions[col_letter].width = 50
                elif col_name in ["Row ID"]:
                    ws.column_dimensions[col_letter].width = 12
                else:
                    max_len = max(len(str(c.value or "")) for c in col_cells[:100])
                    ws.column_dimensions[col_letter].width = min(max(max_len + 3, 14), 30)

                for c in col_cells:
                    c.border = border_thin
                    if c.row > 1 and col_name in ["Row ID", "Loại", "Final Status", "Final Sentiment", "Old Status", "New Status", "Old Sentiment", "New Sentiment", "Confidence", "Change Type", "Số lượng", "Tỷ lệ"]:
                        c.alignment = Alignment(horizontal="center", vertical="center")

    print(f"Đã xuất thành công file Excel: {output_path}")
    return df_all_reviewed, df_change_log, df_human_review, df_summary


# ==============================================================================
# 6. Main
# ==============================================================================
def main():
    print("=" * 80)
    print("CHẠY RÀ SOÁT VÀ HIỆU CHỈNH TOÀN BỘ 3.776 BÌNH LUẬN (BATCH SIZE = 30)")
    print("=" * 80)

    if not INPUT_FILE.exists():
        print(f"LỖI: Không tìm thấy file {INPUT_FILE}")
        return 1

    df_full = pd.read_excel(INPUT_FILE, sheet_name="ALL_ANNOTATED")
    df_full["row_id"] = df_full.index + 2
    total_rows = len(df_full)
    print(f"Đã đọc file gốc: {INPUT_FILE} ({total_rows:,} dòng).")

    # Khởi tạo KeyManager
    raw_keys_env = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
    key_list = [k.strip() for k in raw_keys_env.replace("\n", ",").split(",") if k.strip()]
    if not key_list:
        print("LỖI: Không tìm thấy API Key!")
        return 1

    key_manager = KeyManager(key_list)
    model_name = "gemini-flash-lite-latest"
    print(f"Mô hình sử dụng: {model_name} | Batch size: {BATCH_SIZE} | Tổng số key: {len(key_manager.keys)}")

    # Nạp checkpoint
    review_dict: Dict[int, Dict[str, Any]] = {}
    if CHECKPOINT_FULL.exists():
        try:
            with open(CHECKPOINT_FULL, "r", encoding="utf-8") as f:
                saved = json.load(f)
                review_dict = {int(k): v for k, v in saved.items()}
            print(f"Tìm thấy checkpoint full: đã xử lý trước đó {len(review_dict):,}/{total_rows:,} dòng.")
        except Exception as e:
            print(f"Lỗi đọc checkpoint full: {e}")
    elif CHECKPOINT_SAMPLE.exists():
        try:
            with open(CHECKPOINT_SAMPLE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                review_dict = {int(k): v for k, v in saved.items()}
            print(f"Kế thừa từ checkpoint mẫu 100 dòng: {len(review_dict)} dòng.")
        except Exception as e:
            print(f"Lỗi đọc checkpoint mẫu: {e}")

    # Xử lý trước các dòng trống nội dung -> REMOVE
    empty_count = 0
    for _, row in df_full.iterrows():
        r_id = int(row["row_id"])
        text = str(row.get("Nội dung", "") or "").strip()
        if not text and r_id not in review_dict:
            review_dict[r_id] = {
                "row_id": r_id,
                "filter_status": "REMOVE",
                "sentiment": None,
                "confidence_level": "HIGH",
                "reason": "Bình luận trống hoặc chỉ có sticker/hình ảnh",
            }
            empty_count += 1
    if empty_count > 0:
        print(f"Đã tự động xác định {empty_count} bình luận trống -> REMOVE.")

    # Chuẩn bị danh sách cần gọi API
    items_list = df_full.to_dict(orient="records")
    items_to_process = [item for item in items_list if int(item["row_id"]) not in review_dict]
    print(f"Số lượng bình luận còn lại cần gọi API: {len(items_to_process):,} / {total_rows:,}")

    total_api_calls = 0
    total_tokens = 0
    start_time = time.time()

    total_batches = (len(items_to_process) + BATCH_SIZE - 1) // BATCH_SIZE if items_to_process else 0

    for b_idx in range(0, len(items_to_process), BATCH_SIZE):
        batch = items_to_process[b_idx : b_idx + BATCH_SIZE]
        batch_num = b_idx // BATCH_SIZE + 1
        t0 = time.time()
        print(f"\n[Batch {batch_num}/{total_batches}] Rà soát {len(batch)} dòng (Row {batch[0]['row_id']} -> {batch[-1]['row_id']})...", end="", flush=True)

        annotations, tokens = review_batch_with_retry(key_manager, model_name, batch)
        if annotations is None:
            print(f"\n[CẢNH BÁO] Không thể hoàn thành batch {batch_num}. Lưu checkpoint hiện tại và tạm dừng.")
            break

        total_api_calls += 1
        b_tok = tokens.get("total_tokens", 0)
        total_tokens += b_tok

        for ann in annotations:
            review_dict[ann.row_id] = ann.model_dump()

        # Lưu checkpoint sau mỗi batch
        with open(CHECKPOINT_FULL, "w", encoding="utf-8") as f:
            json.dump(review_dict, f, ensure_ascii=False, indent=2)

        t1 = time.time()
        print(f" Xong ({t1 - t0:.2f}s, {b_tok} tokens) -> Tiến độ: {len(review_dict)}/{total_rows} ({len(review_dict)/total_rows*100:.1f}%)")
        time.sleep(1.5)

    elapsed = time.time() - start_time
    print(f"\nHoàn thành rà soát {len(review_dict)}/{total_rows} dòng trong {elapsed:.2f}s. Tổng token: {total_tokens:,}")

    # Xuất file Excel
    df_all_reviewed, df_change_log, df_human_review, df_summary = export_reviewed_excel(df_full, review_dict, OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("BÁO CÁO TỔNG KẾT RÀ SOÁT TOÀN BỘ DATASET:")
    print("=" * 80)
    print(f"1. Tổng số dòng trong dataset: {total_rows:,} bình luận")
    print(f"2. Đã rà soát thực tế: {len(review_dict):,} ({len(review_dict)/total_rows*100:.1f}%)")
    print(f"3. Tổng số dòng được điều chỉnh nhãn: {len(df_change_log):,} / {len(review_dict):,} ({len(df_change_log)/len(review_dict)*100:.1f}%)")
    print(f"4. Số dòng cần kiểm tra thủ công (HUMAN_REVIEW): {len(df_human_review):,}")
    print(f"5. File kết quả Excel hoàn chỉnh: {OUTPUT_FILE}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

