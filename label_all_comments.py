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

RAW_DATA_PATH = ROOT_DIR / "simple_post_comments_merged.xlsx"
OUTPUT_PATH = ROOT_DIR / "all_comments_annotated.xlsx"
CHECKPOINT_PATH = ROOT_DIR / "checkpoint_all_annotations.json"
TARGET_MOVIE = "Trại buôn người"
BATCH_SIZE = 30


# ==========================================
# 1. Pydantic Schema cho Structured Output
# ==========================================
class CommentAnnotation(BaseModel):
    row_uid: str = Field(description="Mã định danh duy nhất của dòng dữ liệu")
    filter_status: Literal["KEEP", "REMOVE", "REVIEW"] = Field(
        description="KEEP: liên quan phim; REMOVE: rác/spam/tag/không liên quan; REVIEW: mơ hồ, thiếu ngữ cảnh"
    )
    sentiment: Optional[Literal["Positive", "Negative", "Neutral"]] = Field(
        default=None,
        description="Chỉ gán nếu filter_status là KEEP. Nếu REMOVE hoặc REVIEW thì bắt buộc là null",
    )
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Độ tin cậy chủ quan của mô hình đối với quyết định"
    )
    review_required: bool = Field(
        description="True nếu confidence là MEDIUM/LOW, câu khó, hỗn hợp hoặc thiếu ngữ cảnh"
    )
    reason: str = Field(
        description="Giải thích ngắn gọn bằng tiếng Việt (1 câu) về quyết định"
    )


class BatchAnnotationResponse(BaseModel):
    annotations: List[CommentAnnotation]


# ==========================================
# 2. Xây dựng bản đồ bình luận cha (Parent Map)
# ==========================================
def build_parent_map(simple_post_dir: Path) -> Dict[str, Dict[str, str]]:
    """Duyệt qua các file JSON gốc để lấy parent comment ID và parent comment text cho các Reply."""
    parent_map = {}
    if not simple_post_dir.exists():
        return parent_map

    for json_file in simple_post_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            posts = data if isinstance(data, list) else [data]
            for post in posts:
                if not isinstance(post, dict):
                    continue
                for c in post.get("comments", []):
                    if not isinstance(c, dict):
                        continue
                    cid = str(c.get("comment_id", "")).strip()
                    ctext = str(c.get("text", "") or "").strip()
                    for r in c.get("replies", []):
                        if not isinstance(r, dict):
                            continue
                        rid = str(r.get("comment_id", "")).strip()
                        if rid:
                            parent_map[rid] = {
                                "parent_comment_id": cid,
                                "parent_comment_text": ctext,
                            }
        except Exception as e:
            print(f"Warning: Không đọc được {json_file.name}: {e}")
    return parent_map


# ==========================================
# 3. Prompt cho Batch 30 Bình Luận
# ==========================================
def build_batch_prompt(batch_items: List[Dict[str, Any]]) -> str:
    prompt = f"""Bạn là chuyên gia NLP gán nhãn dữ liệu cảm xúc tiếng Việt cho bài toán phân loại bình luận Facebook về bộ phim "{TARGET_MOVIE}".

Nhiệm vụ của bạn là thực hiện 2 bước gán nhãn tuần tự cho danh sách bình luận dưới đây:

### NHIỆM VỤ 1: SÀNG LỌC BÌNH LUẬN (filter_status)
- KEEP: Bình luận liên quan đến phim "{TARGET_MOVIE}", bao gồm:
  + Đánh giá, khen chê nội dung, diễn xuất của diễn viên, kịch bản, âm thanh, bối cảnh, cái kết.
  + Cảm nhận, xúc cảm chân thật của người xem (hồi hộp, sợ, khóc, ám ảnh, xúc động, thất vọng).
  + Câu hỏi thông tin về phim (chiếu rạp nào, khi nào chiếu, diễn viên nào đóng, xem ở đâu).
  + So sánh với phim khác nhưng vẫn có nhận xét/đánh giá về phim "{TARGET_MOVIE}".
- REMOVE: Bình luận không phù hợp để huấn luyện mô hình cảm xúc phim:
  + Spam, link bán hàng, quảng cáo, cờ bạc, kiếm tiền online.
  + Chỉ tag tên người khác (@Tên) mà không có bất kỳ nội dung nhận xét nào về phim.
  + Nội dung rỗng, icon/emoji vô nghĩa hoặc ký tự linh tinh không thể hiện ý kiến.
  + Trò chuyện cá nhân hoàn toàn riêng tư giữa người dùng, không dính líu đến phim.
  + Bình luận chỉ bàn về phim khác, diễn viên khác ngoài đời mà không liên quan đến phim "{TARGET_MOVIE}".
  * CHÚ Ý: KHÔNG ĐƯỢC coi bình luận không liên quan là Neutral! Bình luận không liên quan BẮT BUỘC phải là REMOVE.
- REVIEW: Bình luận chưa thể xác định chắc chắn:
  + Không rõ đang nói về bộ phim hay đang nói về việc gì khác.
  + Bình luận Reply quá ngắn, tối nghĩa hoặc thiếu ngữ cảnh dù đã tham khảo bình luận cha.
  + Ngữ nghĩa quá mơ hồ, đa nghĩa, không đủ dữ liệu để quyết định KEEP hay REMOVE.

### NHIỆM VỤ 2: GÁN NHÃN CẢM XÚC (sentiment)
* ĐIỀU KIỆN TIÊN QUYẾT: CHỈ gán sentiment cho các dòng có filter_status là KEEP.
* NẾU filter_status là REMOVE hoặc REVIEW thì sentiment BẮT BUỘC PHẢI LÀ null!
- Positive: Khen ngợi phim, diễn xuất hay, phim cuốn/đỉnh, khuyến khích xem, xúc động tích cực.
- Negative: Chê bai phim dở, kịch bản tệ, diễn đơ, phí tiền vé, thất vọng, khuyên không nên xem.
- Neutral: Liên quan đến phim nhưng không khen không chê, thái độ trung tính, hỏi thông tin khách quan (chiếu mấy giờ, ở rạp nào, bao nhiêu phút).

### NGUYÊN TẮC TIẾNG VIỆT:
1. Hiểu tiếng lóng mạng, viết tắt: "đỉnh", "cuốn", "bánh cuốn", "cháy" -> tích cực; "dở tệ", "flop", "chán òm", "cấn", "ảo ma", "nhảm" -> tiêu cực; "cmn", "vcl", "vl" là từ cảm thán nhấn mạnh cường độ.
2. Phân biệt cảm xúc cốt truyện với thái độ đánh giá:
   - "Phim xem ám ảnh và sợ thót tim" -> Positive (khen phim kinh dị/kịch tính tạo hiệu ứng tốt).
   - "Xem mà khóc ướt khăn giấy" -> Positive (khen phim chạm cảm xúc).
   - Không mặc định từ "khóc", "buồn", "sợ" là Negative.
3. Chú ý phủ định & mỉa mai (sarcasm):
   - "Không chê vào đâu được" -> Positive.
   - "Phim hay đến mức ngủ quên 3 lần" -> Negative.
4. Nếu bình luận vừa khen vừa chê cân bằng, không có thái độ nổi trội -> chuyển filter_status thành REVIEW và sentiment = null.

### ĐỘ TIN CẬY & RÀ SOÁT:
- confidence_level: "HIGH" (rõ ràng), "MEDIUM" (có dấu hiệu mơ hồ), "LOW" (không chắc chắn).
- review_required: true nếu confidence_level là MEDIUM/LOW hoặc ý kiến hỗn hợp/câu khó; ngược lại false.
- reason: 1 câu tiếng Việt giải thích ngắn gọn lý do.

---
DANH SÁCH {len(batch_items)} BÌNH LUẬN CẦN GÁN NHÃN:
"""
    for item in batch_items:
        uid = item["row_uid"]
        text = item["Nội dung"]
        c_type = item["Loại"]
        parent_text = item.get("parent_comment_text", "")
        prompt += f"\n- [ID: {uid}] | Loại: {c_type}\n"
        if c_type == "Reply" and parent_text:
            prompt += f"  (Ngữ cảnh bình luận cha: \"{parent_text}\")\n"
        prompt += f"  Nội dung: \"{text}\"\n"

    prompt += "\nTrả về danh sách đối tượng json tuân thủ schema BatchAnnotationResponse đã định nghĩa."
    return prompt


# ==========================================
# 4. Quản lý Pool API Key & Tự Động Đảo Key
# ==========================================
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
        print(f" -> Đang kích hoạt API Key [{self.current_idx + 1}/{len(self.keys)}]: {masked}")

    @property
    def client(self):
        return self._client

    def rotate_key(self) -> bool:
        """Đánh dấu key hiện tại hết hạn ngạch và chuyển sang key tiếp theo."""
        self.exhausted_keys.add(self.current_idx)
        print(f"\n[QUOTA EXHAUSTED] Key [{self.current_idx + 1}/{len(self.keys)}] đã hết hạn ngạch hôm nay!")
        for i in range(len(self.keys)):
            if i not in self.exhausted_keys:
                self.current_idx = i
                self._init_client()
                return True
        print("\n[TẤT CẢ KEY ĐÃ HẾT QUOTA] Toàn bộ danh sách API key đều đã chạm giới hạn!")
        return False


def annotate_batch_with_retry(
    key_manager: KeyManager,
    model_name: str,
    batch_items: List[Dict[str, Any]],
    max_retries: int = 5,
):
    from google.genai import types

    prompt = build_batch_prompt(batch_items)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BatchAnnotationResponse,
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
            batch_res = BatchAnnotationResponse.model_validate(res_dict)

            # Kiểm tra đủ ID
            returned_uids = {item.row_uid for item in batch_res.annotations}
            expected_uids = {item["row_uid"] for item in batch_items}
            if not expected_uids.issubset(returned_uids):
                missing = expected_uids - returned_uids
                print(f"Warning: Batch thiếu ID: {missing}. Retry lần {attempt}...")
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
                    print("-> Đã chuyển key mới thành công, thử lại batch ngay lập tức...")
                    attempt = 0  # reset attempt cho key mới
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


# ==========================================
# 5. Xuất Excel Theo Đúng Yêu Cầu Của Người Dùng
# ==========================================
def export_excel(df_all_rows: pd.DataFrame, annotated_dict: Dict[str, Dict[str, Any]], output_path: Path):
    print("\n--- Đang tạo và định dạng file Excel đầu ra ---")

    # Ghép thông tin gán nhãn vào dataframe gốc
    records = []
    for _, row in df_all_rows.iterrows():
        uid = row["row_uid"]
        ann = annotated_dict.get(uid, {})
        records.append({
            "row_uid": uid,
            "Tên tác giả": row.get("Tên tác giả", ""),
            "Nội dung": row.get("Nội dung", ""),
            "Loại": row.get("Loại", ""),
            "filter_status": ann.get("filter_status", ""),
            "sentiment": ann.get("sentiment", "") if ann.get("filter_status") == "KEEP" else "",
            "confidence_level": ann.get("confidence_level", ""),
            "review_required": ann.get("review_required", ""),
            "reason": ann.get("reason", ""),
        })

    df_full = pd.DataFrame(records)

    # Các cột xuất ra theo yêu cầu chính xác của người dùng:
    # tên tác giả, nội dung, loại(comment, reply), filter status, sentiment, confident level, review_required và reason
    standard_columns = [
        "Tên tác giả",
        "Nội dung",
        "Loại",
        "filter_status",
        "sentiment",
        "confidence_level",
        "review_required",
        "reason",
    ]

    # Sheet 1: ALL_ANNOTATED
    df_all_annotated = df_full[standard_columns].copy()

    # Sheet 2: KEEP
    df_keep = df_full[df_full["filter_status"] == "KEEP"][standard_columns].copy()

    # Sheet 3: REMOVE
    df_remove = df_full[df_full["filter_status"] == "REMOVE"][standard_columns].copy()

    # Sheet 4: HUMAN_REVIEW (có thêm cột human_label)
    df_human_review = df_full[
        (df_full["filter_status"] == "REVIEW") | (df_full["review_required"] == True)
    ][standard_columns].copy()
    df_human_review["human_label"] = ""

    review_columns = standard_columns + ["human_label"]
    df_human_review = df_human_review[review_columns]

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
            "ALL_ANNOTATED": df_all_annotated,
            "KEEP": df_keep,
            "REMOVE": df_remove,
            "HUMAN_REVIEW": df_human_review,
        }

        for sheet_name, sheet_df in sheets_data.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            # Format header
            for col_idx in range(1, len(sheet_df.columns) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

            # Column dimensions
            for col_cells in ws.columns:
                col_letter = col_cells[0].column_letter
                col_name = str(col_cells[0].value or "")
                if col_name in ["Nội dung", "reason"]:
                    ws.column_dimensions[col_letter].width = 50
                elif col_name in ["Tên tác giả"]:
                    ws.column_dimensions[col_letter].width = 22
                else:
                    max_len = max(len(str(c.value or "")) for c in col_cells[:100])
                    ws.column_dimensions[col_letter].width = min(max(max_len + 3, 14), 30)

                for c in col_cells:
                    c.border = border_thin
                    if c.row > 1 and col_name in ["Loại", "filter_status", "sentiment", "confidence_level", "review_required", "human_label"]:
                        c.alignment = Alignment(horizontal="center", vertical="center")

    print(f"Đã xuất thành công file Excel: {output_path}")


# ==========================================
# 6. Pipeline Chính
# ==========================================
def main():
    print("=" * 75)
    print("CHẠY GÁN NHÃN TOÀN BỘ DỮ LIỆU BÌNH LUẬN VỚI GEMINI API (BATCH SIZE = 30)")
    print("=" * 75)

    if not RAW_DATA_PATH.exists():
        print(f"Lỗi: Không tìm thấy file RAW tại {RAW_DATA_PATH}")
        return 1

    df_raw = pd.read_excel(RAW_DATA_PATH, sheet_name=0)
    total_rows = len(df_raw)
    print(f"Tổng số dòng dữ liệu: {total_rows:,} bình luận.")

    # Ánh xạ parent context
    simple_post_dir = ROOT_DIR / "data" / "simple_post"
    parent_map = build_parent_map(simple_post_dir)
    print(f"Đã ánh xạ được {len(parent_map)} replies với bình luận cha.")

    df_raw["row_uid"] = [f"ROW_{i+1:04d}" for i in range(total_rows)]
    df_raw["Nội dung"] = df_raw["Nội dung"].fillna("").astype(str)
    df_raw["parent_comment_text"] = df_raw["Comment ID"].astype(str).map(
        lambda cid: parent_map.get(cid, {}).get("parent_comment_text", "")
    )

    # Đọc checkpoint nếu có
    annotated_dict: Dict[str, Dict[str, Any]] = {}
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                annotated_dict = json.load(f)
            print(f"Tìm thấy checkpoint: đã xử lý trước đó {len(annotated_dict):,}/{total_rows:,} dòng.")
        except Exception as e:
            print(f"Lỗi đọc checkpoint: {e}")

    # Xử lý trước các dòng trống nội dung (không cần tốn API token)
    empty_count = 0
    for _, row in df_raw.iterrows():
        uid = row["row_uid"]
        text = row["Nội dung"].strip()
        if not text and uid not in annotated_dict:
            annotated_dict[uid] = {
                "row_uid": uid,
                "filter_status": "REMOVE",
                "sentiment": None,
                "confidence_level": "HIGH",
                "review_required": False,
                "reason": "Bình luận trống hoặc chỉ có sticker/hình ảnh",
            }
            empty_count += 1
    if empty_count > 0:
        print(f"Đã tự động xác định {empty_count} bình luận trống -> REMOVE.")

    # Kiểm tra danh sách API Keys
    raw_keys_env = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
    key_list = [k.strip() for k in raw_keys_env.replace("\n", ",").split(",") if k.strip()]
    if not key_list:
        print("LỖI: Chưa có biến môi trường GEMINI_API_KEYS hoặc GEMINI_API_KEY trong .env")
        return 1

    key_manager = KeyManager(key_list)
    model_name = "gemini-flash-lite-latest"
    print(f"Sử dụng mô hình: {model_name} | Batch size: {BATCH_SIZE} | Tổng số key cấu hình: {len(key_manager.keys)}")

    # Chuẩn bị danh sách cần gọi API
    items_list = df_raw.to_dict(orient="records")
    items_to_process = [item for item in items_list if item["row_uid"] not in annotated_dict]
    print(f"Số lượng bình luận còn lại cần gọi API: {len(items_to_process):,} / {total_rows:,}")

    total_api_calls = 0
    total_tokens = 0
    prompt_tokens = 0
    candidates_tokens = 0
    start_time = time.time()

    total_batches = (len(items_to_process) + BATCH_SIZE - 1) // BATCH_SIZE if items_to_process else 0

    for b_idx in range(0, len(items_to_process), BATCH_SIZE):
        batch = items_to_process[b_idx : b_idx + BATCH_SIZE]
        batch_num = b_idx // BATCH_SIZE + 1
        t_batch_start = time.time()

        print(f"\n[Batch {batch_num}/{total_batches}] Đang gửi {len(batch)} bình luận ({batch[0]['row_uid']} -> {batch[-1]['row_uid']})...", end="", flush=True)

        annotations, tokens = annotate_batch_with_retry(key_manager, model_name, batch)

        if annotations is None:
            print(f"\n[CẢNH BÁO] Không thể xử lý Batch {batch_num}. Lưu checkpoint hiện tại và tạm dừng.")
            break

        total_api_calls += 1
        b_tok = tokens.get("total_tokens", 0)
        total_tokens += b_tok
        prompt_tokens += tokens.get("prompt_tokens", 0)
        candidates_tokens += tokens.get("candidates_tokens", 0)

        for ann in annotations:
            if ann.filter_status in ["REMOVE", "REVIEW"]:
                ann.sentiment = None
            annotated_dict[ann.row_uid] = ann.model_dump()

        # Lưu checkpoint sau mỗi batch
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(annotated_dict, f, ensure_ascii=False, indent=2)

        t_batch_end = time.time()
        print(f" Xong ({t_batch_end - t_batch_start:.2f}s, {b_tok} tokens) -> Tổng tiến độ: {len(annotated_dict)}/{total_rows} ({len(annotated_dict)/total_rows*100:.1f}%)")

        # Giãn cách 1.5s để đảm bảo an toàn rate limit
        time.sleep(1.5)

    elapsed = time.time() - start_time
    print(f"\nHoàn thành xử lý {len(annotated_dict)}/{total_rows} bình luận trong {elapsed:.2f}s.")
    print(f"Tổng token sử dụng: {total_tokens:,} (Prompt: {prompt_tokens:,} | Output: {candidates_tokens:,})")

    # Xuất file Excel
    export_excel(df_raw, annotated_dict, OUTPUT_PATH)

    # Thống kê tổng hợp
    filter_counts = {}
    sentiment_counts = {}
    confidence_counts = {}
    review_required_count = 0

    for ann in annotated_dict.values():
        fs = ann.get("filter_status", "UNKNOWN")
        filter_counts[fs] = filter_counts.get(fs, 0) + 1

        sent = ann.get("sentiment")
        if sent:
            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

        conf = ann.get("confidence_level", "UNKNOWN")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        if ann.get("review_required"):
            review_required_count += 1

    print("\n" + "=" * 75)
    print("BÁO CÁO TỔNG KẾT TOÀN BỘ TẬP DỮ LIỆU:")
    print("=" * 75)
    print(f"1. Tổng số dòng dữ liệu: {total_rows:,} bình luận")
    print(f"2. Đã gán nhãn thành công: {len(annotated_dict):,} ({len(annotated_dict)/total_rows*100:.1f}%)")
    print("\n3. Sàng lọc (filter_status):")
    for k, v in filter_counts.items():
        print(f"   - {k:8s}: {v:5d} ({v/len(annotated_dict)*100:5.1f}%)")
    print("\n4. Cảm xúc (sentiment - cho các dòng KEEP):")
    total_keep = filter_counts.get("KEEP", 0)
    for k, v in sentiment_counts.items():
        print(f"   - {k:8s}: {v:5d} ({v/total_keep*100:5.1f}%)")
    print("\n5. Độ tin cậy (confidence_level):")
    for k, v in confidence_counts.items():
        print(f"   - {k:8s}: {v:5d} ({v/len(annotated_dict)*100:5.1f}%)")
    print(f"\n6. Số bình luận cần con người kiểm tra (HUMAN_REVIEW): {review_required_count:,}")
    print(f"7. Đường dẫn file Excel kết quả: {OUTPUT_PATH}")
    print("=" * 75 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

