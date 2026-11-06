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

# Load biến môi trường từ .env nếu có
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

RAW_DATA_PATH = ROOT_DIR / "data" / "facebook" / "raw" / "simple_post_comments_merged.xlsx"
OUTPUT_DIR = ROOT_DIR / "data" / "facebook" / "intermediate" / "output_llm_test"
SAMPLE_RAW_PATH = OUTPUT_DIR / "sample_100_raw.xlsx"
SAMPLE_ANNOTATED_PATH = OUTPUT_DIR / "sample_100_annotated.xlsx"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_annotations.json"
TARGET_MOVIE = "Trại buôn người"


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
    context_dependent: bool = Field(
        description="True nếu bình luận phụ thuộc ngữ cảnh cha/bài đăng để hiểu đúng"
    )
    reason_code: str = Field(
        description="Mã lý do viết hoa, ví dụ: PRAISE_ACTING, CRITIQUE_PLOT, TAG_ONLY, UNRELATED_CHAT, LACK_CONTEXT, etc."
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
# 3. Đọc dữ liệu RAW và lấy mẫu 100 bình luận
# ==========================================
def inspect_and_sample_data(raw_path: Path, output_dir: Path) -> pd.DataFrame:
    print("\n--- [BƯỚC 1: KIỂM TRA DỮ LIỆU RAW VÀ LẤY MẪU] ---")
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file RAW tại: {raw_path}")

    print(f"Đường dẫn tuyệt đối file RAW: {raw_path}")
    excel_file = pd.ExcelFile(raw_path)
    print(f"Danh sách sheets: {excel_file.sheet_names}")

    df_raw = pd.read_excel(raw_path, sheet_name=0)
    print(f"Tổng số dòng trong RAW: {len(df_raw)}")
    print(f"Danh sách cột: {df_raw.columns.tolist()}")

    # Kiểm tra Comment ID, Post ID, Loại
    print(f"Số lượng Post ID duy nhất: {df_raw['Post ID'].nunique()}")
    print(f"Phân bố loại bình luận:\n{df_raw['Loại'].value_counts().to_dict()}")
    print(f"Số Comment ID trùng lặp: {df_raw['Comment ID'].duplicated().sum()}")
    empty_content = (df_raw["Nội dung"].isna() | (df_raw["Nội dung"].astype(str).str.strip() == "")).sum()
    print(f"Số dòng nội dung trống: {empty_content}")

    # Lấy thông tin ngữ cảnh bình luận cha
    simple_post_dir = ROOT_DIR / "data" / "simple_post"
    print(f"Đang ánh xạ ngữ cảnh cha từ các file JSON trong: {simple_post_dir}...")
    parent_map = build_parent_map(simple_post_dir)
    print(f"Đã ánh xạ được {len(parent_map)} replies với bình luận cha.")

    # Lấy mẫu ngẫu nhiên 100 bình luận với random_state = 42
    print("Tiến hành lấy mẫu ngẫu nhiên 100 bình luận (random seed = 42)...")
    sample_df = df_raw.sample(n=100, random_state=42).copy().reset_index(drop=True)

    # Thêm row_uid và ngữ cảnh cha
    sample_df["row_uid"] = [f"UID_{i+1:03d}" for i in range(len(sample_df))]

    # Ánh xạ parent comment
    sample_df["parent_comment_id"] = sample_df["Comment ID"].astype(str).map(
        lambda cid: parent_map.get(cid, {}).get("parent_comment_id", "")
    )
    sample_df["parent_comment_text"] = sample_df["Comment ID"].astype(str).map(
        lambda cid: parent_map.get(cid, {}).get("parent_comment_text", "")
    )

    # Giữ nguyên dữ liệu RAW, điền chuỗi rỗng nếu NaN
    sample_df["Nội dung"] = sample_df["Nội dung"].fillna("").astype(str)

    # Thống kê mẫu
    print(f"Mẫu đã lấy: {len(sample_df)} dòng.")
    print(f"Phân bố loại trong mẫu: {sample_df['Loại'].value_counts().to_dict()}")
    print(f"Số lượng Post ID đại diện trong mẫu: {sample_df['Post ID'].nunique()}")

    # Lưu sample_100_raw.xlsx
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_sample_cols = [
        "row_uid",
        "Post ID",
        "Comment ID",
        "Loại",
        "Nội dung",
        "parent_comment_id",
        "parent_comment_text",
    ]
    sample_df[raw_sample_cols].to_excel(SAMPLE_RAW_PATH, index=False)
    # Lưu một bản tại root để người dùng tiện tra cứu
    sample_df[raw_sample_cols].to_excel(ROOT_DIR / "sample_100_raw.xlsx", index=False)
    print(f"Đã lưu mẫu thô tại: {SAMPLE_RAW_PATH}")

    return sample_df


# ==========================================
# 4. Kiểm tra cấu hình Gemini API & Model
# ==========================================
def setup_gemini_client():
    print("\n--- [BƯỚC 2: CẤU HÌNH VÀ KIỂM TRA GEMINI API] ---")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n" + "!" * 70)
        print("LỖI: Chưa tìm thấy biến môi trường 'GEMINI_API_KEY'!")
        print("HƯỚNG DẪN CẤU HÌNH API KEY:")
        print("Cách 1 (Khuyên dùng): Tạo file '.env' tại thư mục:")
        print(f"       {ROOT_DIR / '.env'}")
        print("       Nội dung file: GEMINI_API_KEY=AIzaSy...")
        print("Cách 2: Đặt biến môi trường trong PowerShell trước khi chạy:")
        print('       $env:GEMINI_API_KEY = "AIzaSy..."')
        print("!" * 70 + "\n")
        return None, None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        print("Đã khởi tạo Gemini Client thành công.")

        # Kiểm tra danh sách mô hình khả dụng để chọn Flash model tối ưu
        print("Đang kiểm tra các model Flash khả dụng...")
        candidate_models = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        selected_model = None

        try:
            available_models = [m.name for m in client.models.list()]
            available_short = [m.split("/")[-1] for m in available_models]
            for cand in candidate_models:
                if cand in available_short or f"models/{cand}" in available_models:
                    selected_model = cand
                    break
        except Exception as e:
            print(f"Lưu ý: Không thể list models ({e}), thử dùng mặc định 'gemini-2.5-flash-lite'...")
            selected_model = "gemini-2.5-flash-lite"

        if not selected_model:
            selected_model = "gemini-2.5-flash-lite"

        print(f"Mô hình được chọn: {selected_model}")
        return client, selected_model
    except Exception as e:
        print(f"Lỗi khi kết nối Gemini API: {e}")
        return None, None


# ==========================================
# 5. Xây dựng Prompt cho Batch Annotation
# ==========================================
def build_batch_prompt(batch_items: List[Dict[str, Any]]) -> str:
    prompt = f"""Bạn là chuyên gia NLP gán nhãn dữ liệu cảm xúc tiếng Việt cho bài toán phân loại bình luận mạng xã hội Facebook về bộ phim "{TARGET_MOVIE}".

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
  * CHÚ Ý QUAN TRỌNG: KHÔNG ĐƯỢC coi bình luận không liên quan là Neutral! Bình luận không liên quan BẮT BUỘC phải là REMOVE.
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

### NGUYÊN TẮC ĐÁNH GIÁ NGÔN NGỮ TIẾNG VIỆT:
1. Hiểu tiếng lóng mạng, viết tắt, teencode: "đỉnh", "cuốn", "bánh cuốn", "cháy", "đỉnk" -> tích cực; "dở tệ", "flop", "chán òm", "cấn", "ảo ma", "nhảm" -> tiêu cực; "cmn", "vcl", "vl" là từ cảm thán nhấn mạnh cường độ.
2. Phân biệt cảm xúc trong cốt truyện phim với thái độ đánh giá:
   - "Phim xem ám ảnh và sợ thót tim" -> Positive (khen phim kinh dị/kịch tính tạo hiệu ứng tốt).
   - "Xem mà khóc ướt cả khăn giấy" -> Positive (khen phim chạm cảm xúc).
   - Không mặc định từ "khóc", "buồn", "sợ", "ác" là Negative!
3. Chú ý câu phủ định và mỉa mai (sarcasm):
   - "Không chê vào đâu được" -> Positive.
   - "Phim hay đến mức ngủ quên 3 lần" -> Negative.
4. Nếu bình luận vừa khen vừa chê cân bằng, không có thái độ nổi trội -> chuyển filter_status thành REVIEW và sentiment = null.

### ĐỘ TIN CẬY & RÀ SOÁT:
- confidence_level: "HIGH" (rất rõ ràng), "MEDIUM" (có dấu hiệu mơ hồ), "LOW" (không chắc chắn, thiếu dữ liệu).
- review_required: true nếu confidence_level là MEDIUM/LOW hoặc ý kiến hỗn hợp; ngược lại false.
- context_dependent: true nếu phụ thuộc vào bình luận cha/bài đăng mới hiểu được; ngược lại false.
- reason_code: Mã ngắn gọn bằng tiếng Anh viết hoa (vd: PRAISE_ACTING, CRITIQUE_PLOT, RECOMMEND_WATCH, INQUIRY_INFO, TAG_ONLY, SPAM_AD, UNRELATED_CHAT, LACK_CONTEXT, MIXED_OPINION).
- reason: 1 câu tiếng Việt giải thích ngắn gọn lý do đưa ra nhãn.

---
DANH SÁCH {len(batch_items)} BÌNH LUẬN CẦN GÁN NHÃN (Dữ liệu đầu vào):
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
# 6. Gọi Gemini API với Retry và Checkpoint
# ==========================================
def annotate_batch_with_retry(
    client,
    model_name: str,
    batch_items: List[Dict[str, Any]],
    max_retries: int = 4,
) -> Optional[List[CommentAnnotation]]:
    from google.genai import types

    prompt = build_batch_prompt(batch_items)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BatchAnnotationResponse,
        temperature=0.1,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            # Parse JSON từ response.text
            res_dict = json.loads(response.text)
            batch_res = BatchAnnotationResponse.model_validate(res_dict)

            # Kiểm tra đủ số lượng ID
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
            print(f"Lỗi khi gọi API (lần {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                wait_time = 2**attempt
                print(f"Chờ {wait_time}s trước khi retry...")
                time.sleep(wait_time)
            else:
                print(f"Đã thử {max_retries} lần không thành công cho batch này.")
                return None, {}


def run_annotation_pipeline(
    client,
    model_name: str,
    sample_df: pd.DataFrame,
    batch_size: int = 10,
) -> Dict[str, Any]:
    print("\n--- [BƯỚC 3: TIẾN HÀNH GÁN NHÃN 100 BÌNH LUẬN QUA GEMINI API] ---")

    # Đọc checkpoint nếu có
    annotated_dict: Dict[str, Dict[str, Any]] = {}
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                annotated_dict = json.load(f)
            print(f"Tìm thấy checkpoint: đã hoàn thành {len(annotated_dict)}/100 bình luận.")
        except Exception as e:
            print(f"Không thể đọc checkpoint: {e}. Sẽ chạy mới.")

    total_items = len(sample_df)
    items_list = sample_df.to_dict(orient="records")

    total_api_calls = 0
    total_errors = 0
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0
    start_time = time.time()

    for i in range(0, total_items, batch_size):
        batch = items_list[i : i + batch_size]
        batch_uids = [item["row_uid"] for item in batch]

        # Kiểm tra xem batch đã hoàn thành chưa
        if all(uid in annotated_dict for uid in batch_uids):
            print(f"Batch {i//batch_size + 1} ({len(batch)} items: {batch_uids[0]} -> {batch_uids[-1]}) đã có trong checkpoint. Bỏ qua.")
            continue

        print(f"\nĐang xử lý Batch {i//batch_size + 1} (dòng {i+1} đến {min(i+batch_size, total_items)}: {batch_uids[0]} -> {batch_uids[-1]})...")
        total_api_calls += 1
        annotations, tokens = annotate_batch_with_retry(client, model_name, batch)

        if annotations is None:
            total_errors += 1
            print(f"LỖI: Không thể gán nhãn cho Batch {i//batch_size + 1}. Dừng pipeline để bảo toàn trạng thái.")
            break

        total_prompt_tokens += tokens.get("prompt_tokens", 0)
        total_candidates_tokens += tokens.get("candidates_tokens", 0)
        total_tokens += tokens.get("total_tokens", 0)

        # Lưu vào checkpoint
        for ann in annotations:
            # Ràng buộc chặt chẽ: nếu REMOVE hoặc REVIEW thì sentiment phải là None
            if ann.filter_status in ["REMOVE", "REVIEW"]:
                ann.sentiment = None

            annotated_dict[ann.row_uid] = ann.model_dump()

        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(annotated_dict, f, ensure_ascii=False, indent=2)

        print(f"Đã hoàn thành Batch {i//batch_size + 1}. Đã lưu checkpoint ({len(annotated_dict)}/100). (Tokens batch này: {tokens.get('total_tokens', 'N/A')})")
        time.sleep(1)  # Giãn cách để tránh rate limit

    elapsed = time.time() - start_time
    print(f"\nThời gian chạy: {elapsed:.2f}s | Số lượt gọi API: {total_api_calls} | Số lỗi: {total_errors} | Tổng token: {total_tokens}")

    return {
        "annotations": annotated_dict,
        "api_calls": total_api_calls,
        "errors": total_errors,
        "prompt_tokens": total_prompt_tokens,
        "candidates_tokens": total_candidates_tokens,
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
    }


# ==========================================
# 7. Định dạng và Xuất kết quả ra Excel
# ==========================================
def export_annotated_excel(
    sample_df: pd.DataFrame,
    annotated_dict: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    print("\n--- [BƯỚC 4: XUẤT FILE EXCEL KẾT QUẢ VỚI 4 SHEETS] ---")

    # Gộp kết quả LLM vào sample_df
    records = []
    for _, row in sample_df.iterrows():
        uid = row["row_uid"]
        rec = row.to_dict()
        ann = annotated_dict.get(uid, {})
        rec["filter_status"] = ann.get("filter_status", "")
        rec["sentiment"] = ann.get("sentiment", "")
        rec["confidence_level"] = ann.get("confidence_level", "")
        rec["review_required"] = ann.get("review_required", "")
        rec["context_dependent"] = ann.get("context_dependent", "")
        rec["reason_code"] = ann.get("reason_code", "")
        rec["reason"] = ann.get("reason", "")
        # Các cột dành cho kiểm tra thủ công (Mục 9)
        rec["human_decision"] = ""  # KEEP / REMOVE / REVIEW
        rec["human_label"] = ""  # Positive / Negative / Neutral
        rec["final_label"] = ""  # Nhãn thống nhất cuối cùng
        records.append(rec)

    df_all = pd.DataFrame(records)

    # 4 Sheets theo yêu cầu:
    # Sheet 1: ALL_ANNOTATED
    # Sheet 2: KEEP
    # Sheet 3: REMOVE
    # Sheet 4: HUMAN_REVIEW
    df_keep = df_all[df_all["filter_status"] == "KEEP"].copy()
    df_remove = df_all[df_all["filter_status"] == "REMOVE"].copy()

    # Nhóm HUMAN_REVIEW: filter_status == 'REVIEW' HOẶC review_required == True
    # Đồng thời đề xuất thêm 5 mẫu ngẫu nhiên từ HIGH và 5 mẫu từ REMOVE để audit kiểm tra chéo (Mục 7)
    df_review_base = df_all[
        (df_all["filter_status"] == "REVIEW") | (df_all["review_required"] == True)
    ].copy()
    df_review_base["audit_type"] = "Cần rà soát (REVIEW / Confidence thấp)"

    # Spot-check audit sample từ HIGH KEEP
    high_keep = df_all[(df_all["filter_status"] == "KEEP") & (df_all["confidence_level"] == "HIGH")]
    audit_high = high_keep.sample(n=min(5, len(high_keep)), random_state=42).copy()
    audit_high["audit_type"] = "Kiểm tra chéo ngẫu nhiên (HIGH confidence)"

    # Spot-check audit sample từ REMOVE
    audit_remove = df_remove.sample(n=min(5, len(df_remove)), random_state=42).copy()
    audit_remove["audit_type"] = "Kiểm tra chéo ngẫu nhiên (REMOVE)"

    df_human_review = pd.concat([df_review_base, audit_high, audit_remove]).drop_duplicates(subset=["row_uid"]).copy()

    # Style Excel với openpyxl
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
            "ALL_ANNOTATED": df_all,
            "KEEP": df_keep,
            "REMOVE": df_remove,
            "HUMAN_REVIEW": df_human_review,
        }

        for sheet_name, sheet_df in sheets_data.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            # Format headers
            for col_idx in range(1, len(sheet_df.columns) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

            # Auto fit column widths
            for col_cells in ws.columns:
                col_letter = col_cells[0].column_letter
                max_len = max(len(str(c.value or "")) for c in col_cells[:100])
                ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 60)

                # Format borders
                for c in col_cells:
                    c.border = border_thin
                    if c.row > 1 and col_letter in ["A", "B", "C", "D"]:
                        c.alignment = Alignment(horizontal="center", vertical="center")

    print(f"Đã xuất file kết quả tại: {output_path}")


# ==========================================
# 8. Báo cáo thống kê chi tiết
# ==========================================
def print_final_report(
    total_raw_rows: int,
    sample_df: pd.DataFrame,
    annotated_dict: Dict[str, Dict[str, Any]],
    pipeline_meta: Dict[str, Any],
):
    print("\n" + "=" * 70)
    print("BÁO CÁO KẾT QUẢ GÁN NHÃN 100 BÌNH LUẬN BẰNG GEMINI API")
    print("=" * 70)

    total_annotated = len(annotated_dict)
    print(f"1. Tổng số dòng dữ liệu RAW trong file: {total_raw_rows:,} dòng")
    print(f"2. Số bình luận lấy mẫu và xử lý thực tế: {total_annotated} / {len(sample_df)}")

    # Phân bố filter_status
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

    print("\n3. Số lượng Sàng lọc (filter_status):")
    print(f"   - KEEP:   {filter_counts.get('KEEP', 0)} ({filter_counts.get('KEEP', 0)/total_annotated*100:.1f}%)")
    print(f"   - REMOVE: {filter_counts.get('REMOVE', 0)} ({filter_counts.get('REMOVE', 0)/total_annotated*100:.1f}%)")
    print(f"   - REVIEW: {filter_counts.get('REVIEW', 0)} ({filter_counts.get('REVIEW', 0)/total_annotated*100:.1f}%)")

    print("\n4. Số lượng Cảm xúc (sentiment - chỉ tính cho KEEP):")
    total_keep = filter_counts.get("KEEP", 0)
    if total_keep > 0:
        print(f"   - Positive: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/total_keep*100:.1f}%)")
        print(f"   - Neutral:  {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0)/total_keep*100:.1f}%)")
        print(f"   - Negative: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/total_keep*100:.1f}%)")
    else:
        print("   - Không có bản ghi KEEP nào.")

    print("\n5. Phân bố Độ tin cậy (confidence_level):")
    print(f"   - HIGH:   {confidence_counts.get('HIGH', 0)}")
    print(f"   - MEDIUM: {confidence_counts.get('MEDIUM', 0)}")
    print(f"   - LOW:    {confidence_counts.get('LOW', 0)}")

    print(f"\n6. Số bình luận cần con người kiểm tra (review_required = True): {review_required_count}")

    prompt_tok = pipeline_meta.get('prompt_tokens', 0)
    cand_tok = pipeline_meta.get('candidates_tokens', 0)
    tot_tok = pipeline_meta.get('total_tokens', 0)
    print(f"\n7. Thống kê API:")
    print(f"   - Số lượt gọi API: {pipeline_meta.get('api_calls', 0)}")
    print(f"   - Lỗi phát sinh:   {pipeline_meta.get('errors', 0)}")
    print(f"   - Tổng Token dùng: {tot_tok:,} tokens (Prompt: {prompt_tok:,} | Output: {cand_tok:,})")
    print(f"   - Thời gian xử lý: {pipeline_meta.get('elapsed_seconds', 0):.2f}s")
    print(f"8. Đường dẫn file Excel kết quả: {SAMPLE_ANNOTATED_PATH}")

    print("\n9. CÁC ĐIỂM CẦN LƯU Ý TRƯỚC KHI ÁP DỤNG CHO TOÀN BỘ DỮ LIỆU (~3.700 BÌNH LUẬN):")
    print("   - Kiểm tra kỹ các câu mỉa mai (sarcasm) và tiếng lóng mạng trong sheet 'HUMAN_REVIEW'.")
    print("   - Bình luận dạng Reply cần tiếp tục giữ ngữ cảnh cha để tránh bị LLM gán nhãn nhầm là thiếu thông tin.")
    print("   - Cần kiểm tra lại các trường hợp REMOVE xem có bị loại nhầm các bình luận ngắn nhưng có nghĩa hay không.")
    print("   - Tối ưu chi phí bằng Batch 20-30 bình luận mỗi lượt gọi khi chạy toàn bộ tập dữ liệu.")
    print("=" * 70 + "\n")


# ==========================================
# 9. Hàm Main
# ==========================================
def main():
    print("=" * 70)
    print("HỆ THỐNG GÁN NHÃN BÁN TỰ ĐỘNG 100 BÌNH LUẬN BẰNG GEMINI API")
    print("=" * 70)

    # 1. Đọc và lấy mẫu
    sample_df = inspect_and_sample_data(RAW_DATA_PATH, OUTPUT_DIR)

    # 2. Kiểm tra Gemini API
    client, model_name = setup_gemini_client()
    if client is None:
        print("\nPipeline dừng lại ở bước chuẩn bị mẫu thô do chưa có API key.")
        print(f"Mẫu thô 100 bình luận đã được lưu sẵn tại: {SAMPLE_RAW_PATH}")
        return 1

    # 3. Chạy annotation
    res = run_annotation_pipeline(client, model_name, sample_df, batch_size=10)
    annotated_dict = res["annotations"]

    if len(annotated_dict) < len(sample_df):
        print(f"Cảnh báo: Chỉ hoàn thành {len(annotated_dict)}/{len(sample_df)} bình luận.")

    # 4. Xuất file kết quả
    export_annotated_excel(sample_df, annotated_dict, SAMPLE_ANNOTATED_PATH)

    # 5. In báo cáo
    df_raw = pd.read_excel(RAW_DATA_PATH, sheet_name=0)
    print_final_report(len(df_raw), sample_df, annotated_dict, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
