"""
Script thu thập bình luận từ video TikTok thông qua đường link (TikTok Comment Scraper via URL).
Hỗ trợ cả link đầy đủ (tiktok.com/@user/video/...) và link rút gọn (vt.tiktok.com/..., vm.tiktok.com/...).

Đặc điểm:
1. Tự động phân giải link rút gọn và trích xuất Video ID.
2. Gọi trực tiếp API phân trang của TikTok để lấy toàn bộ bình luận gốc và bình luận trả lời (replies).
3. Định dạng đầu ra tương thích 100% với pipeline của dự án (lưu vào data/tiktok/raw/).
4. Cơ chế nghỉ giữa các request (jitter delay) để tránh bị rate limit.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests

# Hỗ trợ tiếng Việt cho Console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "tiktok" / "raw"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    "Referer": "https://www.tiktok.com/",
}


def extract_video_id(url_or_id: str) -> str:
    """
    Trích xuất Video ID từ chuỗi nhập vào:
    - ID số nguyên: 7540245360342469895
    - Link chuẩn: https://www.tiktok.com/@user/video/7540245360342469895
    - Link rút gọn: https://vt.tiktok.com/ZSjabcde/ hoặc https://vm.tiktok.com/...
    """
    cleaned = url_or_id.strip()
    if cleaned.isdigit():
        return cleaned

    # Xử lý link rút gọn
    if "vt.tiktok.com" in cleaned or "vm.tiktok.com" in cleaned:
        print(f"[*] Đang phân giải link rút gọn: {cleaned}")
        try:
            resp = requests.head(cleaned, allow_redirects=True, headers=DEFAULT_HEADERS, timeout=10)
            cleaned = resp.url
            print(f"[*] Link đích sau chuyển hướng: {cleaned}")
        except Exception as e:
            print(f"[!] Lỗi phân giải link rút gọn ({e}), thử tiếp với biểu thức chính quy...")

    # Tìm mẫu /video/(\d+)
    match = re.search(r"/video/(\d+)", cleaned)
    if match:
        return match.group(1)

    # Tìm bất kỳ dãy số nào có độ dài 18 - 21 chữ số
    match_digits = re.search(r"(\d{18,21})", cleaned)
    if match_digits:
        return match_digits.group(1)

    raise ValueError(f"Không thể tìm thấy TikTok Video ID hợp lệ từ link: '{url_or_id}'")


class TikTokCommentScraper:
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self.session = requests.Session()
        self.session.headers.update(headers or DEFAULT_HEADERS)

    def fetch_comments_page(self, video_id: str, cursor: int = 0, count: int = 20) -> Dict:
        """Lấy một trang bình luận cấp 1 của video."""
        url = "https://www.tiktok.com/api/comment/list/"
        params = {
            "aweme_id": video_id,
            "cursor": cursor,
            "count": count,
            "aid": "1988",
            "app_name": "tiktok_web",
        }
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=12)
                if resp.status_code == 200:
                    return resp.json()
                time.sleep(1.0)
            except requests.RequestException:
                time.sleep(1.5)
        return {}

    def fetch_replies_page(self, video_id: str, comment_id: str, cursor: int = 0, count: int = 20) -> Dict:
        """Lấy các bình luận phản hồi (replies) cho một bình luận gốc."""
        url = "https://www.tiktok.com/api/comment/list/reply/"
        params = {
            "item_id": video_id,
            "comment_id": comment_id,
            "cursor": cursor,
            "count": count,
            "aid": "1988",
            "app_name": "tiktok_web",
        }
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=12)
                if resp.status_code == 200:
                    return resp.json()
                time.sleep(1.0)
            except requests.RequestException:
                time.sleep(1.5)
        return {}

    def scrape_all_replies(self, video_id: str, comment_id: str, max_replies: int = 50) -> List[Dict]:
        """Duyệt và thu thập tất cả reply của 1 bình luận."""
        replies = []
        cursor = 0
        while len(replies) < max_replies:
            data = self.fetch_replies_page(video_id, comment_id, cursor=cursor, count=20)
            items = data.get("comments") or []
            if not items:
                break

            for item in items:
                parsed = self._parse_single_comment(item, video_id, reply_to=comment_id)
                if parsed:
                    replies.append(parsed)

            has_more = data.get("has_more", 0)
            cursor = data.get("cursor", 0)
            if not has_more or cursor == 0:
                break
            time.sleep(0.3)
        return replies

    def _parse_single_comment(self, item: Dict, video_id: str, reply_to: str = "") -> Optional[Dict]:
        """Chuẩn hóa dữ liệu 1 bình luận thành định dạng chuẩn của dự án."""
        cid = str(item.get("cid") or item.get("id") or "")
        text = str(item.get("text") or "").strip()
        if not text:
            return None

        create_time = item.get("create_time", 0)
        formatted_date = ""
        if create_time:
            try:
                formatted_date = datetime.fromtimestamp(create_time).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                formatted_date = str(create_time)

        user_info = item.get("user") or {}
        username = user_info.get("unique_id") or ""
        nickname = user_info.get("nickname") or ""
        likes = item.get("digg_count", 0)
        reply_total = item.get("reply_comment_total", 0)

        return {
            "comment_id": cid,
            "video_id": video_id,
            "comment_text": text,
            "author_username": username,
            "author_nickname": nickname,
            "likes": likes,
            "reply_total": reply_total,
            "create_time": formatted_date,
            "reply_to": reply_to,
            "comment_type": "Reply" if reply_to else "Comment",
        }

    def scrape_video_comments(
        self,
        video_id: str,
        max_comments: int = 1000,
        include_replies: bool = True,
        delay: float = 0.5,
    ) -> List[Dict]:
        """Thu thập toàn bộ bình luận cho 1 video TikTok."""
        all_comments = []
        cursor = 0
        page = 1

        print(f"\n[*] Bắt đầu cào bình luận Video ID: {video_id}")
        while len(all_comments) < max_comments:
            data = self.fetch_comments_page(video_id, cursor=cursor, count=20)
            items = data.get("comments") or []
            if not items:
                print("  -> Đã duyệt hết bình luận khả dụng.")
                break

            print(f"  -> Đang tải trang {page} (lấy được {len(items)} bình luận gốc, tổng hiện tại: {len(all_comments)})...")

            for item in items:
                parsed = self._parse_single_comment(item, video_id)
                if parsed:
                    all_comments.append(parsed)

                # Thu thập thêm replies nếu có
                cid = parsed["comment_id"] if parsed else ""
                reply_count = item.get("reply_comment_total", 0)
                if include_replies and cid and reply_count > 0:
                    replies = self.scrape_all_replies(video_id, cid)
                    all_comments.extend(replies)

                if len(all_comments) >= max_comments:
                    break

            has_more = data.get("has_more", 0)
            cursor = data.get("cursor", 0)
            if not has_more or cursor == 0:
                print("  -> Đã duyệt hết toàn bộ bình luận của video.")
                break

            page += 1
            time.sleep(delay)

        # Lọc trùng lặp comment_id
        seen_ids = set()
        unique_comments = []
        for c in all_comments:
            if c["comment_id"] not in seen_ids:
                seen_ids.add(c["comment_id"])
                unique_comments.append(c)

        print(f"[✓] Đã thu thập thành công: {len(unique_comments)} bình luận!")
        return unique_comments


def save_to_files(comments: List[Dict], video_id: str, output_dir: Path) -> Tuple[Path, Optional[Path]]:
    """Lưu danh sách bình luận vào định dạng CSV và Excel chuẩn dự án."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(comments)

    # Đặt tên file khớp chuẩn file raw hiện tại của dự án:
    # TTCommentExporter-{video_id}-{count}-comments.csv
    base_name = f"TTCommentExporter-{video_id}-{count}-comments"
    csv_path = output_dir / f"{base_name}.csv"
    excel_path = output_dir / f"{base_name}.xlsx"

    df = pd.DataFrame(comments)
    # Lưu CSV UTF-8 with BOM (utf-8-sig) để mở trên Excel không bị lỗi phông tiếng Việt
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    saved_excel = excel_path
    try:
        df.to_excel(excel_path, index=False, engine="openpyxl")
    except (ImportError, ModuleNotFoundError):
        saved_excel = None
    except Exception:
        saved_excel = None

    return csv_path, saved_excel


def main():
    parser = argparse.ArgumentParser(description="Bộ cào bình luận TikTok qua link / Video ID")
    parser.add_argument("--url", "-u", type=str, help="Đường link video TikTok hoặc Video ID")
    parser.add_argument("--max", "-m", type=int, default=1000, help="Số lượng bình luận tối đa cần lấy (mặc định: 1000)")
    parser.add_argument("--no-replies", action="store_true", help="Không lấy các bình luận trả lời (replies)")
    parser.add_argument("--outdir", "-o", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Thư mục lưu kết quả")

    args = parser.parse_args()

    input_url = args.url
    if not input_url:
        print("=" * 70)
        print(" BỘ THU THẬP BÌNH LUẬN TIKTOK (TIKTOK COMMENT SCRAPER)")
        print("=" * 70)
        input_url = input("Nhập đường link video TikTok (hoặc Video ID): ").strip()
        if not input_url:
            print("[!] Bạn chưa nhập đường link. Chương trình kết thúc.")
            return

    try:
        vid = extract_video_id(input_url)
    except Exception as e:
        print(f"[!] Lỗi: {e}")
        return

    out_dir = Path(args.outdir)
    scraper = TikTokCommentScraper()
    comments = scraper.scrape_video_comments(
        video_id=vid,
        max_comments=args.max,
        include_replies=not args.no_replies,
    )

    if not comments:
        print("[!] Không lấy được bình luận nào. Video có thể đã bị ẩn bình luận hoặc đặt ở chế độ riêng tư.")
        return

    csv_path, excel_path = save_to_files(comments, vid, out_dir)
    print("\n" + "=" * 70)
    print(" KẾT QUẢ ĐÃ LƯU VÀO THƯ MỤC RAW CỦA DỰ ÁN:")
    print("=" * 70)
    print(f"1. Tệp CSV  : {csv_path}")
    if excel_path:
        print(f"2. Tệp Excel: {excel_path}")
    else:
        print("2. Tệp Excel: (Chưa cài thư viện openpyxl, tệp CSV đã sẵn sàng sử dụng)")
    print("\n[MẸO] Bạn có thể chạy ngay lệnh sau để gộp vào tập dữ liệu lớn của dự án:")
    print("      python merge_tiktok_comments.py")


if __name__ == "__main__":
    main()
