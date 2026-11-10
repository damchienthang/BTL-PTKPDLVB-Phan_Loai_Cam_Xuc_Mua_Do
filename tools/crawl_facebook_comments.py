"""
Script thu thập bình luận Facebook qua đường link bài viết (Facebook Comment Scraper via URL).
Thiết kế chuyên biệt cho đồ án PTKPDLVB - Tương thích 100% với luồng clean_merge_comments.py.

Đặc điểm:
1. Tự động nhận diện và trích xuất Post ID từ nhiều định dạng link Facebook (permalink, story_fbid, groups, watch, v.v.).
2. Hỗ trợ cào dữ liệu qua Facebook Web GraphQL / Mobile API sử dụng Cookie (c_user, xs) từ trình duyệt.
3. Hỗ trợ import/convert trực tiếp từ file JSON / response xuất từ DevTools nếu bài viết yêu cầu 2FA.
4. Xuất chuẩn xác 2 định dạng dữ liệu cho pipeline của dự án:
   - JSON: data/facebook/raw/simple_post/{post_id}/{post_id}.json
   - Excel: data/facebook/raw/simple_post_excel/{post_id}.xlsx
   Đúng 100% các trường: Post ID, Comment ID, Author ID, Tên tác giả, Nội dung, Loại, Lượt thích.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests

# Hỗ trợ UTF-8 cho Console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name == "tools":
    ROOT_DIR = ROOT_DIR.parent

DATA_FB_RAW = ROOT_DIR / "data" / "facebook" / "raw"
OUTPUT_JSON_DIR = DATA_FB_RAW / "simple_post"
OUTPUT_EXCEL_DIR = DATA_FB_RAW / "simple_post_excel"
TOOLS_DATA_DIR = ROOT_DIR / "tools" / "data" / "simple_post"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Dest": "document",
}


def extract_post_id(url_or_id: str) -> str:
    """
    Trích xuất Facebook Post ID từ chuỗi nhập vào:
    - ID số nguyên: 10227454655591471
    - Link trực tiếp: https://www.facebook.com/10227454655591471
    - Link story_fbid: https://www.facebook.com/permalink.php?story_fbid=2582033612270780&id=100015059863858
    - Link group: https://www.facebook.com/groups/12345/posts/2582033612270780/
    - Link posts: https://www.facebook.com/user/posts/10227454655591471/
    - Link pfbid: https://www.facebook.com/user/posts/pfbid02rHM91CeKG4Fan...
    """
    cleaned = url_or_id.strip()
    if cleaned.isdigit():
        return cleaned

    # story_fbid=...
    match_fbid = re.search(r"[?&]story_fbid=([a-zA-Z0-9_]+)", cleaned)
    if match_fbid:
        return match_fbid.group(1)

    # /posts/...
    match_posts = re.search(r"/posts/([a-zA-Z0-9_]+)", cleaned)
    if match_posts:
        return match_posts.group(1)

    # /videos/...
    match_videos = re.search(r"/videos/([0-9]+)", cleaned)
    if match_videos:
        return match_videos.group(1)

    # /watch/?v=...
    match_watch = re.search(r"[?&]v=([0-9]+)", cleaned)
    if match_watch:
        return match_watch.group(1)

    # facebook.com/10227454655591471
    match_direct = re.search(r"facebook\.com/([0-9]{12,25})", cleaned)
    if match_direct:
        return match_direct.group(1)

    # pfbid...
    match_pfbid = re.search(r"(pfbid[a-zA-Z0-9]+)", cleaned)
    if match_pfbid:
        return match_pfbid.group(1)

    # Bất kỳ chuỗi số dài nào (12-25 chữ số)
    match_digits = re.search(r"([0-9]{12,25})", cleaned)
    if match_digits:
        return match_digits.group(1)

    raise ValueError(f"Không thể trích xuất Post ID hợp lệ từ đường link Facebook: '{url_or_id}'")


def load_cookie_string(cookie_input: Optional[str] = None) -> Dict[str, str]:
    """Chuyển chuỗi cookie dạng 'c_user=...; xs=...' thành dict."""
    raw = cookie_input
    if not raw:
        # Kiểm tra file .env
        env_file = ROOT_DIR / ".env"
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("FB_COOKIE="):
                        raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not raw:
        cookie_file = ROOT_DIR / "tools" / "fb_cookies.txt"
        if cookie_file.exists():
            with open(cookie_file, "r", encoding="utf-8") as f:
                raw = f.read().strip()

    if not raw:
        return {}

    cookies = {}
    for part in raw.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies


class FacebookCommentScraper:
    def __init__(self, cookies: Optional[Dict[str, str]] = None):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        if cookies:
            self.session.cookies.update(cookies)
        self.cookies = cookies or {}

    def fetch_post_html(self, post_id: str) -> str:
        """Tải mã nguồn HTML của bài viết."""
        urls = [
            f"https://www.facebook.com/{post_id}",
            f"https://m.facebook.com/{post_id}",
        ]
        for url in urls:
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.text) > 2000:
                    return resp.text
            except Exception:
                pass
        return ""

    def parse_comments_from_graphql_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Phân tích comment từ payload JSON / GraphQL của Facebook."""
        extracted = []

        def _search_comments(obj: Any):
            if isinstance(obj, dict):
                # Nhận diện cấu trúc comment node
                if "body" in obj and ("text" in obj["body"] or "ranges" in obj["body"]):
                    cid = str(obj.get("id") or obj.get("comment_id") or "")
                    text = ""
                    if isinstance(obj.get("body"), dict):
                        text = str(obj["body"].get("text") or "").strip()
                    elif isinstance(obj.get("body"), str):
                        text = obj["body"].strip()

                    author_info = obj.get("author") or {}
                    author_name = str(author_info.get("name") or "Facebook User")
                    author_id = str(author_info.get("id") or "")
                    author_url = str(author_info.get("url") or "")

                    reaction_count = 0
                    feedback = obj.get("feedback") or {}
                    if "reactors" in feedback and isinstance(feedback["reactors"], dict):
                        reaction_count = feedback["reactors"].get("count", 0)

                    # Phản hồi (replies)
                    replies = []
                    sub_comments = obj.get("replies") or obj.get("sub_comments") or {}
                    if isinstance(sub_comments, dict) and "edges" in sub_comments:
                        for e in sub_comments["edges"]:
                            node = e.get("node") if isinstance(e, dict) else e
                            if node:
                                r_cid = str(node.get("id") or "")
                                r_text = ""
                                if isinstance(node.get("body"), dict):
                                    r_text = str(node["body"].get("text") or "").strip()
                                r_auth = node.get("author") or {}
                                replies.append({
                                    "comment_id": r_cid,
                                    "author": str(r_auth.get("name") or "Facebook User"),
                                    "author_id": str(r_auth.get("id") or ""),
                                    "author_url": str(r_auth.get("url") or ""),
                                    "text": r_text,
                                    "reaction_count": str(node.get("feedback", {}).get("reactors", {}).get("count", 0)),
                                })

                    if text and cid:
                        extracted.append({
                            "comment_id": cid,
                            "author": author_name,
                            "author_id": author_id,
                            "author_url": author_url,
                            "text": text,
                            "reaction_count": str(reaction_count),
                            "replies": replies,
                        })

                for v in obj.values():
                    _search_comments(v)
            elif isinstance(obj, list):
                for item in obj:
                    _search_comments(item)

        _search_comments(payload)
        return extracted

    def scrape_from_json_file(self, json_path: Path) -> List[Dict[str, Any]]:
        """Đọc và trích xuất bình luận từ file JSON sẵn có."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Nếu đã đúng định dạng dự án
        if isinstance(data, dict) and "comments" in data and isinstance(data["comments"], list):
            return data["comments"]

        # Nếu là raw payload
        return self.parse_comments_from_graphql_payload(data)


def export_dataset_files(
    post_id: str,
    comments: List[Dict[str, Any]],
    permalink: str = "",
) -> Tuple[Path, Path]:
    """
    Xuất dữ liệu đúng 100% thuộc tính cấu trúc của pipeline dự án:
    1. Tệp JSON tại: data/facebook/raw/simple_post/{post_id}/{post_id}.json
    2. Tệp Excel tại: data/facebook/raw/simple_post_excel/{post_id}.xlsx
    """
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    TOOLS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    post_json_folder = OUTPUT_JSON_DIR / str(post_id)
    post_json_folder.mkdir(parents=True, exist_ok=True)
    json_path = post_json_folder / f"{post_id}.json"

    # Bản backup trong tools/data/
    tool_post_folder = TOOLS_DATA_DIR / str(post_id)
    tool_post_folder.mkdir(parents=True, exist_ok=True)
    tool_json_path = tool_post_folder / f"{post_id}.json"

    excel_path = OUTPUT_EXCEL_DIR / f"{post_id}.xlsx"

    post_data = {
        "post_id": str(post_id),
        "type": "simple_post",
        "permalink": permalink or f"https://www.facebook.com/{post_id}",
        "post_info": {
            "post_story_id": "",
            "media_id": None,
        },
        "comments": comments,
    }

    # Ghi JSON UTF-8
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(post_data, f, ensure_ascii=False, indent=2)

    with open(tool_json_path, "w", encoding="utf-8") as f:
        json.dump(post_data, f, ensure_ascii=False, indent=2)

    # Chuyển đổi thành bảng dữ liệu dẹt theo đúng COLUMNS của clean_merge_comments.py:
    # ['Post ID', 'Comment ID', 'Author ID', 'Tên tác giả', 'Nội dung', 'Loại']
    rows = []
    for c in comments:
        cid = c.get("comment_id", "")
        aid = c.get("author_id", "")
        author = c.get("author", "")
        text = c.get("text", "")
        likes = c.get("reaction_count", "0")

        rows.append({
            "Post ID": str(post_id),
            "Comment ID": str(cid),
            "Author ID": str(aid),
            "Tên tác giả": author,
            "Nội dung": text,
            "Loại": "Comment",
            "Lượt thích": likes,
        })

        for r in c.get("replies", []):
            rows.append({
                "Post ID": str(post_id),
                "Comment ID": str(r.get("comment_id", "")),
                "Author ID": str(r.get("author_id", "")),
                "Tên tác giả": r.get("author", ""),
                "Nội dung": r.get("text", ""),
                "Loại": "Reply",
                "Lượt thích": r.get("reaction_count", "0"),
            })

    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False, engine="openpyxl")

    return json_path, excel_path


def main():
    parser = argparse.ArgumentParser(description="Bộ cào bình luận Facebook qua URL / Post ID (Chuẩn pipeline đồ án)")
    parser.add_argument("--url", "-u", type=str, help="Đường link bài viết Facebook hoặc Post ID")
    parser.add_argument("--cookie", "-c", type=str, help="Chuỗi Cookie Facebook (c_user=...; xs=...)")
    parser.add_argument("--input-json", "-j", type=str, help="Đường dẫn file JSON response có sẵn (nếu xuất từ DevTools)")
    parser.add_argument("--max", "-m", type=int, default=500, help="Số lượng bình luận tối đa cần lấy")

    args = parser.parse_args()

    print("=" * 80)
    print(" BỘ THU THẬP BÌNH LUẬN FACEBOOK (FACEBOOK COMMENT CRAWLER - CHUẨN ĐỒ ÁN)")
    print("=" * 80)

    # Nếu có file JSON sẵn thì import trực tiếp
    if args.input_json:
        p = Path(args.input_json)
        if not p.exists():
            print(f"[!] Không tìm thấy file: {p}")
            return
        post_id = p.stem
        scraper = FacebookCommentScraper()
        comments = scraper.scrape_from_json_file(p)
        print(f"[✓] Đã phân tích được {len(comments)} bình luận từ file JSON: {p.name}")
        json_path, excel_path = export_dataset_files(post_id, comments)
        print(f"[1] JSON chuẩn pipeline: {json_path}")
        print(f"[2] Excel chuẩn:         {excel_path}")
        print("\n[MẸO] Bạn có thể chạy ngay lệnh sau để gộp vào tập dữ liệu lớn của dự án:")
        print("      python clean_merge_comments.py")
        return

    input_url = args.url
    if not input_url:
        input_url = input("Nhập đường link bài viết Facebook (hoặc Post ID): ").strip()
        if not input_url:
            print("[!] Bạn chưa nhập đường link. Kết thúc.")
            return

    try:
        post_id = extract_post_id(input_url)
    except Exception as e:
        print(f"[!] Lỗi: {e}")
        return

    print(f"[*] Post ID xác định: {post_id}")
    cookies = load_cookie_string(args.cookie)
    if not cookies:
        print("[LƯU Ý] Chưa cấu hình Cookie Facebook (c_user & xs).")
        print("       Facebook yêu cầu đăng nhập để xem đầy đủ bình luận.")
        opt = input("Bạn có muốn dán chuỗi cookie (c_user=...; xs=...) ngay bây giờ không? (Enter để bỏ qua): ").strip()
        if opt:
            cookies = load_cookie_string(opt)

    scraper = FacebookCommentScraper(cookies=cookies)

    # Kiểm tra xem bài viết này đã có trong thư mục data/ chưa
    existing_candidates = [
        OUTPUT_JSON_DIR / str(post_id) / f"{post_id}.json",
        TOOLS_DATA_DIR / str(post_id) / f"{post_id}.json",
    ]
    found_existing = next((p for p in existing_candidates if p.exists()), None)
    if found_existing:
        print(f"[THÔNG BÁO] Tìm thấy dữ liệu đã cào sẵn cho Post ID {post_id} tại: {found_existing}")
        re_parse = input("Bạn có muốn đọc lại và chuẩn hóa xuất ra file Excel chuẩn ngay không? (y/n): ").strip().lower()
        if re_parse in ["y", "yes", ""]:
            comments = scraper.scrape_from_json_file(found_existing)
            json_path, excel_path = export_dataset_files(post_id, comments, permalink=input_url)
            print(f"[✓] Đã đồng bộ {len(comments)} bình luận:")
            print(f"  -> JSON : {json_path}")
            print(f"  -> Excel: {excel_path}")
            print("\n[MẸO] Chạy lệnh sau để gộp vào tập dữ liệu Facebook lớn:")
            print("      python clean_merge_comments.py")
            return

    # Nếu chưa có, tiến hành gửi request cào dữ liệu
    print(f"[*] Đang kết nối tới Facebook để cào bình luận bài viết {post_id}...")
    html_content = scraper.fetch_post_html(post_id)

    # Tìm comment trong script JSON nhúng
    extracted_comments = []
    if html_content:
        # Tìm các khối JSON dạng require("...") hoặc relay
        script_matches = re.findall(r'<script type="application/json"[^>]*>(.*?)</script>', html_content)
        for s in script_matches:
            try:
                data = json.loads(s)
                c_list = scraper.parse_comments_from_graphql_payload(data)
                if c_list:
                    extracted_comments.extend(c_list)
            except Exception:
                pass

    if not extracted_comments:
        print("\n[HƯỚNG DẪN THU THẬP TỰ ĐỘNG FACEBOOK KHÔNG BỊ CHẶN]:")
        print("1. Facebook hiện tại chặn chặt chẽ các lượt cào ẩn danh (cần cookie đăng nhập).")
        print("2. Cách lấy Cookie cực nhanh trong 10 giây:")
        print("   - Đăng nhập Facebook trên trình duyệt Chrome/Edge.")
        print("   - Cài extension 'Cookie-Editor' hoặc nhấn F12 -> Application -> Cookies -> facebook.com.")
        print("   - Copy giá trị 2 cookie: 'c_user' và 'xs' (ví dụ: 'c_user=1000...; xs=...').")
        print("   - Thêm vào file .env: FB_COOKIE=c_user=...; xs=...")
        print("   - Sau đó chạy lại: python tools/crawl_facebook_comments.py --url <link>")
        return

    json_path, excel_path = export_dataset_files(post_id, extracted_comments, permalink=input_url)
    print("\n" + "=" * 80)
    print(" [THÀNH CÔNG] DỮ LIỆU ĐÃ ĐƯỢC XUẤT ĐÚNG CHUẨN THUỘC TÍNH:")
    print("=" * 80)
    print(f"1. Tệp JSON chuẩn pipeline: {json_path}")
    print(f"2. Tệp Excel chuẩn:         {excel_path}")
    print(f"Tổng số bình luận thu thập: {len(extracted_comments):,} bình luận.")
    print("\n[MẸO] Bạn có thể chạy ngay lệnh sau để gộp vào tập dữ liệu lớn của dự án:")
    print("      python clean_merge_comments.py")


if __name__ == "__main__":
    main()
