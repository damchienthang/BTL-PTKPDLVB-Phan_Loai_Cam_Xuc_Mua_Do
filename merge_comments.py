"""
Script gộp dữ liệu bình luận từ thư mục raw cho cả Facebook và TikTok.
Hỗ trợ:
  - Facebook: Đọc tất cả các file JSON bài viết trong data/facebook/raw/simple_post/
  - TikTok: Đọc tất cả các file CSV cào được trong data/tiktok/raw/

Xuất ra định dạng Excel và CSV thống nhất với bộ cột chuẩn:
['Post ID', 'Comment ID', 'Author ID', 'Tên tác giả', 'Nội dung RAW', 'Loại', 'Trả lời cho']
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent

# Cột chuẩn thống nhất
UNIFIED_COLUMNS: Tuple[str, ...] = (
    "Post ID",
    "Comment ID",
    "Author ID",
    "Tên tác giả",
    "Nội dung RAW",
    "Loại",
    "Trả lời cho",
)

PLATFORM_DIRS: Dict[str, Dict[str, Path]] = {
    "facebook": {
        "raw_dir": ROOT_DIR / "data" / "facebook" / "raw" / "simple_post",
        "output_excel": ROOT_DIR / "data" / "facebook" / "raw" / "simple_post_comments_merged.xlsx",
        "output_csv": ROOT_DIR / "data" / "facebook" / "raw" / "simple_post_comments_merged.csv",
    },
    "tiktok": {
        "raw_dir": ROOT_DIR / "data" / "tiktok" / "raw",
        "output_excel": ROOT_DIR / "data" / "tiktok" / "tiktok_comments_merged.xlsx",
        "output_csv": ROOT_DIR / "data" / "tiktok" / "tiktok_comments_merged.csv",
    },
}

TIKTOK_ALIAS_MAP: Dict[str, str] = {
    "comment_id": "Comment ID",
    "ID nhận xét": "Comment ID",
    "video_id": "Post ID",
    "ID bài đăng": "Post ID",
    "author_id": "Author ID",
    "user_id": "Author ID",
    "ID người dùng": "Author ID",
    "author_nickname": "Tên tác giả",
    "Nickname": "Tên tác giả",
    "Biệt danh": "Tên tác giả",
    "author_username": "Username",
    "Username": "Username",
    "comment_text": "Nội dung RAW",
    "Nội dung": "Nội dung RAW",
    "RAW Text": "Nội dung RAW",
    "Bình luận": "Nội dung RAW",
    "comment_type": "Loại",
    "Loại bình luận": "Loại",
    "reply_to": "Trả lời cho",
    "Trả lời cho": "Trả lời cho",
}


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


# ==============================================================================
# MERGE FACEBOOK COMMENTS
# ==============================================================================

def collect_facebook_items(
    items: List[Dict[str, Any]],
    post_id: str,
    rows: List[Dict[str, Any]],
    comment_type: str = "Comment",
    parent_id: str = "",
) -> None:
    """Thu thập bình luận và phản hồi từ cấu trúc JSON Facebook theo đệ quy."""
    for item in items:
        if not isinstance(item, dict):
            continue

        cid = str(item.get("comment_id", "")).strip()
        rows.append({
            "Post ID": str(post_id),
            "Comment ID": cid,
            "Author ID": str(item.get("author_id", "")),
            "Tên tác giả": str(item.get("author", "")),
            "Nội dung RAW": str(item.get("text", "")).strip(),
            "Loại": comment_type,
            "Trả lời cho": parent_id,
        })

        replies = item.get("replies", [])
        if isinstance(replies, list) and replies:
            collect_facebook_items(replies, post_id, rows, comment_type="Reply", parent_id=cid)


def merge_facebook(raw_dir: Path) -> pd.DataFrame:
    """Đọc và gộp tất cả file JSON Facebook trong raw_dir."""
    json_files = list(raw_dir.glob("**/*.json"))
    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file JSON nào trong {raw_dir}")

    print(f"-> Tìm thấy {len(json_files)} file JSON Facebook.")
    rows: List[Dict[str, Any]] = []

    for path in json_files:
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            posts = data if isinstance(data, list) else [data]
            for post in posts:
                if not isinstance(post, dict):
                    continue
                pid = str(post.get("post_id", path.stem))
                comments = post.get("comments", [])
                collect_facebook_items(comments, pid, rows)
        except Exception as e:
            print(f"Lỗi đọc file {path.name}: {e}")

    df = pd.DataFrame(rows)
    return df


# ==============================================================================
# MERGE TIKTOK COMMENTS
# ==============================================================================

def merge_tiktok(raw_dir: Path) -> pd.DataFrame:
    """Đọc và gộp tất cả file CSV TikTok trong raw_dir."""
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {raw_dir}")

    print(f"-> Tìm thấy {len(csv_files)} file CSV TikTok.")
    dfs: List[pd.DataFrame] = []

    for path in csv_files:
        try:
            temp_df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except Exception:
            temp_df = pd.read_csv(path, dtype=str, encoding="utf-8", errors="replace")

        # Map tên cột về alias
        temp_df = temp_df.rename(columns=TIKTOK_ALIAS_MAP)
        dfs.append(temp_df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def run_merge(source: str) -> None:
    print("=" * 80)
    print(f"GỘP DỮ LIỆU BÌNH LUẬN NỀN TẢNG: {source.upper()}")
    print("=" * 80)

    cfg = PLATFORM_DIRS[source]
    raw_dir = cfg["raw_dir"]
    out_excel = cfg["output_excel"]
    out_csv = cfg["output_csv"]

    if not raw_dir.exists():
        raise FileNotFoundError(f"Thư mục nguồn không tồn tại: {raw_dir}")

    if source == "facebook":
        df = merge_facebook(raw_dir)
    else:
        df = merge_tiktok(raw_dir)

    total_raw = len(df)
    print(f"Tổng số bình luận đọc được từ files: {total_raw:,}")

    # Đảm bảo đầy đủ các cột chuẩn
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            # Map fallback nếu còn tên khác
            for alias, target in TIKTOK_ALIAS_MAP.items():
                if target == col and alias in df.columns:
                    df[col] = df[alias]
                    break
            else:
                df[col] = ""

    # Chuẩn hóa kiểu dữ liệu
    df["Comment ID"] = df["Comment ID"].astype(str).str.strip()
    df["Nội dung RAW"] = df["Nội dung RAW"].fillna("").astype(str).str.strip()
    df["Loại"] = df["Loại"].replace({"Bình luận": "Comment", "Phản hồi": "Reply", "": "Comment"}).fillna("Comment")

    # Khử trùng lặp theo Comment ID
    df = df[df["Comment ID"] != ""].drop_duplicates(subset=["Comment ID"], keep="first").reset_index(drop=True)
    print(f"Sau khi khử trùng lặp theo Comment ID: {len(df):,} dòng (loại {total_raw - len(df):,} dòng trùng).")

    # Chỉ giữ các cột chuẩn theo đúng thứ tự
    df = df[list(UNIFIED_COLUMNS)]

    # Xuất file CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"-> Đã xuất file CSV: {out_csv}")

    # Xuất file Excel kèm styling
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="COMMENTS", index=False)

    wb = openpyxl.load_workbook(out_excel)
    style_worksheet(wb["COMMENTS"], len(UNIFIED_COLUMNS), len(df))
    wb.save(out_excel)
    print(f"-> Đã xuất file Excel: {out_excel}")
    print(f"\n[THÀNH CÔNG] Hoàn tất gộp dữ liệu {source.upper()}!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gộp dữ liệu bình luận Facebook hoặc TikTok về schema chuẩn.")
    parser.add_argument(
        "--source",
        choices=["facebook", "tiktok"],
        default="facebook",
        help="Chọn nền tảng cần gộp: facebook hoặc tiktok (mặc định: facebook)",
    )
    args = parser.parse_args()
    run_merge(args.source)


if __name__ == "__main__":
    main()
