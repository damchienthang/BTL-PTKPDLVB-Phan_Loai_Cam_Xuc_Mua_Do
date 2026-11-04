import json
from pathlib import Path

import pandas as pd
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "simple_post"
OUTPUT_FILE = ROOT / "du_lieu_goc.xlsx"
PART_OUTPUT_TEMPLATE = "simple_post_comments_phan_{part}.xlsx"
STATUS_COLUMN = "Trạng thái"
LABEL_COLUMN = "Nhãn"
COLUMNS = [
    "Post ID",
    "Comment ID",
    "Author ID",
    "Tên tác giả",
    "Nội dung",
    "Loại",
]


def collect_comments(items, post_id, rows, comment_type="Comment"):
    for item in items:
        if not isinstance(item, dict):
            continue

        rows.append(
            {
                "Post ID": post_id,
                "Comment ID": item.get("comment_id", ""),
                "Author ID": item.get("author_id", ""),
                "Tên tác giả": item.get("author", ""),
                "Nội dung": item.get("text", ""),
                "Loại": comment_type,
            }
        )

        replies = item.get("replies", [])
        if isinstance(replies, list):
            collect_comments(replies, post_id, rows, "Reply")


def load_data(files):
    rows = []
    for path in files:
        with path.open(encoding="utf-8") as source:
            posts = json.load(source)
        if isinstance(posts, dict):
            posts = [posts]
        if not isinstance(posts, list):
            raise ValueError(f"Dữ liệu JSON không hợp lệ: {path}")
        for post in posts:
            if not isinstance(post, dict):
                continue
            post_id = post.get("post_id", path.stem)
            comments = post.get("comments", [])
            if isinstance(comments, list):
                collect_comments(comments, post_id, rows)

    return pd.DataFrame(rows, columns=COLUMNS)


def write_workbook(data, output_file, include_annotations=False):
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Du_lieu", index=False)
        worksheet = writer.sheets["Du_lieu"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        if include_annotations:
            status_column = data.columns.get_loc(STATUS_COLUMN) + 1
            label_column = data.columns.get_loc(LABEL_COLUMN) + 1
            last_row = len(data) + 1

            status_validation = DataValidation(
                type="list",
                formula1='"keep,remove"',
                allow_blank=True,
            )
            label_validation = DataValidation(
                type="list",
                formula1='"positive,negative,neutral"',
                allow_blank=True,
            )
            worksheet.add_data_validation(status_validation)
            worksheet.add_data_validation(label_validation)
            if last_row >= 2:
                status_validation.add(
                    f"{worksheet.cell(2, status_column).column_letter}2:"
                    f"{worksheet.cell(last_row, status_column).column_letter}"
                    f"{last_row}"
                )
                label_validation.add(
                    f"{worksheet.cell(2, label_column).column_letter}2:"
                    f"{worksheet.cell(last_row, label_column).column_letter}"
                    f"{last_row}"
                )

        for column_cells in worksheet.columns:
            max_length = max(
                len(str(cell.value or "")) for cell in column_cells[:100]
            )
            worksheet.column_dimensions[
                column_cells[0].column_letter
            ].width = min(max(max_length + 2, 12), 55)

    print(f"Created {output_file} with {len(data)} rows.")


def write_split_workbooks(data):
    base_size, remainder = divmod(len(data), 4)
    start = 0
    for part in range(1, 5):
        part_size = base_size + (1 if part <= remainder else 0)
        part_data = data.iloc[start:start + part_size].copy()
        part_data[STATUS_COLUMN] = ""
        part_data[LABEL_COLUMN] = ""
        output_file = ROOT / PART_OUTPUT_TEMPLATE.format(part=part)
        write_workbook(part_data, output_file, include_annotations=True)
        start += part_size


def main():
    files = sorted(INPUT_DIR.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file JSON trong: {INPUT_DIR}")

    all_data = load_data(files)
    write_workbook(all_data, OUTPUT_FILE)
    write_split_workbooks(all_data)


if __name__ == "__main__":
    main()
