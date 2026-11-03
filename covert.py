import json
from pathlib import Path

import pandas as pd


def clean_comment_text(text, author):
    if not isinstance(text, str) or not isinstance(author, str):
        return text

    text_words = text.split()
    author_words = author.split()
    if not text_words or not author_words:
        return text
    if not all(word.isalpha() and word[0].isupper() for word in text_words):
        return text

    text_suffix = [word.casefold() for word in text_words[-len(author_words):]]
    if text_suffix == [word.casefold() for word in author_words]:
        return ""
    return text


def extract_comments_to_excel(json_file_path, output_dir="simple_post_excel"):
    input_path = Path(json_file_path)
    if input_path.is_dir():
        json_files = sorted(input_path.rglob("*.json"))
    elif input_path.is_file():
        json_files = [input_path]
    else:
        raise FileNotFoundError(f"Không tìm thấy file hoặc thư mục: {input_path}")
    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file JSON trong: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_posts = 0

    def process_comments(comment_list, post_id, extracted_data, comment_type="Comment"):
        for item in comment_list:
            if not isinstance(item, dict):
                continue

            author = item.get("author", "")
            text = item.get("text", "")
            extracted_data.append({
                "Post ID": post_id,
                "Comment ID": item.get("comment_id", ""),
                "Author ID": item.get("author_id", ""),
                "Tên tác giả": author,
                "Nội dung": text,
                "Nội dung đã làm sạch": clean_comment_text(text, author),
                "Lượt tương tác": item.get("reaction_count", 0),
                "Loại": comment_type
            })

            replies = item.get("replies", [])
            if replies:
                process_comments(replies, post_id, extracted_data, comment_type="Reply")

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        posts = data if isinstance(data, list) else [data]
        for post in posts:
            if not isinstance(post, dict):
                continue
            post_id = post.get("post_id", json_file.stem)
            comments = post.get("comments", [])
            extracted_data = []
            if isinstance(comments, list):
                process_comments(comments, post_id, extracted_data)

            columns = [
                "Post ID",
                "Comment ID",
                "Author ID",
                "Tên tác giả",
                "Nội dung",
                "Nội dung đã làm sạch",
                "Lượt tương tác",
                "Loại",
            ]
            df = pd.DataFrame(extracted_data, columns=columns)
            output_file = output_dir / f"{post_id}.xlsx"
            df.to_excel(output_file, index=False, engine="openpyxl")
            exported_posts += 1
            print(f"Exported {len(df)} rows to: {output_file}")

    print(f"Exported {exported_posts} posts from {len(json_files)} JSON files.")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "data" / "simple_post"
    output_dir = script_dir / "simple_post_excel"
    extract_comments_to_excel(input_dir, output_dir)