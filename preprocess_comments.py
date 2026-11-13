"""
Module Tiền Xử Lý Dữ Liệu Bình Luận Tiếng Việt (Vietnamese Comment Preprocessor)
Clean Code Refactored:
  - PEP 8 naming convention & constants separation.
  - Flattened nested logic & early return guard clauses.
  - Zero hardcoding / magic values.
  - Dual output:
      1. Text_PhoBERT: Giữ nguyên cấu trúc tự nhiên, giữ từ dừng, giữ dấu câu & điểm số (7/10).
                       Xóa sạch emoji & emoticon. Tách từ ghép (_) bằng PyVi.
      2. Text_TFIDF: Tách từ ghép (PyVi), gắn từ phủ định (không_hay), loại bỏ hư từ (stopwords),
                     xóa sạch dấu câu, giữ điểm số dạng token đặc trưng (diem_7_tren_10).
"""

import html
import re
import unicodedata
from typing import Dict, List, Optional, Pattern, Set, Tuple

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None


EMOTICON_LIST: Tuple[str, ...] = (
    ":)))", ":))", ":)", ":-)", "=)))", "=))", "=)",
    ":D", ":d", ":P", ":p", ":(", ":-(", ":'(", ":'-(",
    "T_T", "T.T", "@@", "-.-", "-_-", "xd", "xD", "XD",
    "^^", "^_^", ":v", ":3", "<3",
)

EMOJI_UNICODE_PATTERN: Pattern[str] = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Mặt cười / emoticons
    "\U0001F300-\U0001F5FF"  # Biểu tượng & pictographs
    "\U0001F680-\U0001F6FF"  # Giao thông & bản đồ
    "\U0001F1E0-\U0001F1FF"  # Cờ các quốc gia
    "\U00002702-\U000027B0"  # Ký hiệu hoa thị, dấu tích
    "\U000024C2-\U0001F251"  # Ký tự bao đóng
    "\U0001F900-\U0001F9FF"  # Ký hiệu bổ sung (thả tim tay, mặt nôn...)
    "\U0001FA70-\U0001FAFF"  # Ký hiệu mở rộng A
    "\U00002600-\U000026FF"  # Biểu tượng linh tinh (ngôi sao, trái tim...)
    "\U0000FE00-\U0000FE0F"  # Biến thể giao diện emoji
    "\U0001F000-\U0001F02F"
    "]+",
    flags=re.UNICODE,
)

EMOJI_TOKEN_PATTERN: Pattern[str] = re.compile(r"\bemoji_\w+\b", re.IGNORECASE)
LAUGHTER_NOISE_PATTERN: Pattern[str] = re.compile(
    r"\b(k{2,}|(ha){2,}|(hi){2,}|(he){2,}|(ka){2,})\b",
    flags=re.IGNORECASE,
)
PUNCTUATION_PADDING_PATTERN: Pattern[str] = re.compile(r"([,\.!?;:])")
REPEATED_CHARS_PATTERN: Pattern[str] = re.compile(r"([a-zA-ZÀ-ỹ])\1{2,}")
SCORE_FRACTION_PATTERN: Pattern[str] = re.compile(r"\b(\d+(?:\.\d+)?)\s*/\s*(10|100)\b")
NON_WORD_SYMBOL_PATTERN: Pattern[str] = re.compile(r"^[^\w\s]+$")
PUNCTUATION_TOKEN_SPLIT_PATTERN: Pattern[str] = re.compile(r"[\w_]+|[,\.!?;:]")
NOISE_PATTERN: Pattern[str] = re.compile(
    r"<.*?>|https?://\S+|www\.\S+|ftp://\S+|@[\w\._-]+|\b0\d{9,10}\b",
    flags=re.IGNORECASE,
)
WHITESPACE_PATTERN: Pattern[str] = re.compile(r"\s+")

# Danh sách tên người dùng được tag ở đầu Reply (đặc thù Facebook không có dấu @)
EXTRA_TAG_NAMES: Tuple[str, ...] = (
    "Quyên Kim Pham", "Tăng Mỹ Vy", "Trương Anh", "Andy Nguyễn", "Andy Nguyễnhay",
    "Mạnh Ke", "Le Quangg", "Đỗ Đình Bảo", "Đỗ Đình Bảo Phúc", "Đinh Nhụy Huỳnh",
    "Huỳnh Thành", "Lvi Pham", "Võ Sơn Hải", "Tống Tuệ Nghi", "Jessica Phan",
    "Nguyễn Thị Thùy", "Nguyễn Thị Thùy Linh", "Tấn Phát", "Hoàng Anh", "Bích Ngân",
    "Ltn Hồng Huong", "Ltn Hồng Huong Lan Dao", "Chin Su", "Nguyễn Quỳnh Trang",
    "Tuấn Phương", "Phan Thị Phương", "Phan Thị Phương Nhi", "Minh Kiên", "Po Phạm",
    "Hộ Linh Tráng", "Hộ Linh Tráng Sĩ", "Huynh Kimlien", "Steven Nguyen", "Nova Grace",
    "Huyền Ỉn", "Tú Đào", "Cô Như Quỳnh", "Trương Anh", "Dương Nguyễn", "Duc Linh Duong Nguyen",
    "Đinh Nhụy Huỳnh", "Ngọc Mai", "Nguyễn Ngô Bảo Phong", "Ty Nguyen", "Châu Như",
    "Nguyễn Dung", "Hồng Quyên", "Thanh Ly", "Thu Hương", "Nhữ Thị Thảo", "Bảo Trâm",
    "Phạm Thoại", "Quỳnh Như", "Thảo My", "Hải Đăng", "Khánh Linh", "Phương Uyên",
)

# Regex bóc tách tên người dùng ở đầu câu
AUTHOR_PREFIX_PATTERN: Pattern[str] = re.compile(
    rf"^(?:{'|'.join(re.escape(name) for name in sorted(EXTRA_TAG_NAMES, key=len, reverse=True))})\s+",
    flags=re.IGNORECASE,
)



RAW_TEENCODE_MAPPINGS: Tuple[Tuple[str, str], ...] = (
    ("k", "không"), ("ko", "không"), ("khong", "không"), ("hong", "không"),
    ("hổng", "không"), ("hok", "không"), ("hem", "không"), ("kô", "không"),
    ("kh", "không"), ("hông", "không"),
    ("dc", "được"), ("đc", "được"), ("duoc", "được"), ("dk", "được"),
    ("đk", "được"), ("ok", "được"), ("oki", "được"), ("oke", "được"),
    ("okay", "được"), ("r", "rồi"), ("roi", "rồi"), ("rùi", "rồi"),
    ("cg", "cũng"), ("cug", "cũng"), ("cx", "cũng"),
    ("vs", "với"), ("voi", "với"), ("va", "và"),
    ("ng", "người"), ("ngừoi", "người"), ("nguoi", "người"), ("mn", "mọi người"),
    ("mng", "mọi người"), ("moi nguoi", "mọi người"),
    ("ae", "anh em"), ("e", "em"), ("a", "anh"),
    ("thg", "thằng"), ("đĩ", "người"), ("con me", "người"),
    ("cl", "quá"), ("clgt", "gì thế"), ("vcl", "quá"),
    ("vl", "quá"), ("vcll", "quá"), ("vkl", "quá"),
    ("vlon", "quá"), ("vlin", "quá"), ("vloz", "quá"),
    ("đm", "rất"), ("dm", "rất"), ("đcm", "rất"), ("dcm", "rất"),
    ("vđ", "vấn đề"), ("vd", "vấn đề"),
    ("bt", "bình thường"), ("bth", "bình thường"), ("bthg", "bình thường"),
    ("j", "gì"), ("gi", "gì"), ("ntn", "như thế nào"),
    ("sao", "sao"), ("thê", "thế"),
    ("cty", "công ty"), ("chán", "chán"), ("nhảm", "nhảm"),
    ("drop", "bỏ xem"), ("skip", "bỏ qua"),
    ("thik", "thích"), ("thix", "thích"), ("thich", "thích"),
    ("iu", "yêu"), ("yeu", "yêu"), ("me", "mê"),
    ("qúa", "quá"), ("wa", "quá"), ("qua", "quá"),
    ("rat", "rất"), ("cuc", "cực"), ("cực kì", "cực kỳ"),
    ("đág", "đáng"), ("dag", "đang"), ("đag", "đang"),
    ("t", "tôi"), ("tui", "tôi"), ("toi", "tôi"),
    ("mik", "mình"), ("mk", "mình"), ("minh", "mình"),
    ("fiml", "phim"), ("fim", "phim"), ("flim", "phim"),
    ("film", "phim"), ("phjm", "phim"), ("pm", "phim"),
    ("dv", "diễn viên"), ("dien vien", "diễn viên"),
    ("nv", "nhân vật"), ("nhan vat", "nhân vật"),
    ("rap", "rạp"), ("ve", "vé"), ("suat", "suất"),
    ("doan", "đoạn"), ("tap", "tập"),
    ("nhiu", "nhiều"), ("nhìu", "nhiều"), ("it", "ít"),
    ("h", "giờ"), ("gio", "giờ"),
    ("review", "đánh giá"), ("rv", "đánh giá"),
    ("acting", "diễn xuất"), ("plot", "cốt truyện"),
    ("twist", "bất ngờ"), ("ending", "kết thúc"),
    ("end", "kết"), ("happy ending", "kết thúc có hậu"),
    ("sad ending", "kết thúc buồn"), ("open ending", "kết thúc mở"),
    ("cringe", "gượng gạo"), ("overrated", "được đánh giá quá cao"),
    ("underrated", "chưa được đánh giá đúng mức"),
    ("flop", "thất bại"), ("hit", "thành công"),
    ("masterpiece", "kiệt tác"), ("trash", "rác phẩm"),
    ("cinema", "điện ảnh"), ("cast", "dàn diễn viên"),
    ("cameo", "khách mời"), ("trailer", "đoạn giới thiệu"),
    ("spoiler", "tiết lộ nội dung"), ("spoil", "tiết lộ"),
)

# Biên dịch trước Regex boundaries cho Teencode để tăng tốc
PRECOMPILED_TEENCODE_RULES: Tuple[Tuple[Pattern[str], str], ...] = tuple(
    (re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE), tgt)
    for src, tgt in sorted(RAW_TEENCODE_MAPPINGS, key=lambda pair: len(pair[0]), reverse=True)
)

ENTITY_NORMALIZATION_MAP: Dict[str, str] = {
    "quách ngọc ngoan": "Quách_Ngọc_Ngoan",
    "quách ngọc tuyên": "Quách_Ngọc_Tuyên",
    "hoàng mèo": "Hoàng_Mèo",
    "mạc văn khoa": "Mạc_Văn_Khoa",
    "trại buôn người": "Trại_Buôn_Người",
    "hùng long phong bá": "Hùng_Long_Phong_Bá",
    "diễn viên": "diễn_viên",
    "diễn xuất": "diễn_xuất",
    "kịch bản": "kịch_bản",
    "âm thanh": "âm_thanh",
    "hình ảnh": "hình_ảnh",
    "kỹ xảo": "kỹ_xảo",
    "cốt truyện": "cốt_truyện",
    "kết thúc": "kết_thúc",
    "phân cảnh": "phân_cảnh",
    "vai diễn": "vai_diễn",
    "gay cấn": "gay_cấn",
    "cảm giác": "cảm_giác",
    "thực tế": "thực_tế",
    "tình tiết": "tình_tiết",
    "chân thật": "chân_thật",
}

PRECOMPILED_ENTITY_RULES: Tuple[Tuple[Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v)
    for k, v in ENTITY_NORMALIZATION_MAP.items()
)


NEGATION_PREFIX_WORDS: Set[str] = {"không", "chưa", "chẳng"}

PUNCTUATION_DELIMITERS: Set[str] = {
    ",", ".", "!", "?", ";", ":", "-", "...", "—", "–",
}

INVALID_NEGATION_TARGETS: Set[str] = {
    "á", "à", "ơi", "nhỉ", "nhé", "nha", "ha", "hả", "đấy", "đâu", "nè", "thế", "vậy", "sao", "chứ", "hở",
    "tôi", "mình", "tao", "anh", "em", "chị", "bạn", "người", "họ", "chúng_tôi", "nó", "mày",
    "và", "hoặc", "nhưng", "mà", "ở", "tại", "của", "cho", "để", "vì", "do",
}

TFIDF_STOPWORDS: Set[str] = {
    "thì", "là", "mà", "ở", "tại", "của", "cho", "bởi", "do",
    "và", "hoặc", "này", "đó", "kia", "ấy", "vậy", "thế",
    "ạ", "ơi", "nè", "nhé", "nha", "ha", "hả", "đấy",
}


class VietnameseCommentPreprocessor:

    @staticmethod
    def normalize_unicode(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def clean_noise(text: str) -> str:
        """Loại bỏ mã html, link url, mention tag (@name), số điện thoại và tên người dùng dính đầu câu."""
        t = html.unescape(text)
        t = NOISE_PATTERN.sub(" ", t)
        t = AUTHOR_PREFIX_PATTERN.sub("", t.strip())
        return t

    @staticmethod
    def remove_emoticons(text: str) -> str:
        for emo in EMOTICON_LIST:
            if emo in text:
                text = text.replace(emo, " ")
        return text

    @classmethod
    def remove_emojis(cls, text: str) -> str:
        t = EMOJI_UNICODE_PATTERN.sub(" ", text)
        t = EMOJI_TOKEN_PATTERN.sub(" ", t)
        return t

    @staticmethod
    def normalize_elongated(text: str) -> str:
        """Loại bỏ tiếng cười lặp (kkk, hahah) và rút gọn ký tự kéo dài (hayyyy -> hay)."""
        t = LAUGHTER_NOISE_PATTERN.sub(" ", text)
        return REPEATED_CHARS_PATTERN.sub(r"\1", t)

    @staticmethod
    def normalize_score(text: str) -> str:
        def format_score(match: re.Match) -> str:
            val = match.group(1).replace(".", "_")
            total = match.group(2)
            return f" diem_{val}_tren_{total} "

        return SCORE_FRACTION_PATTERN.sub(format_score, text)

    @staticmethod
    def replace_teencode(text: str) -> str:
        """Chuẩn hóa teencode, từ viết tắt và tiếng lóng."""
        t = PUNCTUATION_PADDING_PATTERN.sub(r" \1 ", text)
        for pattern, replacement in PRECOMPILED_TEENCODE_RULES:
            t = pattern.sub(replacement, t)
        return t

    @staticmethod
    def attach_negation_in_tokens(tokens_with_punct: List[str]) -> List[str]:
        """
        Nối từ phủ định tiền tố (không, chưa, chẳng) với từ sau nó (không_hay).
        Flattened logic với early guard clauses để tránh lồng nhau.
        """
        result: List[str] = []
        n = len(tokens_with_punct)
        i = 0

        while i < n:
            tok = tokens_with_punct[i]

            # Guard clause 1: Không phải từ phủ định hoặc là từ cuối cùng
            if tok not in NEGATION_PREFIX_WORDS or i + 1 >= n:
                result.append(tok)
                i += 1
                continue

            next_tok = tokens_with_punct[i + 1]

            # Guard clause 2: Có dấu câu ngăn cách ở giữa
            if next_tok in PUNCTUATION_DELIMITERS:
                result.append(tok)
                i += 1
                continue

            # Guard clause 3: Từ tiếp theo thuộc danh sách cấm nối
            is_valid_target = (
                not next_tok.startswith("diem_")
                and next_tok not in INVALID_NEGATION_TARGETS
                and not NON_WORD_SYMBOL_PATTERN.match(next_tok)
            )

            if not is_valid_target:
                result.append(tok)
                i += 1
                continue

            result.append(f"{tok}_{next_tok}")
            i += 2

        return result

    def preprocess_phobert(self, text: str) -> str:
        if not text or not str(text).strip():
            return ""

        t = self.normalize_unicode(str(text))
        t = self.clean_noise(t)
        t = self.remove_emoticons(t)
        t = self.remove_emojis(t)
        t = self.normalize_elongated(t)
        t = self.normalize_score(t)
        t = self.replace_teencode(t)

        t = t.lower()
        t = WHITESPACE_PATTERN.sub(" ", t).strip()

        # Chuẩn hóa thực thể
        for pattern, replacement in PRECOMPILED_ENTITY_RULES:
            t = pattern.sub(replacement, t)

        if ViTokenizer is not None:
            t = ViTokenizer.tokenize(t)

        return WHITESPACE_PATTERN.sub(" ", t).strip()

    def preprocess_tfidf(self, text: str) -> str:
        if not text or not str(text).strip():
            return ""

        t = self.preprocess_phobert(text)

        tokens_with_punct = PUNCTUATION_TOKEN_SPLIT_PATTERN.findall(t)
        attached_tokens = self.attach_negation_in_tokens(tokens_with_punct)

        filtered_tokens = [
            tok for tok in attached_tokens
            if tok not in PUNCTUATION_DELIMITERS and tok not in TFIDF_STOPWORDS
        ]

        return WHITESPACE_PATTERN.sub(" ", " ".join(filtered_tokens)).strip()
