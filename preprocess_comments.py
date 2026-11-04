"""
Module Tiền Xử Lý Dữ Liệu Bình Luận Tiếng Việt (Vietnamese Social Media Comment Preprocessor)
Dành cho bài toán phân loại cảm xúc (Sentiment Analysis) về phim ảnh.
"""

import html
import re
import unicodedata
from typing import Dict, List, Optional, Set

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None


# ==============================================================================
# 1. BẢNG TỪ ĐIỂN CHUẨN HÓA TEENCODE & TỪ VIẾT TẮT MẠNG XÃ HỘI
# ==============================================================================
TEENCODE_DICT: Dict[str, str] = {
    # Phủ định
    "ko": "không",
    "k": "không",
    "kh": "không",
    "khong": "không",
    "hong": "không",
    "hông": "không",
    "chank": "chẳng",
    "cha": "chả",
    "chua": "chưa",
    # Đại từ & xưng hô
    "mik": "mình",
    "m": "mình",
    "mk": "mình",
    "t": "tôi",
    "to": "tôi",
    "b": "bạn",
    "bn": "bạn",
    "ng": "người",
    "ngta": "người ta",
    "mn": "mọi người",
    "ae": "anh em",
    "c": "chị",
    "a": "anh",
    "e": "em",
    "ad": "admin",
    # Từ ngữ thường gặp trong phim ảnh & giải trí
    "film": "phim",
    "fim": "phim",
    "movie": "phim",
    "act": "diễn xuất",
    "actor": "diễn viên",
    "dv": "diễn viên",
    "nv": "nhân vật",
    "rap": "rạp",
    "cgv": "rạp",
    "bhd": "rạp",
    "lotte": "rạp",
    "trailer": "đoạn giới thiệu",
    "spoiler": "tiết lộ nội dung",
    "spoil": "tiết lộ nội dung",
    "review": "đánh giá",
    "rv": "đánh giá",
    "rate": "đánh giá",
    # Đánh giá & cảm xúc
    "dink": "đỉnh",
    "đỉnk": "đỉnh",
    "dinh": "đỉnh",
    "cuon": "cuốn",
    "chay": "cháy",
    "phe": "phê",
    "hayyy": "hay",
    "ok": "tốt",
    "oke": "tốt",
    "oki": "tốt",
    "okay": "tốt",
    "chán": "chán",
    "te": "tệ",
    "nhảm": "nhảm nhí",
    "flop": "thất bại",
    "hot": "nổi bật",
    # Trợ từ, liên từ & từ thông dụng
    "dc": "được",
    "đc": "được",
    "duoc": "được",
    "dk": "được",
    "vs": "với",
    "voi": "với",
    "cx": "cũng",
    "cug": "cũng",
    "cung": "cũng",
    "r": "rồi",
    "roi": "rồi",
    "rui": "rồi",
    "ntn": "như thế nào",
    "sao": "sao",
    "thg": "thường",
    "thgian": "thời gian",
    "tg": "thời gian",
    "bh": "bây giờ",
    "bjo": "bây giờ",
    "h": "giờ",
    "j": "gì",
    "gi": "gì",
    "wa": "quá",
    "qua": "quá",
    "nhiu": "nhiều",
    "nhieu": "nhiều",
    "chut": "chút",
    "ty": "tí",
    "ti": "tí",
    "nhug": "nhưng",
    "nhung": "nhưng",
    "ma": "mà",
    "lun": "luôn",
    "luon": "luôn",
    "thui": "thôi",
    "thoii": "thôi",
    "nhe": "nhé",
    "nha": "nha",
    "nheee": "nhé",
    "tks": "cảm ơn",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "cam on": "cảm ơn",
    "cảm ơn": "cảm ơn",
    # Từ cảm thán / nói tục viết tắt (chuẩn hóa về ngữ cảnh cảm thán)
    "vl": "rất",
    "vcl": "rất",
    "vch": "rất",
    "vđ": "rất",
    "vailon": "rất",
    "cmn": "luôn",
    "cmnl": "luôn",
    "omg": "trời ơi",
}

# ==============================================================================
# 2. BẢNG TỪ ĐIỂN THỰC THỂ PHIM & DIỄN VIÊN (NAMED ENTITY NORMALIZATION)
# ==============================================================================
ENTITY_DICT: Dict[str, str] = {
    # Tên phim
    "trại buôn người": "trại_buôn_người",
    "trai buon nguoi": "trại_buôn_người",
    "thiên đường máu": "thiên_đường_máu",
    "thien duong mau": "thiên_đường_máu",
    # Diễn viên & Nhân vật
    "quách ngọc ngoan": "quách_ngọc_ngoan",
    "quach ngoc ngoan": "quách_ngọc_ngoan",
    "quách ngọc tuyên": "quách_ngọc_tuyên",
    "quach ngoc tuyen": "quách_ngọc_tuyên",
    "quách ngọc": "quách_ngọc",
    "quach ngoc": "quách_ngọc",
    "qn tuyên": "quách_ngọc_tuyên",
    "qntuyen": "quách_ngọc_tuyên",
    "qn ngoan": "quách_ngọc_ngoan",
    "qnngoan": "quách_ngọc_ngoan",
    "hoàng kim ngọc": "hoàng_kim_ngọc",
    "hoang kim ngoc": "hoàng_kim_ngọc",
    "kim ngọc": "kim_ngọc",
    "võ thành tâm": "võ_thành_tâm",
    "vo thanh tam": "võ_thành_tâm",
    "hải anh kore": "hải_anh_kore",
    "hai anh kore": "hải_anh_kore",
    "ngọc trân": "ngọc_trân",
    "ngoc tran": "ngọc_trân",
    # Thuật ngữ phim ảnh hay bị tách từ sai
    "phân cảnh": "phân_cảnh",
    "phan canh": "phân_cảnh",
    "vai diễn": "vai_diễn",
    "vai dien": "vai_diễn",
    "gay cấn": "gay_cấn",
    "gay can": "gay_cấn",
    "cảm giác": "cảm_giác",
    "cam giac": "cảm_giác",
    "thực tế": "thực_tế",
    "thuc te": "thực_tế",
}

# ==============================================
# 3. BẢNG ÁNH XẠ EMOJI TOÀN DIỆN (124 EMOJI DATASET)
# ==============================================
EMOJI_DICT: Dict[str, str] = {
    # Tích cực / Yêu thích / Khen
    "❤️": " emoji_yeu_thich ", "💖": " emoji_yeu_thich ", "💕": " emoji_yeu_thich ", "🤍": " emoji_yeu_thich ",
    "🧡": " emoji_yeu_thich ", "❣": " emoji_yeu_thich ", "❤": " emoji_yeu_thich ", "🫶": " emoji_yeu_thich ",
    "😍": " emoji_yeu_thich ", "🥰": " emoji_yeu_thich ", "😘": " emoji_yeu_thich ", "🤩": " emoji_nguong_mo ",
    "🤤": " emoji_thich ", "👍": " emoji_khen ", "👏": " emoji_khen ", "👌": " emoji_khen ",
    "✌": " emoji_khen ", "💪": " emoji_khen ", "👑": " emoji_khen ", "☝": " emoji_khen ",
    "🔥": " emoji_tuyet_voi ", "💯": " emoji_tuyet_doi ", "⭐": " emoji_ngoi_sao ", "✨": " emoji_toa_sang ",
    "💥": " emoji_tuyet_voi ", "🎉": " emoji_tuyet_voi ",
    "😁": " emoji_vui ", "😄": " emoji_vui ", "😃": " emoji_vui ", "😀": " emoji_vui ",
    "😊": " emoji_vui ", "🙂": " emoji_vui ", "😉": " emoji_vui ", "😋": " emoji_vui ",
    "😌": " emoji_vui ", "😎": " emoji_vui ", "🥳": " emoji_vui ",
    "😆": " emoji_cuoi ", "🤣": " emoji_cuoi_to ", "😂": " emoji_cuoi_ra_nuoc_mat ", "🤪": " emoji_cuoi_treu ",
    "🤭": " emoji_cuoi_ngai ", "🙃": " emoji_vui ", "😏": " emoji_cuoi ",
    # Tiêu cực / Chê / Thất vọng / Buồn
    "👎": " emoji_che ", "😡": " emoji_tuc_gian ", "😠": " emoji_tuc_gian ", "🤬": " emoji_chui_the ",
    "🤮": " emoji_kinh_tom ", "🤢": " emoji_kinh_tom ", "💩": " emoji_te_hai ",
    "😭": " emoji_khoc ", "😢": " emoji_buon ", "😥": " emoji_buon ", "😞": " emoji_that_vong ",
    "😔": " emoji_that_vong ", "😒": " emoji_kho_chiu ", "🙄": " emoji_ngao_ngan ",
    "😴": " emoji_buon_ngu ", "🥱": " emoji_ngap_chan ", "🥺": " emoji_xuc_dong ", "🥹": " emoji_xuc_dong ",
    "🥲": " emoji_cuoi_trong_nuoc_mat ", "🤦": " emoji_ngao_ngan ", "🤷": " emoji_bo_tay ", "🤡": " emoji_he ",
    "😕": " emoji_that_vong ", "😖": " emoji_kho_chiu ", "😵": " emoji_choang ", "🥴": " emoji_choang ",
    "🥵": " emoji_kho_chiu ", "🫠": " emoji_that_vong ",
    # Ngạc nhiên / Bất ngờ / Suy nghĩ
    "🤔": " emoji_suy_nghi ", "🧐": " emoji_to_mo ", "😱": " emoji_soc ", "😳": " emoji_ngac_nhien ",
    "😮": " emoji_bat_ngo ", "😰": " emoji_lo_lang ", "😬": " emoji_ngai ", "🤫": " emoji_im_lang ",
    "🫢": " emoji_bat_ngo ", "🫣": " emoji_ngai ", "🫡": " emoji_chao ", "👀": " emoji_nhin ",
    "🍿": " emoji_bap_rang ", "🎬": " emoji_xem_phim ", "🎥": " emoji_xem_phim ", "🎟": " emoji_ve_xem_phim ",
    "❓": " emoji_dau_hoi ", "❗": " emoji_dau_cham_than ",
}

# ==============================================================================
# 4. DANH SÁCH TỪ PHỦ ĐỊNH TIỀN TỐ (PREFIX NEGATION WORDS)
# (Chỉ chứa các từ phủ định đứng trước vị từ; tuyệt đối không chứa 'đâu', 'thiếu', 'kém')
# ==============================================================================
NEGATION_WORDS: Set[str] = {
    "không",
    "chưa",
    "chẳng",
}

# Các từ và ký tự cấm gắn vào sau từ phủ định
INVALID_NEGATION_TARGETS: Set[str] = {
    "á", "à", "ơi", "nhỉ", "nhé", "nha", "ha", "hả", "đấy", "đâu", "nè", "thế", "vậy", "sao", "chứ", "hở",
    "tôi", "mình", "tao", "anh", "em", "chị", "bạn", "người", "họ", "chúng_tôi", "nó", "mày",
    "và", "hoặc", "nhưng", "mà", "ở", "tại", "của", "cho", "để", "vì", "do",
}

# ==============================================================================
# 4. DANH SÁCH STOPWORDS AN TOÀN CHO PHÂN LOẠI CẢM XÚC
# (Tuyệt đối không chứa từ cảm xúc như 'hay', 'tốt' và từ phủ định)
# ==============================================================================
BASIC_STOPWORDS: Set[str] = {
    "thì",
    "là",
    "mà",
    "ở",
    "tại",
    "của",
    "cho",
    "bởi",
    "do",
    "và",
    "hoặc",
    "này",
    "đó",
    "kia",
    "ấy",
    "vậy",
    "thế",
    "ạ",
    "ơi",
    "nè",
    "nhé",
    "nha",
    "ha",
    "hả",
    "đấy",
    "đây",
}


# ==============================================================================
# 5. CLASS CHÍNH: TIỀN XỬ LÝ BÌNH LUẬN (VietnameseCommentPreprocessor)
# ==============================================================================
class VietnameseCommentPreprocessor:
    """
    Bộ tiền xử lý văn bản chuyên biệt cho bình luận mạng xã hội tiếng Việt.
    Hỗ trợ chuẩn hóa Unicode, teencode, emoji, từ kéo dài, phủ định và tách từ.
    """

    def __init__(
        self,
        lower: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        normalize_teencode: bool = True,
        normalize_entities: bool = True,
        handle_emoji_mode: str = "translate",  # 'translate', 'remove', hoặc 'keep'
        normalize_elongated_chars: bool = True,
        handle_negation: bool = True,
        segment_words: bool = True,
        remove_stopwords: bool = False,
    ):
        self.lower = lower
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.normalize_teencode = normalize_teencode
        self.normalize_entities = normalize_entities
        self.handle_emoji_mode = handle_emoji_mode
        self.normalize_elongated_chars = normalize_elongated_chars
        self.handle_negation = handle_negation
        self.segment_words = segment_words
        self.remove_stopwords = remove_stopwords

        # Regex tiền biên dịch để tối ưu hiệu năng
        self.url_regex = re.compile(
            r"https?://\S+|www\.\S+|ftp://\S+", re.IGNORECASE
        )
        self.mention_regex = re.compile(r"@[\w\._-]+", re.IGNORECASE)
        self.phone_regex = re.compile(r"\b0\d{9,10}\b")
        self.html_tag_regex = re.compile(r"<.*?>")
        self.special_char_regex = re.compile(
            r"[^\w\s\.,!\?_-]"
        )  # Giữ lại một số dấu câu cảm xúc
        self.whitespace_regex = re.compile(r"\s+")

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Chuẩn hóa Unicode về dạng dựng sẵn NFC (tránh lỗi tổ hợp NFD).
        """
        if not isinstance(text, str):
            return ""
        return unicodedata.normalize("NFC", text)

    def clean_noise(self, text: str) -> str:
        """
        Loại bỏ các thành phần rác: HTML tags, URLs, Mentions (@), số điện thoại.
        """
        # Giải mã các ký tự HTML (ví dụ: &amp;, &quot;)
        text = html.unescape(text)

        # Xóa HTML tag
        text = self.html_tag_regex.sub(" ", text)

        # Xóa URL
        if self.remove_urls:
            text = self.url_regex.sub(" ", text)

        # Xóa mention (@tag)
        if self.remove_mentions:
            text = self.mention_regex.sub(" ", text)

        # Xóa số điện thoại rác
        text = self.phone_regex.sub(" ", text)

        return text

    def process_emojis(self, text: str) -> str:
        """
        Xử lý emoji: Dịch sang từ cảm xúc tương ứng hoặc loại bỏ.
        """
        if self.handle_emoji_mode == "translate":
            for emoji, translation in EMOJI_DICT.items():
                if emoji in text:
                    text = text.replace(emoji, translation)
        elif self.handle_emoji_mode == "remove":
            for emoji in EMOJI_DICT.keys():
                text = text.replace(emoji, " ")
        return text

    @staticmethod
    def remove_elongated_chars(text: str) -> str:
        """
        Chuẩn hóa từ kéo dài cảm xúc và tiếng cười.
        Ví dụ: 'phimmm hayyyy quáaaa' -> 'phim hay quá'
               'kkkk', 'hahaha' -> 'emoji_cuoi' (tránh kkk biến thành 'không')
        """
        # Chuẩn hóa các dạng tiếng cười teencode trước khi rút gọn ký tự
        text = re.sub(r"\b(k{2,}|(ha){2,}|(hi){2,}|(he){2,}|(ka){2,})\b", " emoji_cuoi ", text, flags=re.IGNORECASE)
        return re.sub(r"([a-zA-ZÀ-ỹ])\1{2,}", r"\1", text)

    @staticmethod
    def replace_teencode(text: str) -> str:
        """
        Ánh xạ teencode và từ viết tắt sang dạng tiếng Việt chuẩn.
        """
        tokens = text.split()
        normalized_tokens = [TEENCODE_DICT.get(tok, tok) for tok in tokens]
        return " ".join(normalized_tokens)

    @staticmethod
    def replace_entities(text: str) -> str:
        """
        Chuẩn hóa tên thực thể phim ảnh và diễn viên thành dạng từ ghép cố định.
        Ví dụ: 'quách ngọc ngoan' -> 'quách_ngọc_ngoan', 'trại buôn người' -> 'trại_buôn_người'
        """
        for ent_key, ent_val in ENTITY_DICT.items():
            pattern = re.compile(rf"\b{re.escape(ent_key)}\b", re.IGNORECASE)
            text = pattern.sub(ent_val, text)
        return text

    @staticmethod
    def attach_negation(text: str) -> str:
        """
        Nối từ phủ định tiền tố (không, chưa, chẳng) với từ đi liền kề sau nó.
        Tuyệt đối không ghép cứng vào emoji, dấu câu, trợ từ hoặc đại từ.
        """
        words = text.split()
        n = len(words)
        result = []
        i = 0
        while i < n:
            word = words[i]
            if word in NEGATION_WORDS and i + 1 < n:
                next_word = words[i + 1]
                # Không nối nếu từ sau là emoji, trợ từ, đại từ, hoặc dấu câu
                if (
                    not next_word.startswith("emoji_")
                    and next_word not in INVALID_NEGATION_TARGETS
                    and not re.match(r"^[^\w\s]+$", next_word)
                ):
                    result.append(f"{word}_{next_word}")
                    i += 2
                    continue
            result.append(word)
            i += 1
        return " ".join(result)

    def tokenize_vietnamese(self, text: str) -> str:
        """
        Tách từ tiếng Việt (Word Segmentation) sử dụng thư viện pyvi.
        Ví dụ: 'bộ phim rất cảm động' -> 'bộ_phim rất cảm_động'
        """
        if ViTokenizer is not None:
            return ViTokenizer.tokenize(text)
        return text

    def filter_stopwords(self, text: str) -> str:
        """
        Loại bỏ các từ hư từ (stopwords) không mang giá trị cảm xúc.
        """
        tokens = text.split()
        filtered = [t for t in tokens if t not in BASIC_STOPWORDS]
        return " ".join(filtered)

    def preprocess(self, text: str) -> str:
        """
        Pipeline tiền xử lý toàn diện cho một chuỗi bình luận.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Chuẩn hóa bảng mã Unicode NFC
        text = self.normalize_unicode(text)

        # 2. Xóa URL, tag, HTML rác
        text = self.clean_noise(text)

        # 3. Xử lý Emoji (dịch hoặc xóa)
        text = self.process_emojis(text)

        # 4. Viết thường (nếu bật)
        if self.lower:
            text = text.lower()

        # 5. Xử lý từ kéo dài (ví dụ: haaaayyyy -> hay)
        if self.normalize_elongated_chars:
            text = self.remove_elongated_chars(text)

        # 6. Chuẩn hóa teencode & từ viết tắt
        if self.normalize_teencode:
            text = self.replace_teencode(text)

        # 6.1. Chuẩn hóa thực thể tên riêng phim và diễn viên
        if self.normalize_entities:
            text = self.replace_entities(text)

        # 7. Xóa các ký tự đặc biệt thừa nhưng giữ dấu cách và từ
        text = self.special_char_regex.sub(" ", text)

        # 8. Chuẩn hóa khoảng trắng
        text = self.whitespace_regex.sub(" ", text).strip()

        # 9. Tách từ tiếng Việt (Word Segmentation)
        if self.segment_words:
            text = self.tokenize_vietnamese(text)

        # 10. Gắn từ phủ định (Negation Handling)
        if self.handle_negation:
            text = self.attach_negation(text)

        # 11. Bỏ stopwords (nếu bật)
        if self.remove_stopwords:
            text = self.filter_stopwords(text)

        # 12. Loại bỏ triệt để mọi dấu câu còn sót lại (chỉ giữ từ ngữ, số và dấu gạch dưới '_')
        text = re.sub(r"[^\w\s_]", " ", text)

        # Chuẩn hóa khoảng trắng lần cuối
        text = self.whitespace_regex.sub(" ", text).strip()

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Tiền xử lý danh sách bình luận."""
        return [self.preprocess(t) for t in texts]


# ==============================================================================
# 6. HÀM HỖ TRỢ XỬ LÝ PANDAS DATAFRAME
# ==============================================================================
def preprocess_dataframe(
    df,
    text_column: str = "Nội dung",
    output_column: str = "Nội dung tiền xử lý",
    preprocessor: Optional[VietnameseCommentPreprocessor] = None,
):
    """
    Hàm tiện ích áp dụng tiền xử lý lên một cột trong DataFrame.
    (Hàm này CHƯA được gọi tự động - sẵn sàng khi bạn muốn kích hoạt).
    """
    if preprocessor is None:
        preprocessor = VietnameseCommentPreprocessor()

    df_copy = df.copy()
    df_copy[output_column] = df_copy[text_column].apply(
        lambda x: preprocessor.preprocess(str(x or ""))
    )
    return df_copy


# ==============================================================================
# 7. VÍ DỤ MINH HỌA (DEMO MINH HỌA - CHỈ CHẠY KHI GỌI TRỰC TIẾP FILE NÀY)
# ==============================================================================
if __name__ == "__main__":
    # Minh họa test thử trên vài câu mẫu độc lập (KHÔNG đụng vào dữ liệu thật)
    demo_samples = [
        "Phimmm này xem cuốn vcl lun á mn ơiii 😍❤️❤️❤️ 10/10",
        "Diễn viên diễn đơ, kịch bản chán òm ko đáng tiền vé tí nào!! 😡👎",
        "phim ko hay như kỳ vọng, đoạn kết hơi cấn và hụt hẫng",
        "@NguyễnVănA https://facebook.com/link link rác nha mn",
    ]

    p = VietnameseCommentPreprocessor(
        lower=True,
        normalize_teencode=True,
        handle_emoji_mode="translate",
        handle_negation=True,
        segment_words=True,
    )

    print("--- DEMO MINH HỌA TIỀN XỬ LÝ VĂN BẢN (CHƯA ÁP DỤNG LÊN DỮ LIỆU) ---")
    for raw in demo_samples:
        clean = p.preprocess(raw)
        print(f"\n[Gốc]  : {raw}")
        print(f"[Xử lý]: {clean}")

