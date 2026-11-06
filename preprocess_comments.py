"""
Module Tiền Xử Lý Dữ Liệu Bình Luận Tiếng Việt (Vietnamese Comment Preprocessor) v3
Hỗ trợ xuất song song:
  1. Text_PhoBERT: Giữ nguyên cấu trúc tự nhiên, giữ từ dừng, giữ dấu câu & điểm số (7/10),
                   chuyển emoji & emoticon sang từ ngữ cảm xúc tiếng Việt tương ứng.
  2. Text_TFIDF: Tách từ ghép (PyVi), gắn từ phủ định (không_hay), loại bỏ hư từ (stopwords),
                 giữ điểm số dạng token đặc trưng (diem_7_tren_10).
"""

import html
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

# 1. BẢNG TỪ ĐIỂN CHUẨN HÓA EMOTICON VĂN BẢN (Text Emoticons)
EMOTICON_DICT: Dict[str, str] = {
    ":)))": " emoji_cuoi_to ",
    ":))": " emoji_cuoi_to ",
    ":)": " emoji_vui ",
    ":-)": " emoji_vui ",
    "=)))": " emoji_cuoi_to ",
    "=))": " emoji_cuoi_to ",
    "=)": " emoji_vui ",
    ":D": " emoji_cuoi ",
    ":d": " emoji_cuoi ",
    ":P": " emoji_cuoi_treu ",
    ":p": " emoji_cuoi_treu ",
    ":(": " emoji_buon ",
    ":-( ": " emoji_buon ",
    ":'(": " emoji_khoc ",
    ":'-( ": " emoji_khoc ",
    "T_T": " emoji_khoc ",
    "T.T": " emoji_khoc ",
    "@@": " emoji_choang ",
    "-.-": " emoji_kho_chiu ",
    "-_-": " emoji_kho_chiu ",
    "xd": " emoji_cuoi_to ",
    "xD": " emoji_cuoi_to ",
    "XD": " emoji_cuoi_to ",
}

# 2. BẢNG ÁNH XẠ EMOJI UNICODE
EMOJI_DICT: Dict[str, str] = {
    # Tích cực / Khen / Yêu thích
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
    # Tiêu cực / Chê / Buồn
    "👎": " emoji_che ", "😡": " emoji_tuc_gian ", "😠": " emoji_tuc_gian ", "🤬": " emoji_chui_the ",
    "🤮": " emoji_kinh_tom ", "🤢": " emoji_kinh_tom ", "💩": " emoji_te_hai ",
    "😭": " emoji_khoc ", "😢": " emoji_buon ", "😥": " emoji_buon ", "😞": " emoji_that_vong ",
    "😔": " emoji_that_vong ", "😒": " emoji_kho_chiu ", "🙄": " emoji_ngao_ngan ",
    "😴": " emoji_buon_ngu ", "🥱": " emoji_ngap_chan ", "🥺": " emoji_xuc_dong ", "🥹": " emoji_xuc_dong ",
    "🥲": " emoji_cuoi_trong_nuoc_mat ", "🤦": " emoji_ngao_ngan ", "🤷": " emoji_bo_tay ", "🤡": " emoji_he ",
    "😕": " emoji_that_vong ", "😖": " emoji_kho_chiu ", "😵": " emoji_choang ", "🥴": " emoji_choang ",
    "🥵": " emoji_kho_chiu ", "🫠": " emoji_that_vong ",
    # Ngạc nhiên / Thắc mắc
    "🤔": " emoji_suy_nghi ", "🧐": " emoji_to_mo ", "😱": " emoji_soc ", "😳": " emoji_ngac_nhien ",
    "😮": " emoji_bat_ngo ", "😰": " emoji_lo_lang ", "😬": " emoji_ngai ", "🤫": " emoji_im_lang ",
    "🫢": " emoji_bat_ngo ", "🫣": " emoji_ngai ", "🫡": " emoji_chao ", "👀": " emoji_nhin ",
    "🍿": " emoji_bap_rang ", "🎬": " emoji_xem_phim ", "🎥": " emoji_xem_phim ", "🎟": " emoji_ve_xem_phim ",
    "❓": " emoji_dau_hoi ", "❗": " emoji_dau_cham_than ",
}

# 3. BẢNG TỪ ĐIỂN CHUẨN HÓA TEENCODE & VIẾT TẮT TOÀN DIỆN
TEENCODE_WORD_BOUNDARIES: List[Tuple[str, str]] = [
    # Cụm từ nhiều từ / trường hợp đặc biệt (ưu tiên khớp trước)
    (r"\b(chứ\s+)?(ko|k|kh|không)\s+p\b", "không phải"),
    (r"\bko\s+phai\b|\bk\s+phai\b", "không phải"),
    (r"\bđc\s+k\b|\bdc\s+k\b|\bđc\s+ko\b|\bdc\s+ko\b", "được không"),
    (r"\bcam\s+on\b|\btks\b|\bthanks\b|\bthank\b", "cảm ơn"),
    (r"\bko\s+sao\b|\bk\s+sao\b", "không sao"),
    (r"\bnhu\s+the\s+nao\b|\bntn\b", "như thế nào"),
    (r"\bthoi\s+gian\b|\bthgian\b|\btg\b", "thời gian"),
    (r"\bnguoi\s+ta\b|\bngta\b", "người ta"),
    (r"\bmoi\s+nguoi\b|\bmn\b", "mọi người"),
    (r"\banh\s+em\b|\bae\b", "anh em"),
    (r"\btiet\s+lo\b|\bspoil\b|\bspoiler\b", "tiết lộ nội dung"),
    (r"\btrailer\b", "đoạn giới thiệu"),
    (r"\breview\b|\brv\b", "đánh giá"),
    (r"\bnham\s+nhi\b|\bnham\b", "nhảm"),
    (r"\btuyet\s+voi\b|\bdink\b|\bđỉnk\b|\bdinh\b", "đỉnh"),
    (r"\bthat\s+bai\b|\bflop\b", "thất bại"),
    (r"\bnoi\s+bat\b|\bhot\b", "nổi bật"),
    
    # Từ ngữ cảm thán tục viết tắt (chuẩn hóa ngữ cảnh)
    (r"\b(dở|tệ|chán|xấu|kém)\s+(vl|vcl|vch|vđ|vailon)\b", r"rất \1"),
    (r"\b(hay|cuốn|đỉnh|đẹp|xuất sắc|cháy)\s+(vl|vcl|vch|vđ|vailon)\b", r"rất \1"),
    (r"\b(vl|vcl|vch|vđ|vailon)\b", "rất"),
    (r"\bcmnl\b|\bcmn\b", "luôn"),
    (r"\bomg\b", "trời ơi"),

    # Phủ định
    (r"\b(ko|k|kh|khong|hong|hông)\b", "không"),
    (r"\bchank\b", "chẳng"),
    (r"\bchua\b", "chưa"),

    # Đại từ xưng hô
    (r"\btui\b|\bto\b", "tôi"),
    (r"\bmik\b|\bmk\b", "mình"),
    (r"\bng\b", "người"),
    (r"\bad\b", "admin"),

    # Động từ & trợ từ thông dụng
    (r"\b(đc|dc|dx|đk|duoc)\b", "được"),
    (r"\b(nhìu|nhiu|nhieu)\b", "nhiều"),
    (r"\b(r|roi|rui)\b", "rồi"),
    (r"\b(trc|truoc)\b", "trước"),
    (r"\b(v|z)\b", "vậy"),
    (r"\bms\b", "mới"),
    (r"\b(nc|nch)\b", "nói chuyện"),
    (r"\b(ak|ah)\b", "à"),
    (r"\b(cx|cug|cung)\b", "cũng"),
    (r"\b(vs|voi)\b", "với"),
    (r"\b(bh|bjo)\b", "bây giờ"),
    (r"\b(j|gi)\b", "gì"),
    (r"\bwa\b", "quá"),
    (r"\b(chut|ty|ti)\b", "tí"),
    (r"\b(nhug|nhung)\b", "nhưng"),
    (r"\b(lun|luon)\b", "luôn"),
    (r"\b(thui|thoii)\b", "thôi"),
    (r"\b(nhe|nha|nheee)\b", "nha"),
    (r"\b(film|fim|movie)\b", "phim"),
    (r"\b(act|actor|dv)\b", "diễn viên"),
    (r"\bnv\b", "nhân vật"),
    (r"\b(rap|cgv|bhd|lotte)\b", "rạp"),
    (r"\b(ok|oke|oki|okay)\b", "tốt"),
    (r"\bcuon\b", "cuốn"),
    (r"\bchay\b", "cháy"),
    (r"\bphe\b", "phê"),
    (r"\bnthe\b", "như thế"),
]

# 4. BẢNG THỰC THỂ PHIM & ĐIỂM SỐ
ENTITY_DICT: Dict[str, str] = {
    "trại buôn người": "trại_buôn_người",
    "trai buon nguoi": "trại_buôn_người",
    "thiên đường máu": "thiên_đường_máu",
    "thien duong mau": "thiên_đường_máu",
    "quách ngọc ngoan": "quách_ngọc_ngoan",
    "quach ngoc ngoan": "quách_ngọc_ngoan",
    "quách ngọc tuyên": "quách_ngọc_tuyên",
    "quach ngoc tuyen": "quách_ngọc_tuyên",
    "hoàng kim ngọc": "hoàng_kim_ngọc",
    "hoang kim ngoc": "hoàng_kim_ngọc",
    "kim ngọc": "kim_ngọc",
    "võ thành tâm": "võ_thành_tâm",
    "vo thanh tam": "võ_thành_tâm",
    "hải anh kore": "hải_anh_kore",
    "ngọc trân": "ngọc_trân",
    "steven nguyễn": "steven_nguyễn",
    "steven nguyen": "steven_nguyễn",
    "hộ linh tráng sĩ": "hộ_linh_tráng_sĩ",
    "như quỳnh": "như_quỳnh",
    "minh kiên": "minh_kiên",
    "phân cảnh": "phân_cảnh",
    "vai diễn": "vai_diễn",
    "gay cấn": "gay_cấn",
    "cảm giác": "cảm_giác",
    "thực tế": "thực_tế",
    "tình tiết": "tình_tiết",
    "chân thật": "chân_thật",
}

# Từ phủ định tiền tố
NEGATION_WORDS: Set[str] = {"không", "chưa", "chẳng"}

INVALID_NEGATION_TARGETS: Set[str] = {
    "á", "à", "ơi", "nhỉ", "nhé", "nha", "ha", "hả", "đấy", "đâu", "nè", "thế", "vậy", "sao", "chứ", "hở",
    "tôi", "mình", "tao", "anh", "em", "chị", "bạn", "người", "họ", "chúng_tôi", "nó", "mày",
    "và", "hoặc", "nhưng", "mà", "ở", "tại", "của", "cho", "để", "vì", "do",
}

# Stopwords an toàn cho TF-IDF (tuyệt đối không xóa từ phủ định hay từ mang cảm xúc)
TFIDF_STOPWORDS: Set[str] = {
    "thì", "là", "mà", "ở", "tại", "của", "cho", "bởi", "do",
    "và", "hoặc", "này", "đó", "kia", "ấy", "vậy", "thế",
    "ạ", "ơi", "nè", "nhé", "nha", "ha", "hả", "đấy",
}


# 5. CLASS TIỀN XỬ LÝ CHÍNH
class VietnameseCommentPreprocessor:
    def __init__(self):
        self.url_regex = re.compile(r"https?://\S+|www\.\S+|ftp://\S+", re.IGNORECASE)
        self.mention_regex = re.compile(r"@[\w\._-]+", re.IGNORECASE)
        self.phone_regex = re.compile(r"\b0\d{9,10}\b")
        self.html_tag_regex = re.compile(r"<.*?>")
        self.whitespace_regex = re.compile(r"\s+")

    @staticmethod
    def normalize_unicode(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return unicodedata.normalize("NFC", text)

    def clean_noise(self, text: str) -> str:
        text = html.unescape(text)
        text = self.html_tag_regex.sub(" ", text)
        text = self.url_regex.sub(" ", text)
        text = self.mention_regex.sub(" ", text)
        text = self.phone_regex.sub(" ", text)
        return text

    @staticmethod
    def replace_emoticons(text: str) -> str:
        for emo, rep in sorted(EMOTICON_DICT.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(emo, rep)
        return text

    @staticmethod
    def replace_emojis(text: str) -> str:
        for emo, rep in EMOJI_DICT.items():
            if emo in text:
                text = text.replace(emo, rep)
        return text

    @staticmethod
    def normalize_elongated(text: str) -> str:
        # Tiếng cười: kkk, hahah, hihi
        text = re.sub(r"\b(k{2,}|(ha){2,}|(hi){2,}|(he){2,}|(ka){2,})\b", " emoji_cuoi ", text, flags=re.IGNORECASE)
        # Rút gọn ký tự lặp >= 3 lần (hayyyy -> hay)
        return re.sub(r"([a-zA-ZÀ-ỹ])\1{2,}", r"\1", text)

    @staticmethod
    def normalize_score(text: str) -> str:
        # Giữ điểm số: 7/10, 8.5/10, 9/10, 10/10
        def score_repl(match):
            val = match.group(1).replace(".", "_")
            total = match.group(2)
            return f" diem_{val}_tren_{total} "

        text = re.sub(r"\b(\d+(?:\.\d+)?)\s*/\s*(10|100)\b", score_repl, text)
        return text

    @staticmethod
    def replace_teencode(text: str) -> str:
        # Đệm khoảng trắng quanh dấu câu cơ bản trước để tránh dính từ (vd: "thật.ko" -> "thật . ko")
        t = re.sub(r"([,\.!?;:])", r" \1 ", text)
        for pattern, replacement in TEENCODE_WORD_BOUNDARIES:
            t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
        return t

    @staticmethod
    def attach_negation_in_tokens(tokens_with_punct: List[str]) -> List[str]:
        """
        Nối từ phủ định tiền tố (không, chưa, chẳng) với từ ngay sau nó,
        NHƯNG CHỈ NỐI NẾU KHÔNG CÓ DẤU CÂU PHÂN TÁCH Ở GIỮA.
        Ví dụ: ['không', ',', 'má'] -> KHÔNG NỐI
               ['không', 'hay'] -> 'không_hay'
        """
        puncts = {",", ".", "!", "?", ";", ":", "-", "...", "—", "–"}
        result = []
        n = len(tokens_with_punct)
        i = 0
        while i < n:
            tok = tokens_with_punct[i]
            if tok in NEGATION_WORDS and i + 1 < n:
                next_tok = tokens_with_punct[i + 1]
                # Nếu từ tiếp theo là dấu câu -> từ phủ định này đứng cuối vế, tuyệt đối không nối
                if next_tok in puncts:
                    result.append(tok)
                    i += 1
                    continue
                # Kiểm tra các điều kiện an toàn khác
                if (
                    not next_tok.startswith("emoji_")
                    and not next_tok.startswith("diem_")
                    and next_tok not in INVALID_NEGATION_TARGETS
                    and not re.match(r"^[^\w\s]+$", next_tok)
                ):
                    result.append(f"{tok}_{next_tok}")
                    i += 2
                    continue
            result.append(tok)
            i += 1
        return result

    def preprocess_phobert(self, text: str) -> str:
        """
        Tiền xử lý chuẩn hóa cho PhoBERT (Word-level Language Model):
        - Chuẩn hóa Unicode NFC.
        - Xóa URL, tag, mention rác.
        - Chuyển Emoticon & Emoji sang mô tả cảm xúc tiếng Việt.
        - Chuẩn hóa teencode & viết tắt.
        - Viết thường (lower).
        - Chuẩn hóa thực thể phim (ENTITY_DICT).
        - TÁCH TỪ TIẾNG VIỆT (Word Segmentation nối dấu gạch dưới `_` bằng PyVi) - Bắt buộc cho PhoBERT.
        - GIỮ NGUYÊN dấu câu ngữ pháp (. , ! ?) và từ dừng (stopwords) để bảo tồn attention context.
        """
        if not text or not str(text).strip():
            return ""

        t = self.normalize_unicode(str(text))
        t = self.clean_noise(t)
        t = self.replace_emoticons(t)
        t = self.replace_emojis(t)
        t = self.normalize_elongated(t)
        t = self.normalize_score(t)
        t = self.replace_teencode(t)

        t = t.lower()
        t = self.whitespace_regex.sub(" ", t).strip()

        # Chuẩn hóa thực thể
        for ent_k, ent_v in ENTITY_DICT.items():
            pattern = re.compile(rf"\b{re.escape(ent_k)}\b", re.IGNORECASE)
            t = pattern.sub(ent_v, t)

        # Tách từ tiếng Việt bằng PyVi (nối `_` giữa các từ ghép, giữ nguyên dấu câu)
        if ViTokenizer is not None:
            t = ViTokenizer.tokenize(t)

        t = self.whitespace_regex.sub(" ", t).strip()
        return t

    def preprocess_tfidf(self, text: str) -> str:
        """
        Tiền xử lý cho TF-IDF / Naive Bayes / SVM:
        - Kế thừa toàn bộ từ bước PhoBERT (đã tách từ ghép `_`).
        - Xử lý gắn từ phủ định (không_hay, chưa_từng).
        - XÓA BỎ DẤU CÂU (chỉ giữ lại từ ngữ, số và ký tự nối `_`).
        - LOẠI BỎ TỪ DỪNG (stopwords) để tránh làm loãng ma trận thưa.
        """
        if not text or not str(text).strip():
            return ""

        # Lấy bản đã chuẩn hóa và tách từ từ bước PhoBERT
        t = self.preprocess_phobert(text)

        # Tokenize có giữ dấu câu tạm thời để gắn từ phủ định an toàn (không nối xuyên dấu câu)
        tokens_with_punct = re.findall(r"[\w_]+|[,\.!?;:]", t)
        attached_tokens = self.attach_negation_in_tokens(tokens_with_punct)

        # Xóa sạch dấu câu
        puncts = {",", ".", "!", "?", ";", ":", "-", "...", "—", "–"}
        clean_tokens = [tok for tok in attached_tokens if tok not in puncts]

        # Loại bỏ stopwords
        final_tokens = [tok for tok in clean_tokens if tok not in TFIDF_STOPWORDS]
        t = " ".join(final_tokens)

        # Chuẩn hóa khoảng trắng lần cuối
        t = self.whitespace_regex.sub(" ", t).strip()
        return t

    def preprocess(self, text: str) -> str:
        return self.preprocess_tfidf(text)
