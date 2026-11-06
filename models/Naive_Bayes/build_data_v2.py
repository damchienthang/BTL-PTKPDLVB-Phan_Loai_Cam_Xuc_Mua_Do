"""Dựng dữ liệu v2: văn bản clean_v2, pool train bên ngoài, test-v2. Không huấn luyện mô hình nào ở đây.

Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/build_data_v2.py

Đầu vào: Data/final_labeled_clean_v3.xlsx và models/Naive_Bayes/datasets/splits/*.csv, models/Naive_Bayes/datasets/external/*/*.parquet,
         models/Naive_Bayes/datasets/label_audit.csv
Đầu ra:  models/Naive_Bayes/datasets/v2/ (không sửa dữ liệu gốc)

Quy đổi nhãn UIT-VSMEC (khai báo trước khi có kết quả mô hình):
    Enjoyment -> Positive; Sadness, Fear, Anger, Disgust -> Negative; Other -> Neutral; Surprise -> bỏ
"""
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from underthesea import word_tokenize

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "models" / "Naive_Bayes" / "datasets"
OUT = DS / "v2"
OUT.mkdir(exist_ok=True)

EMOTION = {"Enjoyment": "Positive", "Sadness": "Negative", "Fear": "Negative", "Anger": "Negative", "Disgust": "Negative", "Other": "Neutral"}
TEEN = {
    "k": "không", "ko": "không", "kh": "không", "kg": "không", "khum": "không", "hong": "không", "hông": "không", "hok": "không", "hk": "không",
    "hem": "không", "khong": "không", "kô": "không", "dc": "được", "đc": "được", "đk": "được", "j": "gì", "vs": "với", "r": "rồi", "rùi": "rồi",
    "cx": "cũng", "ms": "mới", "mn": "mọi người", "mng": "mọi người", "ng": "người", "ngta": "người ta", "ntn": "như thế nào", "bth": "bình thường",
    "bthuong": "bình thường", "lm": "làm", "fim": "phim", "film": "phim", "z": "vậy", "v": "vậy", "t": "tôi", "m": "mày", "nhìu": "nhiều", "nhiu": "nhiều",
    "thui": "thôi", "iu": "yêu", "tks": "cảm ơn", "thanks": "cảm ơn",
}
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF⭐]")
CJK = re.compile(r"[㐀-鿿]")


def strip_tag(t):
    """Bỏ tên được tag đầu câu: chuỗi từ 2 từ viết hoa chữ đầu trở lên, hoặc từ đầu tiên là ký tự CJK."""
    w = t.split()
    n = 0
    while n < min(len(w), 5) and w[n][:1].isupper() and not w[n].isupper():
        n += 1
    if n >= 2:
        return " ".join(w[n:])
    if w and CJK.search(w[0]):
        return " ".join(w[1:])
    return t


def clean_v2(text, tagged=False):
    t = unicodedata.normalize("NFC", str(text))
    if tagged:
        t = strip_tag(t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\S+", " ", t).lower()
    t = EMOJI.sub(lambda m: f" emo{ord(m.group()):x} ", t)
    t = re.sub(r"[:=;]+-?\)+", " emocuoi ", t)
    t = re.sub(r"[:=;]+-?\(+", " emobuon ", t)
    t = re.sub(r"([^\W\d_])\1{2,}", r"\1", t)              # "hayyy" -> "hay"
    t = t.replace("?", " qmark ").replace("!", " exmark ")
    t = re.sub(r"[^\w\s]", " ", t)                         # giữ chữ số, bỏ dấu câu khác
    toks = [TEEN.get(w, w) for w in t.split()]
    return " ".join(toks)


def segment(t):
    return word_tokenize(t, format="text") if t else t


def norm_key(t):
    t = unicodedata.normalize("NFD", t.lower().replace("đ", "d"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", t)


def shingles(t, n=5):
    t = re.sub(r"\s+", " ", t)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


t0 = time.time()
xl = pd.read_excel(ROOT / "Data" / "final_labeled_clean_v3.xlsx", keep_default_na=False, usecols=["Comment ID", "RAW Text", "Final Label"]).set_index("Comment ID")
film = []
for s in ("train", "val", "test"):
    d = pd.read_csv(ROOT / "models" / "Naive_Bayes" / "datasets" / "splits" / f"{s}.csv", encoding="utf-8-sig", keep_default_na=False)
    film.append(d[["Comment ID", "Final Label"]].assign(split=s))
film = pd.concat(film, ignore_index=True)
film["text_v2"] = [segment(clean_v2(xl.loc[i, "RAW Text"], tagged=True)) for i in film["Comment ID"]]
film = film.rename(columns={"Final Label": "label"})
print("film", len(film), "| rỗng:", int((film.text_v2.str.strip() == "").sum()), f"| {time.time() - t0:.0f}s")
film[["Comment ID", "split", "label", "text_v2"]].to_csv(OUT / "film_clean_v2.csv", index=False, encoding="utf-8-sig")

ext = DS / "external"
vs = {s: pd.read_parquet(ext / "UIT-VSMEC" / f"{s}.parquet") for s in ("train", "validation", "test")}


def vsmec(d, split):
    o = d[d.Emotion != "Surprise"].copy()
    o["label"] = o.Emotion.map(EMOTION)
    o["text_v2"] = [segment(clean_v2(x)) for x in o.Sentence]
    return o.assign(source=f"UIT-VSMEC/{split}")[["source", "Emotion", "label", "text_v2"]]


pool_me = vsmec(vs["train"], "train")
test_v2 = pd.concat([vsmec(vs["validation"], "validation"), vsmec(vs["test"], "test")], ignore_index=True)
fc = pd.read_parquet(ext / "UIT-VSFC" / "train.parquet")
pool_fc = pd.DataFrame({"source": "UIT-VSFC/train", "label": fc.label.str.capitalize(), "text_v2": [segment(clean_v2(x)) for x in fc.text]})
print("xong clean_v2 ngoài", f"{time.time() - t0:.0f}s")

report = {"raw": {"test_v2": len(test_v2), "pool_vsmec": len(pool_me), "pool_vsfc": len(pool_fc)}}
for name, d in (("test_v2", test_v2), ("pool_vsmec", pool_me), ("pool_vsfc", pool_fc)):
    d["key"] = d.text_v2.map(norm_key)
# 1. bỏ dòng rỗng, trùng nội bộ test-v2 giữ một bản nếu cùng nhãn, bỏ hết nếu mâu thuẫn
test_v2 = test_v2[test_v2.key.str.len() > 0]
conf = test_v2.groupby("key").label.nunique()
test_v2 = test_v2[~test_v2.key.isin(conf[conf > 1].index)].drop_duplicates("key")
# 2. test-v2 không được trùng dữ liệu phim
film_keys = set(film.text_v2.map(norm_key))
n0 = len(test_v2)
test_v2 = test_v2[~test_v2.key.isin(film_keys)]
report["test_v2_removed_film_overlap"] = n0 - len(test_v2)
# 3. pool không được trùng test-v2 (kể cả gần trùng, Jaccard 5-gram ký tự >= 0.8)
tk = set(test_v2.key)
post = defaultdict(list)
docs = [shingles(x) for x in test_v2.key]
for i, s_ in enumerate(docs):
    for g in s_:
        post[g].append(i)


def near(k):
    s_ = shingles(k)
    if len(s_) < 10:
        return False
    c = Counter(i for g in s_ for i in post.get(g, ())[:200])
    return any(len(s_ & docs[i]) / len(s_ | docs[i]) >= 0.8 for i, _ in c.most_common(5))


for name in ("pool_vsmec", "pool_vsfc"):
    d = pool_me if name == "pool_vsmec" else pool_fc
    n0 = len(d)
    d = d[(d.key.str.len() > 0) & (~d.key.isin(tk))].drop_duplicates("key")
    d = d[~d.key.map(near)]
    report[name + "_removed"] = n0 - len(d)
    if name == "pool_vsmec":
        pool_me = d
    else:
        pool_fc = d

test_v2 = test_v2.reset_index(drop=True)
test_v2.insert(0, "tv2_id", range(len(test_v2)))
test_v2[["tv2_id", "source", "Emotion", "label", "text_v2"]].to_csv(OUT / "test_v2.csv", index=False, encoding="utf-8-sig")
pool_me[["source", "Emotion", "label", "text_v2"]].to_csv(OUT / "pool_vsmec.csv", index=False, encoding="utf-8-sig")
pool_fc[["source", "label", "text_v2"]].to_csv(OUT / "pool_vsfc.csv", index=False, encoding="utf-8-sig")

aud = pd.read_csv(DS / "label_audit.csv", encoding="utf-8-sig", keep_default_na=False)
fix = aud[(aud.de_xuat == "đề xuất sửa") & (aud.split == "train")][["comment_id", "final_label", "nhan_de_xuat", "ly_do"]]
fix.to_csv(OUT / "label_fixes_train.csv", index=False, encoding="utf-8-sig")

man = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "mapping": EMOTION, "dropped_emotion": "Surprise", "report": report,
       "counts": {"test_v2": test_v2.label.value_counts().to_dict(), "pool_vsmec": pool_me.label.value_counts().to_dict(),
                  "pool_vsfc": pool_fc.label.value_counts().to_dict(), "film_dev": film[film.split != "test"].label.value_counts().to_dict()},
       "n": {"test_v2": len(test_v2), "pool_vsmec": len(pool_me), "pool_vsfc": len(pool_fc), "label_fixes": len(fix)},
       "sha256": {f.name: sha(f) for f in sorted(OUT.glob("*.csv"))}}
(OUT / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: man[k] for k in ("report", "counts", "n")}, ensure_ascii=False, indent=1))
print("xong", f"{time.time() - t0:.0f}s")
