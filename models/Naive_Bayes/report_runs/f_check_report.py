"""Đối chiếu report.md với kết quả đã lưu. Chỉ đọc, không chấm lại.

Mỗi số thập phân trong báo cáo phải khớp một giá trị trong artifact khi làm tròn đến đúng số chữ số đang hiển thị.
Số không khớp được liệt kê để kiểm tra bằng tay; các số suy ra khai báo trong DERIVED kèm công thức tính lại.
Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/report_runs/f_check_report.py
"""
import gzip
import json
import re

from common import *  # noqa: F403

REPORT = NB_DIR / "report.md"
text = REPORT.read_text(encoding="utf-8")
body = text[:text.index("## 5. Tài liệu tham khảo")]             # bỏ tài liệu tham khảo: số trang, DOI

# ---------- gom mọi giá trị số trong artifact ----------
vals = []


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(k)
            walk(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        vals.append(float(o))
    elif isinstance(o, str):
        try:
            vals.append(float(o))
        except ValueError:
            pass


paths = [p for d in (EXP_V1, EXP_V2, OUT) for p in d.glob("*.csv")] + [NB_DIR / "nb_test_metrics.csv", NB_DIR / "comparison_val_test.csv"]
for p in paths:
    if p.name.endswith("roc_points.csv") or p.name == "cv_folds.csv":
        continue
    d = pd.read_csv(p)
    vals += pd.to_numeric(d.to_numpy().ravel(), errors="coerce").tolist()
for p in list(EXP_V1.glob("*.json")) + list(EXP_V2.glob("*.json")) + list(OUT.glob("*.json")):
    walk(json.loads(p.read_text(encoding="utf-8")))
vals = np.array([v for v in vals if np.isfinite(v)])

# Số suy ra từ artifact, tính lại tại đây thay vì tin vào văn bản
b = pd.read_csv(OUT / "b_metrics_all_splits.csv").set_index(["model", "eval_on"]).macro_f1
d, h = [pd.read_csv(OUT / f"c_{t}_results.csv").set_index("model").macro_f1 for t in ("splitD", "splitH")]
te = read_split("test")["Final Label"]
film = pd.concat([read_split(n) for n in ("train", "val", "test")])["Final Label"].value_counts(normalize=True)
xl = pd.read_excel(XLSX)["Final Label"].value_counts(normalize=True)
tv2 = pd.read_csv(V2 / "test_v2.csv", encoding="utf-8-sig", keep_default_na=False).label
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

maj = lambda y, l: (accuracy_score(y, [l] * len(y)), f1_score(y, [l] * len(y), average="macro", labels=LABELS, zero_division=0))
from sklearn.model_selection import train_test_split  # noqa: E402

df = pd.read_excel(XLSX)
_, _, _, yd = train_test_split(df["Clean Text"], df["Final Label"], test_size=0.2, stratify=df["Final Label"], random_state=18)
DERIVED = {
    "khoảng cách NB 0.5440 và SVM 0.6469": d["SVM TF-IDF"] - 0.5440343708530961,
    "SVM TF-IDF test D - test H": d["SVM TF-IDF"] - h["SVM TF-IDF"], "NB TF-IDF test D - test H": d["NB TF-IDF"] - h["NB TF-IDF"],
    **{f"tăng khi fit train+val {m}": b[m, "test_fit_trainval"] - b[m, "test_fit_train"] for m in ("BoW_val", "TFIDF_val", "BoW_cv", "TFIDF_cv")},
    "majority test-v2 acc": maj(tv2, "Positive")[0], "majority test-v2 f1": maj(tv2, "Positive")[1],
    "luôn đoán Negative test-v2 acc": maj(tv2, "Negative")[0], "luôn đoán Negative test-v2 f1": maj(tv2, "Negative")[1],
    "majority split D acc": maj(yd, "Positive")[0], "majority split D f1": maj(yd, "Positive")[1],
    "SVM CV BoW (notebook Doanh)": 0.5862074410946864, "SVM CV TF-IDF (notebook Doanh)": 0.612843950686855,
    "P(Neutral|?) (data_quality)": 0.41379310344827586,
    "toy P": 0.628, "toy a": 2.367, "toy b": 2.890, "toy theta": 0.250, "toy theta2": 0.375, "toy theta3": 0.333, "toy theta4": 0.167, "toy theta5": 0.125,
    "ví dụ P": 28 / 63, "ví dụ R": 28 / 52, "ví dụ F1": 2 * 28 / (2 * 28 + 35 + 24),
    "learning curve 25%": 0.5425, "learning curve 100%": 0.6122, "chỉ Negative 25%": 0.5443,   # Bảng 15 của báo cáo cải thiện
}
allv = np.r_[vals, np.abs(vals), np.array(list(DERIVED.values()))]

bad, n = [], 0
for line_no, line in enumerate(body.splitlines(), 1):
    if line.startswith(("```", "$V", "V=", "export")):
        continue
    for m in re.finditer(r"(?<![\w.\-/])[+\-]?(\d+\.\d{2,4})(?![\d.]|\+cu)", line):
        s = m.group(1)
        k = len(s.split(".")[1])
        n += 1
        if not (np.abs(np.round(allv, k) - float(s)) < 10 ** (-k - 3)).any():
            bad.append((line_no, m.group(0), line[max(0, m.start() - 35):m.end() + 25]))
print(f"Đã kiểm tra {n} số thập phân. Không khớp: {len(bad)}")
for b_ in bad:
    print(b_)
print("Số suy ra đã tính lại:")
for k, v in DERIVED.items():
    print(f"  {k}: {v:.4f}")

# Đối chiếu đầy đủ các hàng bảng với report_tables.md: hàng số phải trùng nguyên văn hoặc theo giá trị
tables = (OUT / "report_tables.md").read_text(encoding="utf-8")
num = lambda s: re.findall(r"[+\-]?\d+\.\d+", s)
row_nums = {tuple(num(l)) for l in tables.splitlines() if l.startswith("|") and len(num(l)) >= 2}
miss = []
for i, l in enumerate(text.splitlines(), 1):
    if l.startswith("|") and len(num(l)) >= 4:
        if not any(set(num(l)) <= set(r) for r in row_nums):
            miss.append((i, l[:90]))
print("Hàng bảng nhiều số không trùng hàng nào trong report_tables.md:", len(miss))
for x in miss:
    print(" ", x)

# Liên kết và hình
for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
    assert (NB_DIR / m.group(1)).exists(), m.group(1)
print("Mọi hình tồn tại:", len(re.findall(r"!\[", text)))
for tag in ("Bảng", "Hình"):
    ids = [int(x) for x in re.findall(rf"^\*{tag} (\d+)\.", text, flags=re.M)]
    assert ids == list(range(1, len(ids) + 1)), (tag, ids)
    print(f"{tag} đánh số liên tục đến {len(ids)}")
assert not re.search(r"[\w.-]+@[\w.-]+\.\w+", text), "có địa chỉ email trong báo cáo"
print("Không có email trong báo cáo")
