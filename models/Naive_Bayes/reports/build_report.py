"""Sinh báo cáo Naive Bayes và kế hoạch dữ liệu từ artifact.

Chạy từ gốc repo:
    models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/reports/build_report.py

Đầu vào: CSV/JSON trong models/Naive_Bayes/experiments/, models/Naive_Bayes/datasets/splits/, models/Naive_Bayes/datasets/label_audit.csv.
Đầu ra (đều trong models/Naive_Bayes/): reports/Bao_cao_Cai_thien_Naive_Bayes.md, docs/data_collection_plan.md,
         reports/figures/nb_*.png, experiments/data_quality.json.

Mọi số liệu trong văn bản là token {{ten}} trong models/Naive_Bayes/reports/templates/, được thay bằng giá trị đọc từ artifact.
Script dừng nếu còn token chưa thay, link hình hỏng, hoặc template chứa số thập phân gõ tay.
"""
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "models" / "Naive_Bayes"
EXP = NB / "experiments"
FIG = NB / "reports" / "figures"
TPL = NB / "reports" / "templates"
LABELS = ["Negative", "Neutral", "Positive"]

# Bảng màu đã kiểm tra bằng validate_palette (3 slot đầu, mọi cặp, nền sáng)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8985", "#ecebe6", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": "#c9c8c2",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "axes.titlesize": 11, "axes.titlelocation": "left",
})

f4 = lambda x: f"{x:.4f}"
f3 = lambda x: f"{x:.3f}"
s4 = lambda x: f"{x:+.4f}"
pv = lambda x: "<0.001" if x < 0.001 else f"{x:.3f}"
pm = lambda m, s: f"{m:.4f} ± {s:.4f}"
ci = lambda a, b: f"[{a:.4f}, {b:.4f}]"
cis = lambda a, b: f"[{a:+.4f}, {b:+.4f}]"


def md_table(df, headers=None):
    cols = list(df.columns)
    head = headers or cols
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "\\|").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------- nạp artifact
res = pd.read_csv(EXP / "results_cv.csv").set_index("variant")
cmp_ = pd.read_csv(EXP / "comparisons_cv.csv")
alpha = pd.read_csv(EXP / "alpha_sensitivity.csv")
lc = pd.read_csv(EXP / "learning_curve.csv")
ext = pd.read_csv(EXP / "learning_curve_extrapolation.csv")
sel = pd.read_csv(EXP / "selected_configs.csv", low_memory=False)
test = pd.read_csv(EXP / "final_test_results.csv").set_index("model")
tcmp = pd.read_csv(EXP / "final_test_comparisons.csv")
tper = pd.read_csv(EXP / "final_test_per_class.csv")
cand = json.loads((EXP / "final_candidates.json").read_text(encoding="utf-8"))
cand_flags = pd.read_csv(EXP / "label_audit_candidates.csv")
err = pd.read_csv(EXP / "error_analysis_labeled.csv", encoding="utf-8-sig", keep_default_na=False)
audit = pd.read_csv(NB / "datasets" / "label_audit.csv", encoding="utf-8-sig", keep_default_na=False)
run_info = json.loads((EXP / "run_info.json").read_text(encoding="utf-8"))
test_info = json.loads((EXP / "final_test_run_info.json").read_text(encoding="utf-8"))
old_cmp = pd.read_csv(ROOT / "models" / "Naive_Bayes" / "comparison_val_test.csv").set_index("model")
old_info = json.loads((ROOT / "models" / "Naive_Bayes" / "nb_run_info.json").read_text(encoding="utf-8"))
nb_json = json.loads((ROOT / "models" / "Naive_Bayes" / "naive_bayes_improvement.ipynb").read_text(encoding="utf-8"))
nb_out = "\n".join("".join(o.get("text", [])) for c in nb_json["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if o.get("output_type") == "stream")
drops = re.findall(r"Số dòng bị bỏ mỗi outer fold: trung bình ([\d.]+) \| min (\d+) \| max (\d+)", nb_out)
assert len(drops) == 2


def cmp_row(variant, reference="R1"):
    r = cmp_[(cmp_.variant == variant) & (cmp_.reference == reference)]
    assert len(r) == 1, (variant, reference)
    return r.iloc[0]


# ---------------------------------------------------------------------- chất lượng dữ liệu (đọc lại từ nguồn)
def read_split(n):
    return pd.read_csv(ROOT / "models" / "Naive_Bayes" / "datasets" / "splits" / f"{n}.csv", encoding="utf-8-sig", keep_default_na=False)


splits = {n: read_split(n) for n in ("train", "val", "test")}
xl = pd.read_excel(ROOT / "Data" / "final_labeled_clean_v3.xlsx", keep_default_na=False,
                   usecols=["Comment ID", "RAW Text", "Clean Text", "Clean Text Old", "Final Label"]).set_index("Comment ID")
tr = splits["train"]
tr_raw = xl.loc[tr["Comment ID"], "RAW Text"].astype(str).values
tr_old = xl.loc[tr["Comment ID"], "Clean Text Old"].astype(str).values
tr_clean = tr["Clean Text"].astype(str).values
tr_y = tr["Final Label"].values
dev_ids = list(splits["train"]["Comment ID"]) + list(splits["val"]["Comment ID"])
dev_raw = [str(t).lower() for t in xl.loc[dev_ids, "RAW Text"]]


def count(texts, pat):
    return sum(1 for t in texts if re.search(pat, t.lower()))


def rate_in(texts, pat, y, lab):
    idx = [i for i, l in enumerate(y) if l == lab]
    return sum(1 for i in idx if re.search(pat, texts[i].lower())) / len(idx)


def post(texts, pat, y, lab):
    hit = [i for i, t in enumerate(texts) if re.search(pat, t.lower())]
    return sum(1 for i in hit if y[i] == lab) / max(len(hit), 1)


KO = r"\b(ko|k|kg|kh|hông|hong|khum|hok|khong|hem|kô)\b"
LOW = r"\b[0-5](\s*điểm|\s*/\s*10)"
HIGH = r"\b10\s*(điểm|đ\b|/\s*10)"
EMO = "[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F1E6-\U0001F1FF\u2B50]"
tr_raw_l, tr_clean_l = list(tr_raw), list(tr_clean)

groups = pd.concat(splits.values())["Clean Text"].value_counts()
dq = {
    "rows": int(sum(len(d) for d in splits.values())),
    "unique_text": int(pd.concat(splits.values())["Clean Text"].nunique()),
    "dup_groups": int((groups > 1).sum()),
    "dup_extra": int((groups - 1).sum()),
    "train_digit_clean": count(tr_clean_l, r"\d"),
    "train_qmark_clean": count(tr_clean_l, r"\?"),
    "train_nhung_clean": count(tr_clean_l, r"\bnhưng\b"),
    "train_nhung_old": count(list(tr_old), r"\bnhưng\b"),
    "train_qmark_raw": count(tr_raw_l, r"\?"),
    "train_qmark_pneu": post(tr_raw_l, r"\?", tr_y, "Neutral"),
    "train_emoji_raw": count(tr_raw_l, EMO),
    "train_emoji_ppos": post(tr_raw_l, EMO, tr_y, "Positive"),
    "train_low_raw": count(tr_raw_l, LOW),
    "train_low_pneg": post(tr_raw_l, LOW, tr_y, "Negative"),
    "train_high_raw": count(tr_raw_l, HIGH),
    "train_high_ppos": post(tr_raw_l, HIGH, tr_y, "Positive"),
    "train_ko_raw": count(tr_raw_l, KO),
    "train_ko_clean": count(tr_clean_l, KO),
    "train_khong_neg": rate_in(tr_clean_l, r"\bkhông\b", tr_y, "Negative"),
    "train_khong_neu": rate_in(tr_clean_l, r"\bkhông\b", tr_y, "Neutral"),
    "train_khong_pos": rate_in(tr_clean_l, r"\bkhông\b", tr_y, "Positive"),
    "base_neu": float((tr_y == "Neutral").mean()),
    "base_pos": float((tr_y == "Positive").mean()),
    "dev_rows": len(dev_raw),
    "dev_muado": count(dev_raw, r"mưa\s*đỏ"),
    "dev_trai_buon": count(dev_raw, r"\btbn\b|trại\s*buôn"),
    "dev_tdm": count(dev_raw, r"thi[eê]n đ[uư]ờng m[aá]u|\btđm\b|\btdm\b"),
    "dev_luadao": count(dev_raw, r"lừa đảo|buôn người|bị lừa|campuchia"),
    "dev_placeholder": int(sum(1 for i in dev_ids if xl.loc[i, "Clean Text"] == "đánh giá phim")),
}
sk = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr["Clean Text"])
X = sk.transform(tr["Clean Text"])
Xc = CountVectorizer(ngram_range=(1, 2), vocabulary=sk.vocabulary_).fit_transform(tr["Clean Text"])
V = X.shape[1]
for lab in LABELS:
    m = tr["Final Label"].values == lab
    dq[f"mass_{lab}"] = float(X[m].sum())
    dq[f"cmass_{lab}"] = float(Xc[m].sum())
dq["V"] = int(V)
dq["l1_mean"] = float(np.asarray(X.sum(1)).mean())
(EXP / "data_quality.json").write_text(json.dumps(dq, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------- bảng
rows = []
for n, d in splits.items():
    c = d["Final Label"].value_counts()
    rows.append(dict(s=n, r=len(d), u=d["Clean Text"].nunique(), neg=f"{c['Negative']} ({c['Negative'] / len(d):.1%})",
                     neu=f"{c['Neutral']} ({c['Neutral'] / len(d):.1%})", pos=f"{c['Positive']} ({c['Positive'] / len(d):.1%})"))
T_SPLIT = md_table(pd.DataFrame(rows), ["Tập", "Số dòng", "Số text khác nhau", "Negative", "Neutral", "Positive"])

a = alpha.copy()
a["cv"] = [pm(m, s) for m, s in zip(a.cv_macro_f1_mean, a.cv_macro_f1_std)]
a["val"] = a.val_macro_f1_old_grid.map(lambda x: "chưa chạy" if pd.isna(x) else f4(x))
T_ALPHA = md_table(a[["alpha", "cv", "val"]], ["$\\alpha$", "Macro-F1 CV (mean ± std, 5 lần lặp)", "Macro-F1 val (lưới cũ, 309 mẫu)"])

ref_names = {"Majority": "Majority", "R0": "R0: NB hiện tại, cố định", "SVC_C1": "LinearSVC $C=1$, cố định", "LogReg_C3": "LogReg $C=3$, cố định",
             "SVC_tuned": "LinearSVC, tune nested", "R1": "R1: NB, tune nested", "LogReg_tuned": "LogReg, tune nested"}


def cv_row(k, name):
    r = res.loc[k]
    return dict(n=name, f=pm(r.macro_f1_mean, r.macro_f1_std), c=ci(r.macro_f1_ci95_lo, r.macro_f1_ci95_hi), neg=f3(r.f1_negative),
                neu=f3(r.f1_neutral), pos=f3(r.f1_positive), acc=f3(r.accuracy))


T_REF = md_table(pd.DataFrame([cv_row(k, n) for k, n in ref_names.items()]),
                 ["Mô hình", "Macro-F1 (mean ± std)", "CI95 bootstrap", "F1 Negative", "F1 Neutral", "F1 Positive", "Accuracy"])

abl = [("E1", "R1", "R0", "Tune nested so với cố định"), ("E2a", "E2a_count", "R1", "Trọng số count"), ("E2b", "E2b_binary", "R1", "Trọng số binary"),
       ("E3a", "E3a_cnb", "R1", "ComplementNB"), ("E3b", "E3b_nblr", "R1", "NB-LR"), ("E3b*", "E3b_cw", "R1", "NB-LR có `class_weight`"),
       ("E4a", "E4a_old", "R1", "Đầu vào `Clean Text Old`"), ("E4b", "E4b_meta", "R1", "Token meta từ RAW"), ("E5", "E5_neg", "R1", "Phạm vi phủ định"),
       ("E6a", "E6a_char", "R1", "Char n-gram"), ("E6b", "E6b_union", "R1", "Word + char"), ("E6c", "E6c_select", "R1", "Chọn đặc trưng chi²/MI"),
       ("E7a", "E7a_balanced", "R1", "Class weight balanced"), ("E7b", "E7b_oversample", "R1", "Oversample"), ("E7c", "E7c_offset", "R1", "Offset log-prior"),
       ("E8a", "E8a_softvote", "R1", "Soft voting NB + LogReg"), ("E8b", "E8b_stacking", "R1", "Stacking NB + LogReg"),
       ("E9", "E9_filter", "R1", "Lọc nhãn, chỉ NB"), ("E9b*", "E9b_strict", "R1", "Lọc nhãn, NB và LogReg đồng ý"),
       ("E10", "E10_aug", "R1", "Augmentation"), ("X", "X_old_bal", "R1", "Thăm dò: E4a + E7a")]


def decision(r):
    if r.accepted:
        return "Chấp nhận"
    return "Kém hơn có ý nghĩa" if r.ci95_hi < 0 else "Không đủ bằng chứng"


def abl_row(eid, var, ref, name):
    q, r = res.loc[var], cmp_row(var, ref)
    return dict(i=eid, n=name, f=pm(q.macro_f1_mean, q.macro_f1_std), c=f"{f3(q.f1_negative)} / {f3(q.f1_neutral)} / {f3(q.f1_positive)}",
                d=s4(r.delta), dc=cis(r.ci95_lo, r.ci95_hi), p=pv(r.p_boot), h=pv(r.p_holm), k=decision(r))


abl_df = pd.DataFrame([abl_row(*x) for x in abl])
HDR = ["ID", "Biến thể", "Macro-F1 CV (mean ± std)", "F1 Neg / Neu / Pos", "$\\Delta$", "CI95 của $\\Delta$", "p", "p Holm", "Quyết định"]
T_ABL = md_table(abl_df, HDR)
EXP_ROW = {x[0]: md_table(pd.DataFrame([abl_row(*x)]), HDR) for x in abl}

t_names = {"R0_train": "R0, fit trên train (lịch sử)", "R0_refit": "R0, fit lại trên train+val", "NB_best": "NB tốt nhất theo quy tắc (R1)",
           "X_old_bal": "`X_old_bal` (thăm dò)", "LogReg": "LogReg", "Hybrid": "Soft voting NB + LogReg"}
rows = [dict(m=n, f=f4(test.loc[k, "macro_f1"]), c=ci(test.loc[k, "ci95_lo"], test.loc[k, "ci95_hi"]), neg=f3(test.loc[k, "f1_negative"]),
             neu=f3(test.loc[k, "f1_neutral"]), pos=f3(test.loc[k, "f1_positive"]), acc=f3(test.loc[k, "accuracy"])) for k, n in t_names.items()]
T_TEST = md_table(pd.DataFrame(rows), ["Mô hình", "Macro-F1 test", "CI95 bootstrap theo group", "F1 Negative", "F1 Neutral", "F1 Positive", "Accuracy"])
rows = [dict(a=f"{t_names[r.a]} so với {t_names[r.b]}", d=s4(r.delta), c=cis(r.ci95_lo, r.ci95_hi), p=pv(r.p_boot), h=pv(r.p_holm),
             s="Có" if r["significant_holm_0.05"] else "Không", n=f"{r.a_right_b_wrong} / {r.a_wrong_b_right}", m=pv(r.p_mcnemar_exact)) for _, r in tcmp.iterrows()]
T_TCMP = md_table(pd.DataFrame(rows), ["So sánh", "$\\Delta$ macro-F1", "CI95 của $\\Delta$", "p bootstrap", "p Holm", "Có ý nghĩa sau Holm", "A đúng B sai / A sai B đúng", "p McNemar (accuracy)"])
tp = tper[tper.model.isin(["R0_refit", "NB_best", "X_old_bal", "Hybrid"])].copy()
tp["model"] = tp.model.map(t_names)
for c_ in ("precision", "recall", "f1"):
    tp[c_] = tp[c_].map(f3)
T_TPER = md_table(tp[["model", "label", "precision", "recall", "f1", "support"]], ["Mô hình", "Lớp", "Precision", "Recall", "F1", "Support"])

cause_vi = {"nhan_mo_ho": "Nhãn mơ hồ hoặc có thể sai", "hon_hop": "Khen chê lẫn lộn", "so_sanh_phim_khac": "So sánh với phim khác", "thieu_ngu_canh": "Thiếu ngữ cảnh hội thoại",
            "tin_hieu_it": "Ngắn hoặc ít tín hiệu", "tranh_luan_doi_tuong": "Tranh luận với người khác, đối tượng không phải phim", "an_y_mia_mai": "Ẩn ý hoặc mỉa mai",
            "phu_dinh": "Phủ định", "ngoai_chu_de": "Ngoài chủ đề"}
vc = err.nguyen_nhan.value_counts()
T_ERR = md_table(pd.DataFrame([dict(n=cause_vi[k], c=int(v), p=f"{v / len(err):.0%}", e=err[err.nguyen_nhan == k].iloc[0].ghi_chu) for k, v in vc.items()]),
                 ["Nguyên nhân", "Số mẫu", "Tỉ lệ", "Ví dụ (ghi chú đọc tay)"])
conf = pd.read_csv(EXP / "confusion_R1_oof_r0.csv", index_col=0)
T_CONF = md_table(conf.reset_index().rename(columns={"index": "Nhãn thật \\ Dự đoán"}))

aud = audit[audit.de_xuat.isin(["đề xuất sửa", "mơ hồ - nhóm quyết"])]
T_AUD = md_table(aud[["comment_id", "clean_text_che_ten", "final_label", "de_xuat", "nhan_de_xuat", "ly_do"]].assign(clean_text_che_ten=lambda d: d.clean_text_che_ten.str.slice(0, 70)),
                 ["ID", "Clean Text (che tên, cắt ngắn)", "Nhãn hiện tại", "Đề xuất", "Nhãn đề xuất", "Lý do"])
T_AUD_COUNT = md_table(audit.groupby(["split", "de_xuat"]).size().reset_index(name="n"), ["Split", "Đề xuất", "Số dòng"])

lc_show = lc.copy()
lc_show["f"] = [pm(m, s) for m, s in zip(lc_show.macro_f1_mean, lc_show.macro_f1_std)]
lc_show["curve"] = lc_show.curve.map({"all": "Mọi lớp", "Negative": "Chỉ Negative", "Neutral": "Chỉ Neutral", "Positive": "Chỉ Positive"})
for c_ in ("n_train", "n_negative", "n_neutral", "n_positive"):
    lc_show[c_] = lc_show[c_].round().astype(int)
for c_ in ("f1_negative", "f1_neutral", "f1_positive"):
    lc_show[c_] = lc_show[c_].map(f3)
T_LC = md_table(lc_show[["curve", "frac", "n_train", "n_negative", "n_neutral", "n_positive", "f", "f1_negative", "f1_neutral", "f1_positive"]],
                ["Đường", "Tỉ lệ giữ", "$n$ train", "$n$ Neg", "$n$ Neu", "$n$ Pos", "Macro-F1 (mean ± std)", "F1 Neg", "F1 Neu", "F1 Pos"])
ex = ext[ext.n_added.isin([200, 400])].copy()
ex["curve"] = ex.curve.map({"all": "Mọi lớp", "Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive"})
ex["n_current"] = ex.n_current.round().astype(int)
ex["b"] = ex.b.map(f3)
ex["gain_pred"] = ex.gain_pred.map(lambda x: f"{x:+.3f}")
T_EXT = md_table(ex[["curve", "n_current", "n_added", "b", "gain_pred"]], ["Thêm mẫu vào", "$n$ hiện tại (outer-train)", "Số mẫu thêm", "Hệ số $b$", "Dự đoán tăng macro-F1"])

# ---------------------------------------------------------------------- hình
SHORT = {x[0]: x[3] for x in abl}


def fig_forest():
    d = cmp_[cmp_.reference == "R1"]
    d = d[d.experiment != "ref"]
    order = [x for x in abl if x[1] in set(d.variant)]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ys = np.arange(len(order))[::-1]
    for y, (eid, var, ref, name) in zip(ys, order):
        r = cmp_row(var, ref)
        col = BLUE if r.accepted else MUTED
        ax.plot([r.ci95_lo, r.ci95_hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.scatter(r.delta, y, s=52, color=col if r.accepted else SURF, edgecolor=col, linewidth=1.6, zorder=3)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ys, [f"{e}  {n.replace('`', '')}" for e, _, _, n in order])
    ax.set_xlabel("Δ macro-F1 so với R1 trên CV (điểm: Δ; thanh: CI95 bootstrap theo group)")
    ax.grid(axis="y", visible=False)
    h = [plt.Line2D([], [], color=BLUE, marker="o", lw=2, markersize=7, label="CI95 nằm trên 0"),
         plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURF, lw=2, markersize=7, label="CI95 chứa 0 hoặc nằm dưới 0")]
    ax.legend(handles=h, loc="lower left")
    ax.set_title("So sánh các biến thể với R1 trên tập phát triển")
    fig.tight_layout()
    fig.savefig(FIG / "nb_cv_forest.png", dpi=200)
    plt.close(fig)


def fig_alpha():
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    a_ = alpha
    ax.errorbar(a_.alpha, a_.cv_macro_f1_mean, a_.cv_macro_f1_std, color=BLUE, lw=2, marker="o", ms=6, capsize=3, label="CV (1800 mẫu, 25 fold)")
    b = a_.dropna(subset=["val_macro_f1_old_grid"])
    ax.plot(b.alpha, b.val_macro_f1_old_grid, color=ORANGE, lw=2, marker="s", ms=6, label="Val cũ (309 mẫu)")
    ax.set_xscale("log")
    ax.set_xticks(a_.alpha, [str(x) for x in a_.alpha])
    ax.minorticks_off()
    ax.set_xlabel("α (thang log)")
    ax.set_ylabel("Macro-F1")
    ax.legend(loc="lower left")
    best_cv = a_.alpha[a_.cv_macro_f1_mean.idxmax()]
    best_val = b.alpha[b.val_macro_f1_old_grid.idxmax()]
    ax.set_title(f"Val cũ đạt đỉnh ở α = {best_val}, CV đạt đỉnh ở α = {best_cv}")
    fig.tight_layout()
    fig.savefig(FIG / "nb_alpha_sensitivity.png", dpi=200)
    plt.close(fig)


def fig_lc():
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    d = lc[lc.curve == "all"]
    axs[0].errorbar(d.n_train, d.macro_f1_mean, d.macro_f1_std, color=BLUE, lw=2, marker="o", ms=6, capsize=3)
    axs[0].set_xlabel("Tổng số mẫu trong outer-train")
    axs[0].set_ylabel("Macro-F1 trên outer-test")
    axs[0].set_title("Giảm mọi lớp theo cùng tỉ lệ")
    for lab, col, mk in zip(LABELS, (BLUE, ORANGE, AQUA), ("o", "s", "^")):
        d = lc[lc.curve == lab]
        x = d[f"n_{lab.lower()}"]
        axs[1].errorbar(x, d.macro_f1_mean, d.macro_f1_std, color=col, lw=2, marker=mk, ms=6, capsize=3, label=lab)
        axs[1].annotate(lab, (x.iloc[0], d.macro_f1_mean.iloc[0]), textcoords="offset points", xytext=(4, -13), color=INK2, fontsize=9)
    axs[1].set_xscale("log")
    ticks = [64, 128, 256, 512, 873]
    axs[1].set_xticks(ticks, [str(t) for t in ticks])
    axs[1].minorticks_off()
    axs[1].set_xlabel("Số mẫu của lớp được giảm (thang log)")
    axs[1].set_title("Chỉ giảm một lớp, giữ nguyên hai lớp còn lại")
    axs[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "nb_learning_curve.png", dpi=200)
    plt.close(fig)


def fig_conf():
    names = [("R0_train", "R0, fit trên train"), ("NB_best", "NB tốt nhất (R1)"), ("X_old_bal", "X_old_bal"), ("Hybrid", "Soft voting NB + LogReg")]
    fig, axs = plt.subplots(1, 4, figsize=(11.4, 3.1), sharey=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f2f6fc", "#a9c9ef", BLUE, "#123f78"])
    for ax, (k, t) in zip(axs, names):
        cm = pd.read_csv(EXP / f"final_test_confusion_{k}.csv", index_col=0).values
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(3), ["Neg", "Neu", "Pos"])
        ax.set_yticks(range(3), ["Neg", "Neu", "Pos"])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Dự đoán")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > 0.6 else INK, fontsize=10)
    axs[0].set_ylabel("Nhãn thật")
    fig.savefig(FIG / "nb_test_confusion.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig_cv_test():
    m = [("R0", "R0_refit", "R0"), ("R1", "NB_best", "NB tốt nhất (R1)"), ("X_old_bal", "X_old_bal", "X_old_bal"),
         ("LogReg_tuned", "LogReg", "LogReg"), ("E8a_softvote", "Hybrid", "Soft voting")]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ys = np.arange(len(m))[::-1]
    for y, (c_, t, _) in zip(ys, m):
        r = res.loc[c_]
        ax.plot([r.macro_f1_ci95_lo, r.macro_f1_ci95_hi], [y + 0.14] * 2, color=BLUE, lw=2)
        ax.scatter(r.macro_f1_mean, y + 0.14, s=46, color=BLUE, zorder=3)
        q = test.loc[t]
        ax.plot([q.ci95_lo, q.ci95_hi], [y - 0.14] * 2, color=ORANGE, lw=2)
        ax.scatter(q.macro_f1, y - 0.14, s=46, color=ORANGE, marker="s", zorder=3)
    ax.set_yticks(ys, [x[2] for x in m])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Macro-F1 (điểm: ước lượng; thanh: CI95 bootstrap theo group)")
    h = [plt.Line2D([], [], color=BLUE, marker="o", lw=2, label="CV (dev, nested)"), plt.Line2D([], [], color=ORANGE, marker="s", lw=2, label="Test (fit trên train+val)")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.06))
    hw_t = (test.loc["NB_best", "ci95_hi"] - test.loc["NB_best", "ci95_lo"]) / 2
    hw_c = (res.loc["R1", "macro_f1_ci95_hi"] - res.loc["R1", "macro_f1_ci95_lo"]) / 2
    ax.set_title(f"CI95 trên test rộng gấp khoảng {hw_t / hw_c:.0f} lần CV")
    fig.tight_layout()
    fig.savefig(FIG / "nb_cv_vs_test.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


for fn in (fig_forest, fig_alpha, fig_lc, fig_conf, fig_cv_test):
    fn()

# ---------------------------------------------------------------------- token
g = lambda variant, n: ext[(ext.curve == variant) & (ext.n_added == n)].iloc[0].gain_pred
hw = (test.loc["R0_train", "ci95_hi"] - test.loc["R0_train", "ci95_lo"]) / 2
n_test = int(test.loc["R0_train", "n_test"])
n_for = lambda w: int(math.ceil(n_test * (hw / w) ** 2))
r1s = sel[sel.variant == "R1"]
mode = r1s[["alpha", "min_df", "max_order"]].value_counts()
alpha_counts = r1s["alpha"].value_counts()
sv = sel[sel.variant == "E8a_softvote"]["w_nb"].value_counts()
c = {r.a + "|" + r.b: r for _, r in tcmp.iterrows()}
T = {
    "T_SPLIT": T_SPLIT, "T_ALPHA": T_ALPHA, "T_REF": T_REF, "T_ABL": T_ABL, "T_TEST": T_TEST, "T_TCMP": T_TCMP, "T_TPER": T_TPER, "T_ERR": T_ERR,
    "T_CONF": T_CONF, "T_AUD": T_AUD, "T_AUD_COUNT": T_AUD_COUNT, "T_LC": T_LC, "T_EXT": T_EXT,
    "r0_cv": f4(res.loc["R0", "macro_f1_mean"]), "r1_cv": f4(res.loc["R1", "macro_f1_mean"]), "lr_cv": f4(res.loc["LogReg_tuned", "macro_f1_mean"]),
    "lrc3_cv": f4(res.loc["LogReg_C3", "macro_f1_mean"]), "svc1_cv": f4(res.loc["SVC_C1", "macro_f1_mean"]),
    "sv_cv": f4(res.loc["E8a_softvote", "macro_f1_mean"]), "x_cv": f4(res.loc["X_old_bal", "macro_f1_mean"]), "maj_cv": f4(res.loc["Majority", "macro_f1_mean"]),
    "nblr_cv": f4(res.loc["E3b_nblr", "macro_f1_mean"]),
    "e1_d": s4(cmp_row("R1", "R0").delta), "e1_ci": cis(cmp_row("R1", "R0").ci95_lo, cmp_row("R1", "R0").ci95_hi),
    "lr_r1_d": s4(cmp_row("LogReg_tuned", "R1").delta), "lr_r1_ci": cis(cmp_row("LogReg_tuned", "R1").ci95_lo, cmp_row("LogReg_tuned", "R1").ci95_hi),
    "sv_d": s4(cmp_row("E8a_softvote").delta), "sv_ci": cis(cmp_row("E8a_softvote").ci95_lo, cmp_row("E8a_softvote").ci95_hi),
    "sv_lr_d": s4(cmp_row("E8a_softvote", "LogReg_tuned").delta), "sv_lr_ci": cis(cmp_row("E8a_softvote", "LogReg_tuned").ci95_lo, cmp_row("E8a_softvote", "LogReg_tuned").ci95_hi),
    "x_d": s4(cmp_row("X_old_bal").delta), "x_ci": cis(cmp_row("X_old_bal").ci95_lo, cmp_row("X_old_bal").ci95_hi),
    "e4a_d": s4(cmp_row("E4a_old").delta), "e7a_d": s4(cmp_row("E7a_balanced").delta),
    "cnb_d": s4(cmp_row("E3a_cnb").delta), "e9_d": s4(cmp_row("E9_filter").delta), "e9b_d": s4(cmp_row("E9b_strict").delta),
    "e9_mean": f"{float(drops[0][0]):.0f}", "e9_min": drops[0][1], "e9_max": drops[0][2],
    "e9b_mean": f"{float(drops[1][0]):.0f}", "e9b_min": drops[1][1], "e9b_max": drops[1][2],
    "e9_pct": f"{float(drops[0][0]) / 1440:.0%}", "e9b_pct": f"{float(drops[1][0]) / 1440:.1%}",
    "flag_nb": int(cand_flags.flag_nb.sum()), "flag_lr": int(cand_flags.flag_lr.sum()), "flag_both": int((cand_flags.flag_nb & cand_flags.flag_lr).sum()),
    "aud_keep": int((audit.de_xuat == "giữ").sum()), "aud_fix": int((audit.de_xuat == "đề xuất sửa").sum()), "aud_amb": int((audit.de_xuat == "mơ hồ - nhóm quyết").sum()),
    "aud_n": len(audit[audit.split != "test"]),
    "t_r0": f4(test.loc["R0_train", "macro_f1"]), "t_r0ci": ci(test.loc["R0_train", "ci95_lo"], test.loc["R0_train", "ci95_hi"]),
    "t_r0r": f4(test.loc["R0_refit", "macro_f1"]), "t_nb": f4(test.loc["NB_best", "macro_f1"]), "t_nbci": ci(test.loc["NB_best", "ci95_lo"], test.loc["NB_best", "ci95_hi"]),
    "t_lr": f4(test.loc["LogReg", "macro_f1"]), "t_sv": f4(test.loc["Hybrid", "macro_f1"]), "t_svci": ci(test.loc["Hybrid", "ci95_lo"], test.loc["Hybrid", "ci95_hi"]),
    "t_x": f4(test.loc["X_old_bal", "macro_f1"]), "t_xci": ci(test.loc["X_old_bal", "ci95_lo"], test.loc["X_old_bal", "ci95_hi"]),
    "t_sv_nb_d": s4(c["Hybrid|NB_best"].delta), "t_sv_nb_ci": cis(c["Hybrid|NB_best"].ci95_lo, c["Hybrid|NB_best"].ci95_hi), "t_sv_nb_p": pv(c["Hybrid|NB_best"].p_holm),
    "t_nb_r0_d": s4(c["NB_best|R0_refit"].delta), "t_nb_r0_ci": cis(c["NB_best|R0_refit"].ci95_lo, c["NB_best|R0_refit"].ci95_hi), "t_nb_r0_p": pv(c["NB_best|R0_refit"].p_holm),
    "t_nb_lr_d": s4(c["NB_best|LogReg"].delta), "t_nb_lr_ci": cis(c["NB_best|LogReg"].ci95_lo, c["NB_best|LogReg"].ci95_hi), "t_nb_lr_p": pv(c["NB_best|LogReg"].p_holm),
    "t_x_nb_d": s4(c["X_old_bal|NB_best"].delta), "t_x_nb_ci": cis(c["X_old_bal|NB_best"].ci95_lo, c["X_old_bal|NB_best"].ci95_hi), "t_x_nb_p": pv(c["X_old_bal|NB_best"].p_holm),
    "t_refit_gain": s4(test.loc["R0_refit", "macro_f1"] - test.loc["R0_train", "macro_f1"]),
    "t_ci_hw": f"{hw:.3f}", "ci_ratio": f"{hw / ((res.loc['R1', 'macro_f1_ci95_hi'] - res.loc['R1', 'macro_f1_ci95_lo']) / 2):.1f}", "cv_ci_hw": f"{(res.loc['R1', 'macro_f1_ci95_hi'] - res.loc['R1', 'macro_f1_ci95_lo']) / 2:.3f}",
    "v_r0": f4(old_info["val"]["macro_f1"]), "old_lr_test": f4(old_cmp.loc["LogReg C=3", "test_macro_f1"]), "old_svc_test": f4(old_cmp.loc["LinearSVC C=1", "test_macro_f1"]),
    "old_maj_test": f4(old_cmp.loc["Majority (baseline)", "test_macro_f1"]),
    "alpha_best_cv": f4(alpha.cv_macro_f1_mean.max()), "alpha_001_cv": f4(alpha.set_index("alpha").loc[0.01, "cv_macro_f1_mean"]),
    "alpha_1_cv": f4(alpha.set_index("alpha").loc[1.0, "cv_macro_f1_mean"]), "alpha_1_val": f4(alpha.set_index("alpha").loc[1.0, "val_macro_f1_old_grid"]),
    "alpha_01_val": f4(alpha.set_index("alpha").loc[0.1, "val_macro_f1_old_grid"]), "alpha_001_val": f4(alpha.set_index("alpha").loc[0.01, "val_macro_f1_old_grid"]),
    "a01_n": int(alpha_counts.get(0.1, 0)), "a03_n": int(alpha_counts.get(0.3, 0)), "a003_n": int(alpha_counts.get(0.03, 0)),
    "sv_w25": int(sv.get(0.25, 0)), "sv_w50": int(sv.get(0.5, 0)), "sv_w0": int(sv.get(0.0, 0)), "sv_w75": int(sv.get(0.75, 0)),
    "nb_a": cand["NB_best"]["alpha"], "nb_md": cand["NB_best"]["min_df"], "lr_c": cand["LogReg"]["C"], "lr_md": cand["LogReg"]["min_df"], "hy_w": cand["Hybrid"]["w_nb"],
    "x_md": cand["Exploratory_X_old_bal"]["min_df"], "x_a": cand["Exploratory_X_old_bal"]["alpha"],
    "ms_neg": f"{dq['mass_Negative']:.0f}", "ms_neu": f"{dq['mass_Neutral']:.0f}", "ms_pos": f"{dq['mass_Positive']:.0f}",
    "av_neg": f"{V / dq['mass_Negative']:.2f}", "av_neu": f"{V / dq['mass_Neutral']:.2f}", "av_pos": f"{V / dq['mass_Positive']:.2f}",
    "ac_neg": f"{V / dq['cmass_Negative']:.2f}", "ac_pos": f"{V / dq['cmass_Positive']:.2f}",
    "ratio": f"{dq['cmass_Negative'] / dq['mass_Negative']:.1f}", "alpha_eq": f"{0.01 * dq['cmass_Negative'] / dq['mass_Negative']:.2f}",
    "l1": f"{dq['l1_mean']:.2f}", "V": V, "th0": f"{0.1 / (dq['mass_Negative'] + 0.1 * V):.2e}", "th2": f"{2.1 / (dq['mass_Negative'] + 0.1 * V):.2e}",
    "tot_mass": f"{dq['mass_Negative'] + 0.1 * V:.0f}", "aV": f"{0.1 * V:.0f}",
    "d_rows": dq["rows"], "d_uniq": dq["unique_text"], "d_grp": dq["dup_groups"], "d_extra": dq["dup_extra"],
    "q_raw": dq["train_qmark_raw"], "q_pneu": f"{dq['train_qmark_pneu']:.2f}", "base_neu": f"{dq['base_neu']:.3f}", "base_pos": f"{dq['base_pos']:.3f}",
    "em_raw": dq["train_emoji_raw"], "em_ppos": f"{dq['train_emoji_ppos']:.2f}", "lo_raw": dq["train_low_raw"], "lo_pneg": f"{dq['train_low_pneg']:.2f}",
    "hi_raw": dq["train_high_raw"], "hi_ppos": f"{dq['train_high_ppos']:.2f}", "ko_raw": dq["train_ko_raw"], "ko_clean": dq["train_ko_clean"],
    "nh_old": dq["train_nhung_old"], "kh_neg": f"{dq['train_khong_neg']:.1%}", "kh_neu": f"{dq['train_khong_neu']:.1%}", "kh_pos": f"{dq['train_khong_pos']:.1%}",
    "dev_rows": dq["dev_rows"], "m_md": dq["dev_muado"], "m_tb": dq["dev_trai_buon"], "m_tdm": dq["dev_tdm"], "m_ld": dq["dev_luadao"], "m_ph": dq["dev_placeholder"],
    "g_neg_200": f"{g('Negative', 200):+.3f}", "g_neu_200": f"{g('Neutral', 200):+.3f}", "g_pos_200": f"{g('Positive', 200):+.3f}",
    "g_neg_400": f"{g('Negative', 400):+.3f}", "g_neu_400": f"{g('Neutral', 400):+.3f}", "g_all_400": f"{g('all', 400):+.3f}",
    "n_ci_05": n_for(0.05), "n_ci_04": n_for(0.04), "n_ci_03": n_for(0.03),
    "e10_n": re.search(r"Số bản sao: (\d+)", nb_out).group(1),
    "py": run_info["python"], "torch": run_info["torch"], "sk": run_info["sklearn"], "np": run_info["numpy"], "sp": run_info["scipy"], "pd": run_info["pandas"],
    "gpu": run_info["gpu"], "commit": run_info["git_commit"][:7], "job": run_info["slurm_job_id"], "cand_sha": test_info["candidates_sha256"][:16], "scored_at": test_info["scored_at"],
    "boot_b": run_info["bootstrap_B"], "boot_t": test_info["bootstrap_B"], "n_groups": test_info["n_groups_test"], "n_test": test_info["n_test"],
    "n_groups_dev": run_info["n_groups_dev"], "runtime": f"{res.runtime_s.sum() / 60:.1f}",
}
for eid, tbl in EXP_ROW.items():
    T["R_" + eid.replace("*", "s")] = tbl

# ---------------------------------------------------------------------- render + tự kiểm tra
ALLOWED = {"0.001", "0.003"}
GEN_NOTE = "<!-- Tệp này được sinh tự động bởi models/Naive_Bayes/reports/build_report.py từ artifact. Sửa template trong reports/templates/, không sửa trực tiếp. -->\n\n"


def render(tpl_name, out):
    src = (TPL / tpl_name).read_text(encoding="utf-8")
    stray = set(re.findall(r"(?<![\w.])\d\.\d{3,}(?![\w])", re.sub(r"\{\{\w+\}\}", "", src))) - ALLOWED
    assert not stray, f"Số gõ tay trong {tpl_name}: {stray}"
    txt = re.sub(r"\{\{(\w+)\}\}", lambda m: str(T[m.group(1)]), src)
    assert "{{" not in txt, "còn token chưa thay"
    low = txt.lower()
    assert "todo" not in low and "placeholder" not in low.replace("chuỗi giữ chỗ", "") and "tbd" not in low, f"còn dấu chưa hoàn thiện trong {out.name}"
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", txt):
        p = (out.parent / link).resolve()
        assert p.exists(), f"hình không tồn tại: {link}"
    for frag in re.findall(r"\]\(#([^)]+)\)", txt):
        heads = [re.sub(r"[^\w\- ]", "", h).strip().lower().replace(" ", "-") for h in re.findall(r"^#+\s+(.+)$", txt, flags=re.M)]
        assert frag in heads, f"link mục lục hỏng: {frag}"
    out.parent.mkdir(exist_ok=True)
    out.write_text(GEN_NOTE + txt, encoding="utf-8")
    print("Đã ghi", out.relative_to(ROOT), len(txt.split()), "từ")


render("Bao_cao_Cai_thien_Naive_Bayes.template.md", NB / "reports" / "Bao_cao_Cai_thien_Naive_Bayes.md")
render("data_collection_plan.template.md", NB / "docs" / "data_collection_plan.md")
for f in ("nb_cv_forest", "nb_alpha_sensitivity", "nb_learning_curve", "nb_test_confusion", "nb_cv_vs_test"):
    assert (FIG / f"{f}.png").stat().st_size > 10000
print("Tự kiểm tra: token, link hình, mục lục, số gõ tay đều đạt")
