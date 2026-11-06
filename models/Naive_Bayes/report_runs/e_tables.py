"""Sinh bảng Markdown cho report.md từ CSV đã lưu, để không phải chép số bằng tay.

Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/report_runs/e_tables.py
Đầu ra: experiments_report/report_tables.md, cv_v2_vs_r0.csv
"""
import json

from common import *  # noqa: F403

md = []


def add(title, df=None, text=None):
    md.append(f"\n### {title}\n")
    if text:
        md.append(text + "\n")
    if df is not None:
        md.append("| " + " | ".join(map(str, df.columns)) + " |")
        md.append("|" + "---|" * len(df.columns))
        for r in df.itertuples(index=False):
            md.append("| " + " | ".join(map(str, r)) + " |")


f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"
sg = lambda x: f"{x:+.4f}"
ci = lambda a, b: f"[{a:.4f}, {b:.4f}]"
cis = lambda a, b: f"[{a:+.4f}, {b:+.4f}]"
pv = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"

# ---------- B ----------
b = pd.read_csv(OUT / "b_metrics_all_splits.csv")
cfg = json.loads((OUT / "b_configs.json").read_text(encoding="utf-8"))["configs"]
NAME = {"BoW_val": "BoW, chọn theo val", "TFIDF_val": "TF-IDF, chọn theo val (R0)", "BoW_cv": "BoW, chọn theo CV", "TFIDF_cv": "TF-IDF, chọn theo CV (R1)"}
rows = []
for m in NAME:
    c = cfg[m]
    nf = b[(b.model == m) & (b.eval_on == "train")].n_features.iloc[0]
    nf2 = b[(b.model == m) & (b.eval_on == "test_fit_trainval")].n_features.iloc[0]
    rows.append({"Mô hình": NAME[m], "n-gram": f"(1, {c['max_order']})", "min_df": c["min_df"], "α": c["alpha"], "fit_prior": c["fit_prior"], "Số đặc trưng (train / train+val)": f"{nf} / {nf2}"})
add("T_cfg: cấu hình", pd.DataFrame(rows))

rows = []
for m in NAME:
    g = b[b.model == m].set_index("eval_on")
    rows.append({"Mô hình": NAME[m], **{lab: f"{g.macro_f1[k]:.4f} / {g.accuracy[k]:.4f}" for lab, k in (("Train", "train"), ("Val", "val"), ("Test, fit trên train", "test_fit_train"), ("Test, fit trên train+val", "test_fit_trainval"))}})
add("T_tvt: macro-F1 / accuracy", pd.DataFrame(rows))

rows = []
for m in NAME:
    r = b[(b.model == m) & (b.eval_on == "test_fit_trainval")].iloc[0]
    rows.append({"Mô hình": NAME[m], "Accuracy": f4(r.accuracy), "Macro-P": f4(r.macro_precision), "Macro-R": f4(r.macro_recall), "Macro-F1": f4(r.macro_f1)})
add("T_b_test", pd.DataFrame(rows))
add("T_b_cmp", pd.read_csv(OUT / "b_test_bow_vs_tfidf.csv").round(4))

# ---------- bảng test phim tổng hợp ----------
v1r, v1p = pd.read_csv(EXP_V1 / "final_test_results.csv").set_index("model"), pd.read_csv(EXP_V1 / "final_test_per_class.csv")
film = pd.read_csv(EXP_V2 / "final_film_v2_results.csv").set_index("model")
bc = pd.read_csv(OUT / "b_test_ci.csv").set_index("model")
rows = [{"Mô hình": "Luôn đoán Positive", "Accuracy": "0.6585", "Macro-P": "", "Macro-R": "", "Macro-F1": "0.2647", "CI95": ""}]


def r1(name, acc, p, r, f, lo, hi):
    rows.append({"Mô hình": name, "Accuracy": f4(acc), "Macro-P": f4(p), "Macro-R": f4(r), "Macro-F1": f4(f), "CI95": ci(lo, hi)})


for key, name in (("R0_train", "R0, fit trên train"), ("R0_refit", "R0, fit trên train+val"), ("NB_best", "R1, fit trên train+val")):
    q = v1p[v1p.model == key]
    x = v1r.loc[key]
    r1(name, x.accuracy, q.precision.mean(), q.recall.mean(), x.macro_f1, x.ci95_lo, x.ci95_hi)
for key, name in (("BoW_val|trainval", "BoW chọn theo val, fit trên train+val"), ("BoW_cv|trainval", "BoW chọn theo CV, fit trên train+val")):
    x = bc.loc[key]
    r1(name, x.accuracy, x.macro_precision, x.macro_recall, x.macro_f1, x.ci95_lo, x.ci95_hi)
q = v1p[v1p.model == "X_old_bal"]
x = v1r.loc["X_old_bal"]
r1("`X_old_bal`, fit trên train+val", x.accuracy, q.precision.mean(), q.recall.mean(), x.macro_f1, x.ci95_lo, x.ci95_hi)
for key, name in (("R0v2", "R0v2, fit trên train+val"), ("NB_v2", "**NB cuối (V2)**, fit trên train+val")):
    x = film.loc[key]
    r1(name, x.accuracy, x.macro_precision, x.macro_recall, x.macro_f1, x.ci95_lo, x.ci95_hi)
add("T_film", pd.DataFrame(rows))

pc = pd.read_csv(EXP_V2 / "final_film_v2_per_class.csv")
pc = pc[pc.model == "NB_v2"]
add("T_final_perclass", pd.DataFrame([{"Lớp": x.label, "Precision": f3(x.precision), "Recall": f3(x.recall), "F1": f3(x.f1), "Support": x.support} for x in pc.itertuples()]))

# ---------- CV ----------
v1c, v2c = pd.read_csv(EXP_V1 / "comparisons_cv.csv"), pd.read_csv(EXP_V2 / "comparisons_cv.csv")
v1res, v2res = pd.read_csv(EXP_V1 / "results_cv.csv").set_index("variant"), pd.read_csv(EXP_V2 / "results_cv.csv").set_index("variant")
LBL = {"E1": "Tune nested CV (R1)", "E2a": "Trọng số count (BoW)", "E2b": "Trọng số binary", "E3a": "ComplementNB", "E3b": "NB-LR", "E4a": "Đầu vào `Clean Text Old`", "E4b": "Thêm token meta từ RAW",
       "E5": "Đánh dấu phủ định", "E6a": "Char n-gram", "E6b": "Word + char n-gram", "E6c": "Chọn đặc trưng chi²/MI", "E7a": "Class weight balanced", "E7b": "Oversample", "E7c": "Offset log-prior",
       "E9": "Lọc nhãn nghi nhiễu", "E10": "Augmentation", "V1": "Làm sạch lại `clean_v2`", "V2": "`clean_v2` + class weight balanced", "V3a": "V2 + pool UIT-VSMEC", "V3b": "V2 + pool UIT-VSFC",
       "V3c": "V2 + cả hai pool", "V4": "V2 + 8 nhãn sửa"}
rows = []
for src, res in ((v1c, v1res), (v2c, v2res)):
    for r in src.itertuples():
        if r.experiment not in LBL:
            continue
        res_row = res.loc[r.variant]
        dec = "Chấp nhận" if r.ci95_lo > 0 else ("Kém hơn" if r.ci95_hi < 0 else "Không đủ bằng chứng")
        rows.append({"Biến thể": LBL[r.experiment], "Đối chứng": r.reference, "Macro-F1 CV": f"{res_row.macro_f1_mean:.4f} ± {res_row.macro_f1_std:.4f}", "Δ": sg(r.delta), "CI95 của Δ": cis(r.ci95_lo, r.ci95_hi), "Kết luận": dec})
add("T_cv", pd.DataFrame(rows))

# V2 so với R0 trên cùng OOF, chỉ dùng dev
ids = pd.read_csv(EXP_V2 / "oof" / "V2.csv.gz")
r0 = pd.read_csv(EXP_V2 / "oof" / "R0.csv.gz")
assert (ids["Comment ID"] == r0["Comment ID"]).all()
y = ids.y_true.map(L2I).to_numpy()
dev = pd.concat([read_split("train"), read_split("val")], ignore_index=True).set_index("Comment ID").loc[ids["Comment ID"]]
W = boot_weights(dev["Clean Text"].to_numpy(), B=5000)
bo = lambda d: np.mean([boot_macro_f1(y, d[f"pred_r{r}"].map(L2I).to_numpy(), W) for r in range(5)], 0)
ba, bb = bo(ids), bo(r0)
pt = lambda d: np.mean([macro_f1_cm(cm3(y, d[f"pred_r{r}"].map(L2I).to_numpy()).astype(float)) for r in range(5)])
cvr = pd.DataFrame([dict(a="V2", b="R0", f1_a=pt(ids), f1_b=pt(r0), delta=pt(ids) - pt(r0), ci95_lo=ci95(ba - bb)[0], ci95_hi=ci95(ba - bb)[1], p_boot=p_boot(ba - bb))])
cvr.to_csv(OUT / "cv_v2_vs_r0.csv", index=False)
add("T_v2_r0", cvr.round(4))
sel = pd.read_csv(EXP_V2 / "selected_configs.csv")
s2 = sel[sel.variant == "V2"]
add("V2 chọn cấu hình", text=f"alpha {s2.alpha.value_counts().to_dict()} | min_df {s2.min_df.value_counts().to_dict()} | max_order {s2.max_order.value_counts().to_dict()} | fit_prior {s2.fit_prior.value_counts().to_dict()}")

# ---------- test-v2 ----------
t = pd.read_csv(EXP_V2 / "final_test_v2_results.csv")
add("T_tv2", pd.DataFrame([{"Mô hình": x.model, "Accuracy": f4(x.accuracy), "Macro-P": f4(x.macro_precision), "Macro-R": f4(x.macro_recall), "Macro-F1": f4(x.macro_f1), "CI95": ci(x.ci95_lo, x.ci95_hi)} for x in t.itertuples()]))
c = pd.read_csv(EXP_V2 / "final_test_v2_comparisons.csv")
add("T_tv2_cmp", pd.DataFrame([{"So sánh": f"{x.a} so với {x.b}", "Δ macro-F1": sg(x.delta), "CI95 của Δ": cis(x.ci95_lo, x.ci95_hi), "p Holm": pv(x.p_holm), "A đúng B sai / A sai B đúng": f"{x.a_right_b_wrong} / {x.a_wrong_b_right}", "p McNemar": pv(x.p_mcnemar_exact)} for x in c.itertuples()]))
f = pd.read_csv(EXP_V2 / "final_film_v2_results.csv")
add("T_film_ref", pd.DataFrame([{"Mô hình": x.model, "Accuracy": f4(x.accuracy), "Macro-F1": f4(x.macro_f1), "CI95": ci(x.ci95_lo, x.ci95_hi)} for x in f.itertuples()]))
add("T_bert_cv", pd.read_csv(EXP_V2 / "results_cv_bert.csv").round(4))
add("T_bert_cmp", pd.read_csv(EXP_V2 / "comparisons_cv_bert.csv").round(4))

# ---------- C ----------
for tag in ("splitD", "splitH"):
    r = pd.read_csv(OUT / f"c_{tag}_results.csv")
    add(f"T_{tag}", pd.DataFrame([{"Mô hình": x.model, "Accuracy": f4(x.accuracy), "Macro-P": f4(x.macro_precision), "Macro-R": f4(x.macro_recall), "Macro-F1": f4(x.macro_f1), "CI95 macro-F1": ci(x.ci95_lo, x.ci95_hi), "Macro-AUC": f4(x.macro_auc), "CI95 macro-AUC": ci(x.macro_auc_lo, x.macro_auc_hi)} for x in r.itertuples()]))
    c = pd.read_csv(OUT / f"c_{tag}_comparisons.csv")
    add(f"T_{tag}_cmp", pd.DataFrame([{"So sánh": f"{x.a} so với {x.b}", "Δ macro-F1": sg(x.delta), "CI95 của Δ": cis(x.ci95_lo, x.ci95_hi), "p Holm": pv(x.p_holm), "A đúng B sai / A sai B đúng": f"{x.a_right_b_wrong} / {x.a_wrong_b_right}", "p McNemar": pv(x.p_mcnemar_exact), "Δ macro-AUC": sg(x.delta_macro_auc), "CI95 của Δ AUC": cis(x.delta_auc_lo, x.delta_auc_hi)} for x in c.itertuples()]))
    p = pd.read_csv(OUT / f"c_{tag}_per_class.csv")
    add(f"T_{tag}_pc", pd.DataFrame([{"Mô hình": m, **{f"{l} ({k})": f3(g.set_index('label').loc[l, v]) for l in LABELS for k, v in (("P", "precision"), ("R", "recall"), ("F1", "f1"))}} for m, g in p.groupby("model", sort=False)]))
    aucs = pd.read_csv(OUT / f"c_{tag}_results.csv")
    add(f"T_{tag}_auc", pd.DataFrame([{"Mô hình": x.model, "Negative": f3(x.auc_Negative), "Neutral": f3(x.auc_Neutral), "Positive": f3(x.auc_Positive), "Macro-AUC": f3(x.macro_auc)} for x in aucs.itertuples()]))
add("T_nodup", pd.read_csv(OUT / "c_splitD_nodup_comparisons.csv").round(4))
add("T_nodup_res", pd.read_csv(OUT / "c_splitD_nodup_results.csv").round(4)[["model", "accuracy", "macro_f1", "ci95_lo", "ci95_hi", "n"]])
cf = json.loads((OUT / "c_configs.json").read_text(encoding="utf-8"))["configs"]
add("Cấu hình C", text="\n".join(f"- {k}: {v}" for k, v in cf.items()))

(OUT / "report_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
