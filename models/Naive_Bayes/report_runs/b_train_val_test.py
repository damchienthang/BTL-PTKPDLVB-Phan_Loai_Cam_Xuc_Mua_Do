"""B: MultinomialNB với BoW và TF-IDF trên split H (train/val/test). Điểm test chỉ mang tính mô tả.

Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/report_runs/b_train_val_test.py
Đầu ra: experiments_report/b_*.csv, b_configs.json, b_confusion_*.csv
"""
import itertools
import json

from common import *  # noqa: F403
from sklearn.metrics import f1_score

check_locked()
tr, va, te = read_split("train"), read_split("val"), read_split("test")
dev = pd.concat([tr, va], ignore_index=True)
X_tr, X_va, X_te, X_dev = [d["Clean Text"].astype(str).to_numpy(dtype=object) for d in (tr, va, te, dev)]
y_tr, y_va, y_te, y_dev = [d["Final Label"].map(L2I).to_numpy() for d in (tr, va, te, dev)]
F = load_folds(dev["Comment ID"].to_numpy())

# ---------- chọn cấu hình, không chạm test ----------
VAL_ALPHAS = (0.01, 0.03, 0.1, 0.3, 0.5, 1.0)                      # lưới của grid_search_val.csv
G_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)               # lưới G


def select_val(kind):
    best, best_f = None, -1.0
    for mo, md in itertools.product((1, 2), (1, 2, 3)):
        v = make_vec(kind, mo, md)
        Xa, Xb = v.fit_transform(X_tr), v.transform(X_va)
        for fp, a in itertools.product((False, True), VAL_ALPHAS):
            s = f1_score(y_va, fit_nb(Xa, y_tr, a, fp).predict(Xb), average="macro")
            if s > best_f + 1e-12:
                best_f, best = s, dict(kind=kind, max_order=mo, min_df=md, alpha=a, fit_prior=fp)
    return best, best_f


def select_cv(kind):
    blocks = list(itertools.product((1, 2), (1, 2, 3)))
    inner = list(itertools.product((False, True), G_ALPHAS))
    fs = []
    for mo, md in blocks:
        cm = np.zeros((len(inner), C, C))
        for r, k in itertools.product(range(5), range(5)):
            a_, b_ = F[r] != k, F[r] == k
            v = make_vec(kind, mo, md)
            Xa, Xb = v.fit_transform(X_dev[a_]), v.transform(X_dev[b_])
            for i, (fp, a) in enumerate(inner):
                cm[i] += cm3(y_dev[b_], fit_nb(Xa, y_dev[a_], a, fp).predict(Xb))
        fs.append(macro_f1_cm(cm))
    f = np.concatenate(fs)
    b, i = divmod(int(f.argmax()), len(inner))
    return dict(kind=kind, max_order=blocks[b][0], min_df=blocks[b][1], alpha=inner[i][1], fit_prior=inner[i][0]), float(f.max())


CFG, SCORE = {}, {}
for name, fn, kind in (("BoW_val", select_val, "bow"), ("TFIDF_val", select_val, "tfidf"), ("BoW_cv", select_cv, "bow"), ("TFIDF_cv", select_cv, "tfidf")):
    CFG[name], SCORE[name] = fn(kind)
    print(name, CFG[name], round(SCORE[name], 4))
assert CFG["TFIDF_val"] == dict(kind="tfidf", max_order=2, min_df=2, alpha=0.01, fit_prior=False), "TFIDF_val phải trùng R0"
lock = json.loads((EXP_V1 / "final_candidates.json").read_text(encoding="utf-8"))["NB_best"]
assert all(CFG["TFIDF_cv"][k] == lock[k] for k in ("max_order", "min_df", "alpha", "fit_prior")), "TFIDF_cv phải trùng R1 đã khóa"
(OUT / "b_configs.json").write_text(json.dumps({**run_info(), "configs": CFG, "selection_scores": SCORE}, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- đánh giá ----------
def fit_pred(c, Xfit, yfit, evals):
    v = make_vec(c["kind"], c["max_order"], c["min_df"])
    m = fit_nb(v.fit_transform(Xfit), yfit, c["alpha"], c["fit_prior"])
    return {k: m.predict(v.transform(x)) for k, x in evals.items()}, len(v.vocabulary_)


rows, per, preds_test = [], [], {}
for name, c in CFG.items():
    p, nf = fit_pred(c, X_tr, y_tr, dict(train=X_tr, val=X_va, test=X_te))
    p2, nf2 = fit_pred(c, X_dev, y_dev, dict(test=X_te))
    for lbl, y_, pr in (("train", y_tr, p["train"]), ("val", y_va, p["val"]), ("test_fit_train", y_te, p["test"]), ("test_fit_trainval", y_te, p2["test"])):
        rows.append(dict(model=name, feature=c["kind"], eval_on=lbl, n=len(y_), n_features=nf2 if lbl == "test_fit_trainval" else nf, **metric_row(y_, pr)))
    preds_test[f"{name}|train"] = p["test"]
    preds_test[f"{name}|trainval"] = p2["test"]

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "b_metrics_all_splits.csv", index=False)
print(tab[["model", "eval_on", "n", "accuracy", "macro_precision", "macro_recall", "macro_f1"]].round(4).to_string(index=False))

# tái lập số đã công bố, để chắc đường sklearn trùng đường GPU
got = tab.set_index(["model", "eval_on"]).macro_f1
assert abs(got["TFIDF_val", "val"] - 0.6146723400992851) < 1e-9
assert abs(got["TFIDF_val", "test_fit_train"] - 0.5440343708530961) < 1e-9
assert abs(got["TFIDF_val", "test_fit_trainval"] - 0.5618270498529397) < 1e-9
assert abs(got["TFIDF_cv", "test_fit_trainval"] - 0.5394595853715349) < 1e-9
print("Tái lập đúng 0.6147 (val), 0.5440, 0.5618, 0.5395 (test)")

gid = te["Clean Text"].astype(str).to_numpy()
W = boot_weights(gid)
res, boots = score_table(y_te, preds_test, W)
res.to_csv(OUT / "b_test_ci.csv", index=False)
print(res[["model", "accuracy", "macro_f1", "ci95_lo", "ci95_hi"]].round(4).to_string(index=False))
for n, p in preds_test.items():
    per += per_class_rows(n, y_te, p)
    pd.DataFrame(cm3(y_te, p), index=LABELS, columns=LABELS).to_csv(OUT / f"b_confusion_{n.replace('|', '_fit_')}.csv")
pd.DataFrame(per).to_csv(OUT / "b_test_per_class.csv", index=False)

c_ = pd.DataFrame([compare("BoW_val|trainval", "TFIDF_val|trainval", y_te, preds_test, boots), compare("BoW_cv|trainval", "TFIDF_cv|trainval", y_te, preds_test, boots)])
c_.to_csv(OUT / "b_test_bow_vs_tfidf.csv", index=False)
print(c_.round(4).to_string(index=False))
out = pd.DataFrame({"Comment ID": te["Comment ID"].values, "y_true": te["Final Label"].values})
for n, p in preds_test.items():
    out[f"pred_{n}"] = [LABELS[i] for i in p]
out.to_csv(OUT / "b_test_predictions.csv", index=False)
