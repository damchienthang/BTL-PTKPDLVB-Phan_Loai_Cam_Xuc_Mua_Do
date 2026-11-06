"""C: so sánh NB với SVM trên cùng split. C1 là split D của Doanh, C2 là split H.

Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/report_runs/c_nb_vs_svm.py
Đầu ra: experiments_report/c_*.csv, c_configs.json, c_scores_*.npz
"""
import itertools
import json

import joblib
from common import *  # noqa: F403
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

check_locked()
N_JOBS = len(os.sched_getaffinity(0))
ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
INNER = list(itertools.product((False, True), (False, True), ALPHAS))   # fit_prior, balanced, alpha
NGRAMS, MIN_DFS = (1, 2, 3), (1, 2, 3)


def blocks(kind):
    subs = (False, True) if kind == "tfidf" else (True,)                # sublinear không tác dụng với BoW
    return list(itertools.product(NGRAMS, MIN_DFS, subs))


def nb_block_cms(kind, b, Xa, ya, Xb, yb):
    v = make_vec(kind, b[0], b[1], sublinear=b[2])
    A, B = v.fit_transform(Xa), v.transform(Xb)
    return np.array([cm3(yb, fit_nb(A, ya, a, fp, bal).predict(B)) for fp, bal, a in INNER], dtype=float)


def select_nb(kind, X, y, folds, per_fold_mean):
    """folds: danh sách (train_idx, eval_idx). per_fold_mean=True: trung bình F1 theo fold như GridSearchCV. False: gộp ma trận nhầm lẫn."""
    bs = blocks(kind)
    res = Parallel(n_jobs=N_JOBS)(delayed(nb_block_cms)(kind, b, X[a], y[a], X[c], y[c]) for b in bs for a, c in folds)
    res = np.array(res).reshape(len(bs), len(folds), len(INNER), C, C)
    score = macro_f1_cm(res).mean(1) if per_fold_mean else macro_f1_cm(res.sum(1))
    bi, ii = np.unravel_index(int(score.argmax()), score.shape)            # argmax trả về vị trí đầu tiên nếu hòa
    b, (fp, bal, a) = bs[bi], INNER[ii]
    return dict(kind=kind, max_order=b[0], min_df=b[1], sublinear=b[2], fit_prior=fp, balanced=bal, alpha=a), float(score.max())


def fit_nb_cfg(c, Xfit, yfit, Xev):
    v = make_vec(c["kind"], c["max_order"], c["min_df"], sublinear=c["sublinear"])
    m = fit_nb(v.fit_transform(Xfit), yfit, c["alpha"], c["fit_prior"], c["balanced"])
    return m.predict_proba(v.transform(Xev))


def make_svm(kind, ngram=2, min_df=1, sublinear=False, c=1.0, cw=None):
    from sklearn.pipeline import make_pipeline
    vec = make_vec(kind, ngram, min_df, sublinear=sublinear)
    return make_pipeline(vec, OneVsOneClassifier(LinearSVC(C=c, class_weight=cw, max_iter=10000, random_state=18)))


def auc_table(y, scores):
    rows, curves = [], {}
    yb = np.eye(C, dtype=int)[y]
    for n, s in scores.items():
        aucs = [roc_auc_score(yb[:, i], s[:, i]) for i in range(C)]
        rows.append(dict(model=n, **{f"auc_{l}": aucs[i] for i, l in enumerate(LABELS)}, macro_auc=float(np.mean(aucs))))
        curves[n] = [roc_curve(yb[:, i], s[:, i])[:2] for i in range(C)]
    return pd.DataFrame(rows), curves


def boot_auc(y, scores, gid, B=2000, seed=SEED):
    W = boot_weights(gid, B=B, seed=seed)
    yb = np.eye(C, dtype=int)[y]

    def one(n, s, w):
        m = w > 0
        return np.mean([roc_auc_score(yb[m, i], s[m, i], sample_weight=w[m]) for i in range(C)])

    return {n: np.array([one(n, s, W[b]) for b in range(B)]) for n, s in scores.items()}


def evaluate(tag, y, scores, gid, family):
    preds = {n: s.argmax(1) for n, s in scores.items()}
    res, boots = score_table(y, preds, boot_weights(gid), extra=dict(split=tag))
    auc, curves = auc_table(y, scores)
    ba = boot_auc(y, scores, gid)
    auc["macro_auc_lo"] = [ci95(ba[n])[0] for n in auc.model]
    auc["macro_auc_hi"] = [ci95(ba[n])[1] for n in auc.model]
    res = res.merge(auc, on="model")
    res.to_csv(OUT / f"c_{tag}_results.csv", index=False)
    print(res[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "ci95_lo", "ci95_hi", "macro_auc", "macro_auc_lo", "macro_auc_hi"]].round(4).to_string(index=False))
    pd.DataFrame([r for n in preds for r in per_class_rows(n, y, preds[n])]).to_csv(OUT / f"c_{tag}_per_class.csv", index=False)
    for n, p in preds.items():
        pd.DataFrame(cm3(y, p), index=LABELS, columns=LABELS).to_csv(OUT / f"c_{tag}_confusion_{n}.csv")
    fam = pd.DataFrame([compare(a, b, y, preds, boots) for a, b in family])
    fam["p_holm"] = holm(fam.p_boot)
    fam["significant_holm_0.05"] = fam.p_holm < 0.05
    fam["delta_macro_auc"] = [auc.set_index("model").macro_auc[a] - auc.set_index("model").macro_auc[b] for a, b in family]
    fam["delta_auc_lo"] = [ci95(ba[a] - ba[b])[0] for a, b in family]
    fam["delta_auc_hi"] = [ci95(ba[a] - ba[b])[1] for a, b in family]
    fam.to_csv(OUT / f"c_{tag}_comparisons.csv", index=False)
    print(fam.round(4).to_string(index=False))
    np.savez_compressed(OUT / f"c_scores_{tag}.npz", y=y, **{n.replace(" ", "_"): s for n, s in scores.items()})
    rows = [dict(model=n, label=LABELS[i], fpr=f, tpr=t) for n, cv in curves.items() for i, (fp_, tp_) in enumerate(cv) for f, t in zip(fp_, tp_)]
    pd.DataFrame(rows).to_csv(OUT / f"c_{tag}_roc_points.csv.gz", index=False)
    return res, preds


FAMILY = [("NB BoW", "SVM BoW"), ("NB TF-IDF", "SVM TF-IDF"), ("NB TF-IDF", "NB BoW"), ("SVM TF-IDF", "SVM BoW")]
INFO = {}

# ---------- C1: split D ----------
df = pd.read_excel(XLSX)
df.columns = df.columns.str.strip()
X, ylab = df["Clean Text"], df["Final Label"]
Xtr, Xte, ytr_l, yte_l = train_test_split(X, ylab, test_size=0.2, stratify=ylab, random_state=18)
assert len(Xtr) == 1669 and len(Xte) == 418
A, B = Xtr.to_numpy(dtype=object), Xte.to_numpy(dtype=object)
ya, yb = ytr_l.map(L2I).to_numpy(), yte_l.map(L2I).to_numpy()
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=18).split(A, ya))

scores, cfg = {}, {}
for name, kind in (("NB BoW", "bow"), ("NB TF-IDF", "tfidf")):
    cfg[name], s = select_nb(kind, A, ya, folds, per_fold_mean=True)
    print(name, cfg[name], "CV macro-F1", round(s, 4))
    INFO[f"D|{name}"] = dict(config=cfg[name], cv_macro_f1=s)
    scores[name] = fit_nb_cfg(cfg[name], A, ya, B)

PUBLISHED = {"SVM BoW": ("models/bow_svm/best_bow_svm.joblib", 0.6147934637435395), "SVM TF-IDF": ("models/bow_tfidf/best_tfidf_svm.joblib", 0.64693914003026)}
IDX = {l: i for i, l in enumerate(LABELS)}
svm_rows = []
for name, (path, want) in PUBLISHED.items():
    m = joblib.load(ROOT / path)
    clf = m.steps[-1][1]
    assert list(clf.classes_) == LABELS, clf.classes_
    scores[name] = np.asarray(m.decision_function(Xte))
    pred = np.array([IDX[l] for l in m.predict(Xte)])
    assert (pred == scores[name].argmax(1)).all(), "argmax decision_function phải trùng predict"
    got = macro_f1_cm(cm3(yb, pred).astype(float))
    assert abs(got - want) < 1e-9, (name, got)
    INFO[f"D|{name}"] = dict(config=str(m), published_macro_f1=want)
print("SVM tái lập đúng số đã công bố")

gid_d = Xte.astype(str).to_numpy()
res_d, preds_d = evaluate("splitD", yb, scores, gid_d, FAMILY)

# độ bền: bỏ dòng test có text trùng y hệt trong train
keep = ~Xte.isin(set(Xtr)).to_numpy()
print("Bỏ", int((~keep).sum()), "dòng test trùng text với train, còn", int(keep.sum()))
sub = {n: p[keep] for n, p in preds_d.items()}
res_s, boots_s = score_table(yb[keep], sub, boot_weights(gid_d[keep]))
res_s.to_csv(OUT / "c_splitD_nodup_results.csv", index=False)
print(res_s[["model", "accuracy", "macro_f1", "ci95_lo", "ci95_hi"]].round(4).to_string(index=False))
fs = pd.DataFrame([compare(a, b, yb[keep], sub, boots_s) for a, b in FAMILY])
fs["p_holm"] = holm(fs.p_boot)
fs.to_csv(OUT / "c_splitD_nodup_comparisons.csv", index=False)
print(fs.round(4).to_string(index=False))
out = pd.DataFrame({"Comment ID": df.loc[Xte.index, "Comment ID"].values, "y_true": yte_l.values, "dup_in_train": ~keep})
for n, p in preds_d.items():
    out[f"pred_{n}"] = [LABELS[i] for i in p]
out.to_csv(OUT / "c_splitD_predictions.csv", index=False)

# ---------- C2: split H ----------
tr, va, te = read_split("train"), read_split("val"), read_split("test")
dev = pd.concat([tr, va], ignore_index=True)
Xd, Xt = dev["Clean Text"].astype(str).to_numpy(dtype=object), te["Clean Text"].astype(str).to_numpy(dtype=object)
yd, yt = dev["Final Label"].map(L2I).to_numpy(), te["Final Label"].map(L2I).to_numpy()
F = load_folds(dev["Comment ID"].to_numpy())
folds_h = [(np.where(F[r] != k)[0], np.where(F[r] == k)[0]) for r, k in itertools.product(range(5), range(5))]


def svm_cm(kind, params, a, c):
    m = make_svm(kind, **params).fit(Xd[a], yd[a])
    return cm3(yd[c], m.predict(Xd[c]))


def svm_search(kind, grid):
    res = Parallel(n_jobs=N_JOBS)(delayed(svm_cm)(kind, p, a, c) for p in grid for a, c in folds_h)
    res = np.array(res, dtype=float).reshape(len(grid), len(folds_h), C, C).sum(1)
    f = macro_f1_cm(res)
    return grid[int(f.argmax())], float(f.max())


scores_h = {}
for name, kind in (("NB BoW", "bow"), ("NB TF-IDF", "tfidf")):
    c_, s = select_nb(kind, Xd, yd, folds_h, per_fold_mean=False)
    print("H", name, c_, "CV macro-F1", round(s, 4))
    INFO[f"H|{name}"] = dict(config=c_, cv_macro_f1=s)
    scores_h[name] = fit_nb_cfg(c_, Xd, yd, Xt)

for name, kind in (("SVM BoW", "bow"), ("SVM TF-IDF", "tfidf")):
    t0 = time.time()
    # đúng hai bước của Doanh: C và class_weight trước, rồi ngram, min_df, sublinear
    Cs = (0.001, 0.01, 0.03, 0.1, 0.3, 1, 10) if kind == "bow" else (0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
    g1 = [dict(c=c, cw=cw) for c, cw in itertools.product(Cs, (None, "balanced"))]
    p1, _ = svm_search(kind, g1)
    subs = (False, True) if kind == "tfidf" else (False,)
    g2 = [dict(ngram=n, min_df=md, sublinear=sl, **p1) for n, md, sl in itertools.product((1, 2, 3), (1, 2, 3), subs)]
    p2, s = svm_search(kind, g2)
    print("H", name, p2, "CV macro-F1", round(s, 4), f"{time.time() - t0:.0f}s")
    INFO[f"H|{name}"] = dict(config=p2, cv_macro_f1=s)
    m = make_svm(kind, **p2).fit(Xd, yd)
    assert list(m.steps[-1][1].classes_) == [0, 1, 2]
    scores_h[name] = np.asarray(m.decision_function(Xt))

res_h, preds_h = evaluate("splitH", yt, scores_h, te["Clean Text"].astype(str).to_numpy(), FAMILY)
out = pd.DataFrame({"Comment ID": te["Comment ID"].values, "y_true": te["Final Label"].values})
for n, p in preds_h.items():
    out[f"pred_{n}"] = [LABELS[i] for i in p]
out.to_csv(OUT / "c_splitH_predictions.csv", index=False)
(OUT / "c_configs.json").write_text(json.dumps({**run_info(), "configs": INFO}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
