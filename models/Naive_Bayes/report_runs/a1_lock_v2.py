"""A1: chốt ứng viên v2. KHÔNG đọc test_v2.csv và KHÔNG giữ dòng test phim.

Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/report_runs/a1_lock_v2.py
Đầu ra: experiments_v2/final_candidates_v2.json, results_cv_bert.csv, comparisons_cv_bert.csv, oof_probs_v2.npz
"""
import itertools
import json
import os
import sys

from common import *  # noqa: F403
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

check_locked()
LOCK = EXP_V2 / "final_candidates_v2.json"
assert not LOCK.exists(), "Ứng viên v2 đã khóa, không ghi đè."
N_JOBS = len(os.sched_getaffinity(0))

film = pd.read_csv(V2 / "film_clean_v2.csv", encoding="utf-8-sig", keep_default_na=False)
film = film[film.split != "test"].reset_index(drop=True)          # bỏ dòng test ngay khi nạp
assert len(film) == 1800
ids = film["Comment ID"].to_numpy()
txt = film.text_v2.to_numpy(dtype=object)
y = film.label.map(L2I).to_numpy()
F = load_folds(ids)                                               # (R, N)
R, K, N = 5, 5, len(film)
clean = read_split("train").pipe(lambda a: pd.concat([a, read_split("val")])).set_index("Comment ID").loc[ids, "Clean Text"]
gid = pd.factorize(clean.to_numpy())[0]

sel = pd.read_csv(EXP_V2 / "selected_configs.csv")


def s_of(name):
    return {(int(s.repeat), int(s.fold)): s for s in sel[sel.variant == name].itertuples()}


def vec(md, mo=2):
    return make_vec("tfidf", mo, md, token_pattern=TOK_V2, lowercase=False)


# ---------- 1. dựng lại xác suất OOF của V2, LR1, HV2 ----------
def fold_probs(r, k, c_nb, c_lr):
    tr, te = np.where(F[r] != k)[0], np.where(F[r] == k)[0]
    v = vec(int(c_nb.min_df), int(c_nb.max_order))
    Xa, Xb = v.fit_transform(txt[tr]), v.transform(txt[te])
    nb = fit_nb(Xa, y[tr], float(c_nb.alpha), str(c_nb.fit_prior) == "True", balanced=True)
    v = vec(int(c_lr.min_df))
    Xa, Xb2 = v.fit_transform(txt[tr]), v.transform(txt[te])
    cw = None if str(c_lr.class_weight) == "None" else c_lr.class_weight
    lr = LogisticRegression(C=float(c_lr.C), class_weight=cw, max_iter=2000).fit(Xa, y[tr])
    return r, te, nb.predict_proba(Xb), lr.predict_proba(Xb2)


s_nb, s_lr, s_hv = s_of("V2"), s_of("LR1"), s_of("HV2")
assert len(s_nb) == len(s_lr) == len(s_hv) == 25
P_nb, P_lr, P_hv = np.zeros((R, N, C)), np.zeros((R, N, C)), np.zeros((R, N, C))
keys = list(itertools.product(range(R), range(K)))
for (r, k), (_, te, a, b) in zip(keys, Parallel(n_jobs=N_JOBS)(delayed(fold_probs)(r, k, s_nb[(r, k)], s_lr[(r, k)]) for r, k in keys)):
    w = float(s_hv[(r, k)].w_nb)
    P_nb[r, te], P_lr[r, te], P_hv[r, te] = a, b, w * a + (1 - w) * b

# đối chiếu với dự đoán OOF đã lưu của notebook
for name, P in (("V2", P_nb), ("LR1", P_lr), ("HV2", P_hv)):
    o = pd.read_csv(EXP_V2 / "oof" / f"{name}.csv.gz")
    assert (o["Comment ID"].to_numpy() == ids).all()
    agree = np.mean([(pd.Series(P[r].argmax(1)).map(dict(enumerate(LABELS))).to_numpy() == o[f"pred_r{r}"].to_numpy()).mean() for r in range(R)])
    f = np.mean([f1_score(y, P[r].argmax(1), average="macro") for r in range(R)])
    print(f"{name}: khớp dự đoán OOF đã lưu {agree:.6f} | macro-F1 {f:.4f}")
    assert agree > 0.999, f"không tái lập được OOF của {name}"

# ---------- 2. điểm CV của P1, P2, PE ----------
P1, P2 = [np.load(EXP_V2 / f"phobert_oof_{n}.npz") for n in ("P1", "P2")]
assert P1["done"].all() and P2["done"].all() and (P1["comment_id"] == ids).all() and (P2["comment_id"] == ids).all()
prob = {"V2": P_nb, "LR1": P_lr, "HV2": P_hv, "P1": P1["prob"], "P2": P2["prob"]}
f_of = lambda P: float(np.mean([f1_score(y, P[r].argmax(1), average="macro") for r in range(R)]))
best_p = "P1" if f_of(prob["P1"]) >= f_of(prob["P2"]) else "P2"
prob["PE"] = 0.5 * prob[best_p] + 0.5 * prob["HV2"]

W = boot_weights(gid, B=5000)


def boot_cv(P):
    return np.mean([boot_macro_f1(y, P[r].argmax(1), W) for r in range(R)], 0)


rows, boots = [], {}
for name in ("V2", "LR1", "HV2", "P1", "P2", "PE"):
    per = [metric_row(y, prob[name][r].argmax(1)) for r in range(R)]
    boots[name] = boot_cv(prob[name])
    lo, hi = ci95(boots[name])
    rows.append(dict(variant=name, macro_f1_mean=np.mean([p["macro_f1"] for p in per]), macro_f1_std=np.std([p["macro_f1"] for p in per], ddof=1),
                     f1_negative=np.mean([p["f1_Negative"] for p in per]), f1_neutral=np.mean([p["f1_Neutral"] for p in per]),
                     f1_positive=np.mean([p["f1_Positive"] for p in per]), accuracy=np.mean([p["accuracy"] for p in per]),
                     macro_f1_ci95_lo=lo, macro_f1_ci95_hi=hi))
res = pd.DataFrame(rows)
res.to_csv(EXP_V2 / "results_cv_bert.csv", index=False)
print(res.round(4).to_string(index=False))

cmp_rows = []
for a, b in (("P1", "HV2"), ("P2", "P1"), ("PE", best_p), ("PE", "HV2"), ("P1", "V2"), ("PE", "V2")):
    fa, fb = f_of(prob[a]), f_of(prob[b])
    d = boots[a] - boots[b]
    lo, hi = ci95(d)
    cmp_rows.append(dict(a=a, b=b, f1_a=fa, f1_b=fb, delta=fa - fb, ci95_lo=lo, ci95_hi=hi, p_boot=p_boot(d), accepted=lo > 0))
cmp_ = pd.DataFrame(cmp_rows)
cmp_["p_holm"] = holm(cmp_.p_boot)
cmp_.to_csv(EXP_V2 / "comparisons_cv_bert.csv", index=False)
print(cmp_.round(4).to_string(index=False))

# ---------- 3. chọn cấu hình cuối trên 25 outer fold ----------
ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
PRIORS = (False, True)
BLOCKS = list(itertools.product((1, 2), (1, 2, 3)))
NB_GRID = [dict(max_order=mo, min_df=md, fit_prior=p, alpha=a) for mo, md in BLOCKS for p in PRIORS for a in ALPHAS]
assert len(NB_GRID) == 84


def nb_block(r, k, mo, md):
    tr, te = np.where(F[r] != k)[0], np.where(F[r] == k)[0]
    v = vec(md, mo)
    Xa, Xb = v.fit_transform(txt[tr]), v.transform(txt[te])
    out = []
    for p in PRIORS:
        for a in ALPHAS:
            out.append(cm3(y[te], fit_nb(Xa, y[tr], a, p, balanced=True).predict(Xb)))
    return np.array(out, dtype=float)


task = [(r, k, mo, md) for (mo, md) in BLOCKS for r in range(R) for k in range(K)]
res_b = Parallel(n_jobs=N_JOBS)(delayed(nb_block)(*t) for t in task)
agg = {}
for t, cm in zip(task, res_b):
    agg[t[2:]] = agg.get(t[2:], 0) + cm
f = np.concatenate([macro_f1_cm(agg[b]) for b in BLOCKS])
nb_cfg = NB_GRID[int(f.argmax())]
print("NB_v2:", nb_cfg, round(float(f.max()), 4))

LR_C, LR_CW, LR_MD = (0.3, 1, 3, 10), (None, "balanced"), (1, 2)
LR_GRID = [dict(max_order=2, min_df=md, C=c, class_weight=cw) for md in LR_MD for c in LR_C for cw in LR_CW]


def lr_block(r, k, md):
    tr, te = np.where(F[r] != k)[0], np.where(F[r] == k)[0]
    v = vec(md)
    Xa, Xb = v.fit_transform(txt[tr]), v.transform(txt[te])
    return [cm3(y[te], LogisticRegression(C=c, class_weight=cw, max_iter=2000).fit(Xa, y[tr]).predict(Xb)) for c in LR_C for cw in LR_CW]


task = [(r, k, md) for md in LR_MD for r in range(R) for k in range(K)]
res_b = Parallel(n_jobs=N_JOBS)(delayed(lr_block)(*t) for t in task)
agg = {}
for t, cm in zip(task, res_b):
    agg[t[2]] = agg.get(t[2], 0) + np.array(cm, dtype=float)
g = np.concatenate([macro_f1_cm(agg[md]) for md in LR_MD])
lr_cfg = LR_GRID[int(g.argmax())]
print("LR_v2:", lr_cfg, round(float(g.max()), 4))

WS = (0.0, 0.25, 0.5, 0.75, 1.0)
scores = [f_of(w * prob["V2"] + (1 - w) * prob["LR1"]) for w in WS]
w_nb = WS[int(np.argmax(scores))]
print("w_nb:", dict(zip(WS, np.round(scores, 4))), "->", w_nb)

bert = max(("P1", "P2", "PE"), key=lambda n: f_of(prob[n]))
print("Bert_v2:", bert, "| best_p:", best_p)

np.savez_compressed(EXP_V2 / "oof_probs_v2.npz", comment_id=ids, **{k: v.astype(np.float32) for k, v in prob.items()})
final = {**run_info(), "locked_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "note": "khóa trước khi mở test-v2; xem experiment_plan_v3.md",
         "R0v2": dict(max_order=2, min_df=2, alpha=0.01, fit_prior=False, balanced=False),
         "NB_v2": dict(variant="V2", balanced=True, cv_score_on_outer_folds=float(f.max()), **nb_cfg),
         "LR_v2": dict(variant="LR1", cv_score_on_outer_folds=float(g.max()), **lr_cfg),
         "Hybrid_v2": dict(variant="HV2", w_nb=w_nb, grid_scores=dict(zip(map(str, WS), scores))),
         "Bert_v2": dict(variant=bert, base=best_p, use_pool=best_p == "P2", seed=SEED, cv_scores={n: f_of(prob[n]) for n in ("P1", "P2", "PE")})}
LOCK.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
print("Đã khóa:", LOCK.name, sha256(LOCK))
