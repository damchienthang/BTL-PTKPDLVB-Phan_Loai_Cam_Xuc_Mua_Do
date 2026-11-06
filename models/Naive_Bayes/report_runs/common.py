"""Hàm dùng chung cho các script trong report_runs/. Thiết kế xem experiment_plan_v3.md."""
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB

NB_DIR = Path(__file__).resolve().parents[1]
ROOT = NB_DIR.parents[1]
SPLITS = NB_DIR / "datasets" / "splits"
V2 = NB_DIR / "datasets" / "v2"
EXP_V1 = NB_DIR / "experiments"
EXP_V2 = NB_DIR / "experiments_v2"
OUT = NB_DIR / "experiments_report"
OUT.mkdir(exist_ok=True)
XLSX = ROOT / "Data" / "final_labeled_clean_v3.xlsx"

SEED = 42
LABELS = ["Negative", "Neutral", "Positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
C = len(LABELS)
TOK_DEFAULT = r"(?u)\b\w\w+\b"          # mặc định của scikit-learn, dùng cho Clean Text
TOK_V2 = r"(?u)\b\w+\b"                 # giữ token một ký tự, dùng cho clean_v2

LOCKED = {
    SPLITS / "train.csv": "c04400673765183a0c904d447b611f48b74389c1b68059d517237d4cdab38505",
    SPLITS / "val.csv": "155c13a615172a7b06c4f2b647e222736d0354dcb603ad90c0c027ecd67f31c7",
    SPLITS / "test.csv": "6135fb08b390b6743968a1fb1dde8319977e3297bad656f00e1d669f972b9ae2",
    XLSX: "ea6c578763a8396f4c065696e50df0431a0a978cfff40399818e2a7b4ef7a89c",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_locked():
    for p, want in LOCKED.items():
        assert sha256(p) == want, f"File bị thay đổi: {p}"
    man = json.loads((V2 / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, want in man["sha256"].items():
        assert sha256(V2 / name) == want, f"File bị thay đổi: {name}"


def run_info(**extra):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "git_commit": commit, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__,
            "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seed": SEED, **extra}


def read_split(name):
    d = pd.read_csv(SPLITS / f"{name}.csv", encoding="utf-8-sig", keep_default_na=False)
    return d.assign(split=name)


def load_folds(ids):
    """Trả về F có kích thước (R, N): F[r, i] là outer fold của dòng i ở lần lặp r."""
    f = pd.read_csv(EXP_V1 / "cv_folds.csv").set_index("Comment ID").loc[ids]
    F = f[[f"fold_r{r}" for r in range(5)]].to_numpy().T
    assert F.shape == (5, len(ids)) and set(np.unique(F)) == set(range(5))
    return F


# ---------- đặc trưng và NB ----------
def make_vec(kind, max_order, min_df, sublinear=True, token_pattern=TOK_DEFAULT, lowercase=True):
    args = dict(ngram_range=(1, max_order), min_df=min_df, token_pattern=token_pattern, lowercase=lowercase)
    return CountVectorizer(**args) if kind == "bow" else TfidfVectorizer(sublinear_tf=sublinear, **args)


def balanced_weights(y):
    y = np.asarray(y)
    return (len(y) / (C * np.bincount(y, minlength=C)))[y]


def fit_nb(X, y, alpha, fit_prior, balanced=False):
    return MultinomialNB(alpha=alpha, fit_prior=fit_prior).fit(X, y, sample_weight=balanced_weights(y) if balanced else None)


# ---------- chỉ số ----------
def cm3(y, p):
    return confusion_matrix(y, p, labels=range(C))


def metric_row(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=range(C), zero_division=0)
    row = dict(accuracy=accuracy_score(y, p), macro_precision=pr.mean(), macro_recall=rc.mean(), macro_f1=f1.mean())
    for i, l in enumerate(LABELS):
        row.update({f"p_{l}": pr[i], f"r_{l}": rc[i], f"f1_{l}": f1[i]})
    return row


def per_class_rows(model, y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(C), zero_division=0)
    return [dict(model=model, label=l, precision=pr[i], recall=rc[i], f1=f1[i], support=int(sup[i])) for i, l in enumerate(LABELS)]


def macro_f1_cm(cm):
    """cm: (..., C, C), hàng là nhãn thật. Trả về macro-F1 (...,)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    prec = tp / np.maximum(cm.sum(-2), 1e-12)
    rec = tp / np.maximum(cm.sum(-1), 1e-12)
    return (2 * prec * rec / np.maximum(prec + rec, 1e-12)).mean(-1)


# ---------- bootstrap theo nhóm ----------
def boot_weights(gid, B=10000, seed=SEED):
    """W: (B, n), trọng số của từng dòng khi lấy mẫu lại theo nhóm."""
    gid = pd.factorize(np.asarray(gid))[0]
    G = gid.max() + 1
    rng = np.random.default_rng(seed)
    W = np.zeros((B, G), dtype=np.float32)
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, G, G), minlength=G)
    return W[:, gid]


def boot_macro_f1(y, p, W):
    code = np.asarray(y) * C + np.asarray(p)
    oh = np.eye(C * C, dtype=np.float32)[code]
    return macro_f1_cm((W @ oh).reshape(-1, C, C).astype(np.float64))


def ci95(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def p_boot(d):
    return max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d))


def mcnemar(y, pa, pb):
    ca, cb = np.asarray(pa) == np.asarray(y), np.asarray(pb) == np.asarray(y)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    out, cur = np.zeros(len(ps)), 0.0
    for i, j in enumerate(ps.argsort()):
        cur = max(cur, min(1.0, (len(ps) - i) * ps[j]))
        out[j] = cur
    return out


def compare(name_a, name_b, y, preds, boots):
    fa, fb = macro_f1_cm(cm3(y, preds[name_a]).astype(float)), macro_f1_cm(cm3(y, preds[name_b]).astype(float))
    d = boots[name_a] - boots[name_b]
    lo, hi = ci95(d)
    n01, n10, pm = mcnemar(y, preds[name_a], preds[name_b])
    return dict(a=name_a, b=name_b, f1_a=fa, f1_b=fb, delta=fa - fb, ci95_lo=lo, ci95_hi=hi, p_boot=p_boot(d),
                a_right_b_wrong=n01, a_wrong_b_right=n10, p_mcnemar_exact=pm)


def score_table(y, preds, W, extra=None):
    """Bảng tổng hợp và bootstrap cho nhiều mô hình trên cùng tập chấm."""
    rows, boots = [], {}
    for name, p in preds.items():
        boots[name] = boot_macro_f1(y, p, W)
        lo, hi = ci95(boots[name])
        rows.append(dict(model=name, **metric_row(y, p), ci95_lo=lo, ci95_hi=hi, n=len(y), **(extra or {})))
    return pd.DataFrame(rows), boots


def _self_test():
    """Đối chiếu phần cài đặt nhanh với bản naive trên dữ liệu ngẫu nhiên nhỏ."""
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, C, 200), rng.integers(0, C, 200)
    gid = rng.integers(0, 60, 200)
    from sklearn.metrics import f1_score
    assert abs(macro_f1_cm(cm3(y, p).astype(float)) - f1_score(y, p, average="macro")) < 1e-12
    W = boot_weights(gid, B=20, seed=1)
    fast = boot_macro_f1(y, p, W)
    ids = pd.factorize(gid)[0]
    for b in range(20):
        w = W[b]
        cm = np.zeros((C, C))
        for yi, pi, wi in zip(y, p, w):
            cm[yi, pi] += wi
        assert abs(macro_f1_cm(cm) - fast[b]) < 1e-5
        for g in np.unique(ids):                         # mọi dòng cùng nhóm phải có cùng trọng số
            assert len(set(w[ids == g].tolist())) == 1
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert mcnemar([1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 1, 1]) == (2, 1, 1.0)


_self_test()
