"""A2: chấm test-v2 và (chỉ mô tả) test phim bằng các ứng viên đã khóa. Không tune gì ở đây.

Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/report_runs/a2_score_v2.py
Đầu ra: experiments_v2/final_test_v2_*.csv, final_film_v2_*.csv
"""
import json
import sys

from common import *  # noqa: F403
from sklearn.linear_model import LogisticRegression

check_locked()
LOCK = EXP_V2 / "final_candidates_v2.json"
assert LOCK.exists(), "Chưa khóa ứng viên, chạy a1_lock_v2.py trước."
RES = EXP_V2 / "final_test_v2_results.csv"
assert os.environ.get("FINAL_TEST_REPRO") == "1" or not RES.exists(), "test-v2 đã được chấm. Đặt FINAL_TEST_REPRO=1 chỉ để tái lập."
CAND = json.loads(LOCK.read_text(encoding="utf-8"))
print("Ứng viên khóa lúc", CAND["locked_at"], "| sha256", sha256(LOCK)[:16])

film = pd.read_csv(V2 / "film_clean_v2.csv", encoding="utf-8-sig", keep_default_na=False)
dev, te = film[film.split != "test"].reset_index(drop=True), film[film.split == "test"].reset_index(drop=True)
tv2 = pd.read_csv(V2 / "test_v2.csv", encoding="utf-8-sig", keep_default_na=False)
assert len(dev) == 1800 and len(te) == 287 and len(tv2) == 1311
y_dev, y_tv2, y_te = [d.label.map(L2I).to_numpy() for d in (dev, tv2, te)]
EVAL = {"test_v2": tv2.text_v2.to_numpy(dtype=object), "film_test": te.text_v2.to_numpy(dtype=object)}
tr_txt = dev.text_v2.to_numpy(dtype=object)


def vec(md, mo):
    return make_vec("tfidf", mo, md, token_pattern=TOK_V2, lowercase=False)


def nb_probs(c):
    v = vec(c["min_df"], c["max_order"])
    Xa = v.fit_transform(tr_txt)
    m = fit_nb(Xa, y_dev, c["alpha"], c["fit_prior"], balanced=c["balanced"])
    return {k: m.predict_proba(v.transform(x)) for k, x in EVAL.items()}


PROB = {"R0v2": nb_probs(CAND["R0v2"]), "NB_v2": nb_probs(CAND["NB_v2"])}
c = CAND["LR_v2"]
v = vec(c["min_df"], c["max_order"])
lr = LogisticRegression(C=c["C"], class_weight=c["class_weight"], max_iter=2000).fit(v.fit_transform(tr_txt), y_dev)
PROB["LR_v2"] = {k: lr.predict_proba(v.transform(x)) for k, x in EVAL.items()}
w = CAND["Hybrid_v2"]["w_nb"]
PROB["Hybrid_v2"] = {k: w * PROB["NB_v2"][k] + (1 - w) * PROB["LR_v2"][k] for k in EVAL}

# PhoBERT: tham số cố định theo bản đăng ký v2
sys.path.insert(0, str(NB_DIR))
from phobert_v2 import Bert  # noqa: E402

t0 = time.time()
b = CAND["Bert_v2"]
ta, ya = list(tr_txt), y_dev
if b["use_pool"]:
    pool = pd.read_csv(V2 / "pool_vsmec.csv", encoding="utf-8-sig", keep_default_na=False)
    ta, ya = ta + pool.text_v2.tolist(), np.r_[y_dev, pool.label.map(L2I).to_numpy()]
pb = Bert().fit_predict(ta, ya, [list(EVAL["test_v2"]), list(EVAL["film_test"])], seed=b["seed"])
PROB["PhoBERT"] = dict(zip(("test_v2", "film_test"), pb))
print(f"PhoBERT xong sau {time.time() - t0:.0f}s")
PROB["Bert_v2"] = PROB["PhoBERT"] if b["variant"] != "PE" else {k: 0.5 * PROB["PhoBERT"][k] + 0.5 * PROB["Hybrid_v2"][k] for k in EVAL}


def score(key, y, gid, names, prefix, family=None):
    preds = {n: PROB[n][key].argmax(1) for n in names}
    res, boots = score_table(y, preds, boot_weights(gid))
    res.to_csv(EXP_V2 / f"{prefix}_results.csv", index=False)
    print(res[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "ci95_lo", "ci95_hi", "f1_Negative", "f1_Neutral", "f1_Positive"]].round(4).to_string(index=False))
    pd.DataFrame([r for n in names for r in per_class_rows(n, y, preds[n])]).to_csv(EXP_V2 / f"{prefix}_per_class.csv", index=False)
    for n in names:
        pd.DataFrame(cm3(y, preds[n]), index=LABELS, columns=LABELS).to_csv(EXP_V2 / f"{prefix}_confusion_{n}.csv")
    if family:
        c_ = pd.DataFrame([compare(a, b_, y, preds, boots) for a, b_ in family])
        c_["p_holm"] = holm(c_.p_boot)
        c_["significant_holm_0.05"] = c_.p_holm < 0.05
        c_.to_csv(EXP_V2 / f"{prefix}_comparisons.csv", index=False)
        print(c_.round(4).to_string(index=False))
    out = pd.DataFrame({"y_true": [LABELS[i] for i in y]})
    for n, p in preds.items():
        out[f"pred_{n}"] = [LABELS[i] for i in p]
    return out


NAMES = ["R0v2", "NB_v2", "Hybrid_v2", "Bert_v2"]
FAMILY = [("NB_v2", "R0v2"), ("Hybrid_v2", "NB_v2"), ("Bert_v2", "Hybrid_v2")]
o = score("test_v2", y_tv2, tv2.tv2_id.to_numpy(), NAMES, "final_test_v2", FAMILY)
o.insert(0, "tv2_id", tv2.tv2_id.values)
o.to_csv(EXP_V2 / "final_test_v2_predictions.csv", index=False)

print("\nTest phim 287 mẫu, chỉ mô tả:")
clean = read_split("test").set_index("Comment ID").loc[te["Comment ID"], "Clean Text"].to_numpy()
o = score("film_test", y_te, clean, NAMES, "final_film_v2")
o.insert(0, "Comment ID", te["Comment ID"].values)
o.to_csv(EXP_V2 / "final_film_v2_predictions.csv", index=False)
(EXP_V2 / "final_test_v2_run_info.json").write_text(json.dumps(run_info(candidates_sha256=sha256(LOCK), bootstrap_B=10000), ensure_ascii=False, indent=2), encoding="utf-8")
