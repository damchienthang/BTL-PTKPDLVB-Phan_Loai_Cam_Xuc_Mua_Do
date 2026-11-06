"""PhoBERT trên 25 outer fold của dữ liệu phim. Không đọc test-v2. Tiếp tục được nếu bị ngắt.

Chạy từ gốc repo: models/Naive_Bayes/.venv/bin/python models/Naive_Bayes/phobert_v2.py
Đầu ra: models/Naive_Bayes/experiments_v2/phobert_oof_{P1,P2}.npz
"""
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "models" / "Naive_Bayes" / "experiments_v2"
EXP.mkdir(exist_ok=True)
MODEL = "vinai/phobert-base-v2"
L2I = {"Negative": 0, "Neutral": 1, "Positive": 2}
DEVICE = "cuda"
SEED = 42


def to_bert(t):
    t = t.replace("qmark", "?").replace("exmark", "!").replace("emocuoi", ":)").replace("emobuon", ":(")
    return re.sub(r"emo([0-9a-f]{4,6})", lambda m: chr(int(m.group(1), 16)), t)


class Bert:
    def __init__(self, max_len=128):
        self.tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
        self.max_len = max_len

    def enc(self, texts):
        return self.tok([to_bert(t) for t in texts], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")

    def fit_predict(self, tr_txt, tr_y, ev_txt_list, seed, epochs=4, bs=32, lr=2e-5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        Etr = self.enc(tr_txt)
        y = torch.tensor(tr_y, device=DEVICE)
        ids, mask = Etr["input_ids"].to(DEVICE), Etr["attention_mask"].to(DEVICE)
        cc = torch.bincount(y, minlength=3).float()
        cw = (len(y) / (3 * cc)).to(DEVICE)
        m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to(DEVICE)
        no = ("bias", "LayerNorm.weight")
        pg = [{"params": [p for n, p in m.named_parameters() if not any(x in n for x in no)], "weight_decay": 0.01},
              {"params": [p for n, p in m.named_parameters() if any(x in n for x in no)], "weight_decay": 0.0}]
        opt = torch.optim.AdamW(pg, lr=lr)
        steps = epochs * math.ceil(len(y) / bs)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler("cuda")
        g = torch.Generator().manual_seed(seed)
        m.train()
        for _ in range(epochs):
            perm = torch.randperm(len(y), generator=g)
            for i in range(0, len(y), bs):
                b = perm[i:i + bs].to(DEVICE)
                L = int(mask[b].sum(1).max())
                with torch.amp.autocast("cuda"):
                    lo = m(input_ids=ids[b][:, :L], attention_mask=mask[b][:, :L]).logits
                    loss = torch.nn.functional.cross_entropy(lo, y[b], weight=cw)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
        m.eval()
        out = []
        for ev in ev_txt_list:
            E = self.enc(ev)
            P = []
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for i in range(0, len(ev), 128):
                    a, b = E["input_ids"][i:i + 128].to(DEVICE), E["attention_mask"][i:i + 128].to(DEVICE)
                    P.append(torch.softmax(m(input_ids=a, attention_mask=b).logits.float(), 1).cpu())
            out.append(torch.cat(P).numpy())
        del m, opt
        torch.cuda.empty_cache()
        return out


def main():
    f = pd.read_csv(ROOT / "models" / "Naive_Bayes" / "datasets" / "v2" / "film_clean_v2.csv", encoding="utf-8-sig", keep_default_na=False)
    f = f[f.split != "test"].reset_index(drop=True)
    folds = pd.read_csv(ROOT / "models" / "Naive_Bayes" / "experiments" / "cv_folds.csv").set_index("Comment ID").loc[f["Comment ID"]]
    y = f.label.map(L2I).values
    txt = f.text_v2.tolist()
    pool = pd.read_csv(ROOT / "models" / "Naive_Bayes" / "datasets" / "v2" / "pool_vsmec.csv", encoding="utf-8-sig", keep_default_na=False)
    p_txt, p_y = pool.text_v2.tolist(), pool.label.map(L2I).values
    b = Bert()
    R, K, N = 5, 5, len(f)
    for name, use_pool in (("P1", False), ("P2", True)):
        path = EXP / f"phobert_oof_{name}.npz"
        if path.exists():
            z = np.load(path)
            prob, done, times = z["prob"], z["done"], z["times"]
        else:
            prob, done, times = np.zeros((R, N, 3)), np.zeros((R, K), dtype=bool), np.zeros((R, K))
        for r in range(R):
            fold = folds[f"fold_r{r}"].values
            for k in range(K):
                if done[r, k]:
                    continue
                t0 = time.time()
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                ta = [txt[i] for i in tr] + (p_txt if use_pool else [])
                ya = np.r_[y[tr], p_y] if use_pool else y[tr]
                prob[r, te] = b.fit_predict(ta, ya, [[txt[i] for i in te]], seed=SEED + 100 * r + k)[0]
                done[r, k] = True
                times[r, k] = time.time() - t0
                np.savez(path, prob=prob, done=done, times=times, comment_id=f["Comment ID"].values)
                print(name, r, k, f"{times[r, k]:.0f}s", flush=True)
    print("XONG", flush=True)


if __name__ == "__main__":
    sys.exit(main())
