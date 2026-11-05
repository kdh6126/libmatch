"""Experiment 16 -- widen the model comparison and fix the sample-size range.

Two gaps in the earlier work:
  * only two model families were compared (ridge, max-pooling CNN), so a reader
    can reasonably ask whether either is a serious baseline
  * the real arm stopped at 5,000 training labels, which is what led us to
    mistake a crossover for a rank inversion

This runs seven models over 200 -> 20,000 labels on four measured targets,
using the ESM embeddings already cached from exp9b. A nearest-neighbour model
is included on purpose: on a mutational library it is the honest "is anything
being learned" control.
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, "src")
from exp8_multi_target import build as build_real, DATASETS
from exp2_search import fit_cnn, cnn_score
# 모델 정의는 exp82(격자 팔)와 공유한다. 복사본을 두면 두 실험의
# 하이퍼파라미터가 갈라지고, 논문은 '같은 일곱 모델'이라고 쓰면서
# 서로 다른 두 벌을 비교하게 된다.
from model_zoo import models

SIZES = [200, 500, 1000, 2000, 5000, 10000, 20000]
SEEDS = 3
HOLD = 4000
CACHE = "data/esm"




rows = []
for name, (ds, ag) in DATASETS.items():
    y, Xs, tok, vocab, antigen = build_real(name, ds, ag)
    p = f"{CACHE}/real_{name}_perpos.npz"
    Xesm = np.load(p)["X"] if os.path.exists(p) else None
    print(f"\n=== {name}  n={len(y):,}  one-hot {Xs.shape[1]}차원  "
          f"ESM {'있음' if Xesm is not None else '없음'} ===", flush=True)

    for s in range(SEEDS):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:HOLD], perm[HOLD:]
        yte = y[te]
        true_top = set(np.argsort(yte)[:100].tolist())
        for m in SIZES:
            if m > len(pool):
                continue
            tr = pool[:m]
            t0 = time.time()
            preds = models(Xs, tok, Xesm, tr, te, y[tr], s)
            for mn, pr in preds.items():
                rho = float(spearmanr(yte, pr).statistic)
                got = len(true_top & set(np.argsort(pr)[:500].tolist())) / 100
                rows.append(dict(target=name, model=mn, n_train=m, seed=s,
                                 spearman=round(rho, 4), top100_at_500=round(got, 3)))
            best = max(preds, key=lambda k: spearmanr(yte, preds[k]).statistic)
            print(f"  n={m:6,} s{s}  최고={best:16s} "
                  + "  ".join(f"{k.split('_')[0][:5]}={spearmanr(yte,v).statistic:+.3f}"
                              for k, v in preds.items())
                  + f"  [{time.time()-t0:.0f}s]", flush=True)

R = pd.DataFrame(rows)
R.to_csv("results/exp16_model_zoo.csv", index=False, encoding="utf-8")

print("\n=== 모델별 평균 Spearman (표적 4종) ===")
piv = R.pivot_table(index="n_train", columns="model", values="spearman").round(3)
print(piv.to_string())

print("\n=== 표적을 재표집한 부트스트랩 95% CI (n=5,000) ===")
sub = R[R.n_train == 5000]
tgts = sorted(sub.target.unique())
rng = np.random.default_rng(0)
for mdl in sorted(sub.model.unique()):
    per = {t: sub[(sub.target == t) & (sub.model == mdl)].spearman.mean()
           for t in tgts}
    vals = np.array([per[t] for t in tgts if not np.isnan(per[t])])
    if len(vals) < 2:
        continue
    boot = np.array([rng.choice(vals, len(vals), replace=True).mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {mdl:18s} {vals.mean():.3f}  [{lo:.3f}, {hi:.3f}]  (표적 {len(vals)}종)")
print("\nwrote results/exp16_model_zoo.csv")
