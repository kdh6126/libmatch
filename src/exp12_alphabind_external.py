"""Experiment 12 -- external corroboration from AlphaBind's released campaign data.

A-Alpha Bio ran a prospective, wet-lab-measured comparison of candidate
generators that differ only in featurisation: one-hot (`ohe_*`) versus ESM-2nv
embeddings (`esm_*`), each with and without pre-training, plus the full
AlphaBind model. Every candidate they proposed was then measured by AlphaSeq.

That is an independent test of the same question our exp9b asked retrospectively:
on measured data, does a protein language model beat a one-hot encoding?

Two things are computed here:
  A  which featurisation produced better candidates, by measured affinity
  B  how well each model's own prediction matched the measurement -- the
     prospective accuracy that our exp1 (T2) found collapses
"""
import glob, os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

D = "tools/alphabind/manuscript/data/alphaseq/validation"
OUT = "results"
# parental_aff_margin = candidate - parent, in log10 Kd; negative = tighter
PAIRS = [("esm_warm", "ohe_warm"), ("esm_cold", "ohe_cold"),
         ("alphabind", "ohe_warm")]


def load(f):
    d = pd.read_csv(f)
    d["target"] = os.path.basename(f).replace("_validation_alphaseq.csv", "")
    return d[d["alphaseq_affinity"].notna()]


frames = [load(f) for f in sorted(glob.glob(f"{D}/*.csv"))]
all_ = pd.concat(frames, ignore_index=True)
print(f"AlphaBind 공개 검증 데이터: {len(all_):,} 측정, "
      f"표적 {all_.target.nunique()}종, 모델 {all_.model.nunique()}종\n")

# ---------------- A. featurisation comparison ----------------
rows = []
print("=== A. 어느 특징표현이 더 좋은 후보를 냈는가 (실측 기준) ===")
print("   parental_aff_margin = 후보 - 모항체 (log10 Kd). 음수일수록 개선\n")
for tgt, g in all_.groupby("target"):
    have = set(g.model.unique())
    for a, b in PAIRS:
        if a not in have or b not in have:
            continue
        xa = g[g.model == a]["parental_aff_margin"].dropna()
        xb = g[g.model == b]["parental_aff_margin"].dropna()
        u = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        # rank-biserial effect size
        rb = 1 - 2 * u.statistic / (len(xa) * len(xb))
        better = a if xa.median() < xb.median() else b
        rows.append(dict(target=tgt, comparison=f"{a} vs {b}",
                         n_a=len(xa), n_b=len(xb),
                         median_a=round(float(xa.median()), 3),
                         median_b=round(float(xb.median()), 3),
                         diff=round(float(xa.median() - xb.median()), 3),
                         winner=better, p=float(u.pvalue),
                         rank_biserial=round(float(rb), 3),
                         frac_a_improved=round(float((xa < 0).mean()), 3),
                         frac_b_improved=round(float((xb < 0).mean()), 3)))
        print(f"  {tgt:22s} {a:9s} {xa.median():+.3f}  vs  {b:9s} {xb.median():+.3f}"
              f"   승자={better:9s} p={u.pvalue:.2e}")

A = pd.DataFrame(rows)
A.to_csv(f"{OUT}/exp12_featurisation.csv", index=False, encoding="utf-8")

print("\n--- 요약: esm 계열 vs ohe 계열 ---")
sub = A[A.comparison.str.startswith(("esm_warm", "esm_cold"))]
wins = (sub.winner.str.startswith("esm")).sum()
print(f"  ESM 승 {wins}/{len(sub)} 비교, 중앙값 차이 평균 {sub['diff'].mean():+.3f} log10")
print(f"  모항체보다 개선된 후보 비율 — ESM {sub.frac_a_improved.mean():.1%} "
      f"vs one-hot {sub.frac_b_improved.mean():.1%}")

# ---------------- B. prospective prediction accuracy ----------------
print("\n=== B. 각 모델의 예측이 실측과 얼마나 맞았나 (전향 검증) ===")
rows = []
for (tgt, m), g in all_.groupby(["target", "model"]):
    g = g[g["predicted_affinity"].notna() & g["alphaseq_affinity"].notna()]
    if len(g) < 200:
        continue
    rho = spearmanr(g["alphaseq_affinity"], g["predicted_affinity"]).statistic
    rmse = float(np.sqrt(np.mean((g["alphaseq_affinity"] - g["predicted_affinity"]) ** 2)))
    rows.append(dict(target=tgt, model=m, n=len(g),
                     spearman=round(float(rho), 3), rmse=round(rmse, 3),
                     bias=round(float((g["predicted_affinity"] -
                                       g["alphaseq_affinity"]).mean()), 3)))
B = pd.DataFrame(rows).sort_values(["target", "model"])
B.to_csv(f"{OUT}/exp12_prospective.csv", index=False, encoding="utf-8")
print(B.to_string(index=False))
print(f"\n  전향 예측 Spearman 평균 {B.spearman.mean():.3f} "
      f"(범위 {B.spearman.min():.3f}~{B.spearman.max():.3f})")
print(f"  참고: 본 프로젝트 exp1 전향 검증에서도 rho가 0.72 -> 0.41로 떨어졌다")

print(f"\nwrote {OUT}/exp12_featurisation.csv, {OUT}/exp12_prospective.csv")
