"""Panning-round analysis on the c-Met campaign (PRJNA607865), full data.

The earlier version had a circular task: the label was derived from the same
round abundances used as features, so it scored a meaningless AUROC of 1.0.
Every task here keeps features and target on disjoint rounds, and each one
carries a baseline and a permutation null so the numbers can be read.

  A  early rounds -> final-round abundance       (can panning stop early?)
     baseline: the last feature round on its own
  B  CDR3 sequence -> final enrichment           (the OPAL premise: score a
     sequence that has never been measured)
  C  sequence + early rounds                     (does sequence add anything?)

VL round 1 (SRR11128264) is excluded: only 5.1% of its reads carry a
recognisable CDR3 against ~89% for every other VL round.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

AA = "ACDEFGHIKLMNPQRSTVWY"
AAI = {a: i for i, a in enumerate(AA)}
NT = CT = 6
MINCOUNT = 10
EXCLUDE = {("VL", 1)}
SEED = 0


def shannon(c):
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def featurise(seqs):
    n = len(seqs)
    X = np.zeros((n, (NT + CT) * 20 + 20 + 1), np.float32)
    for i, s in enumerate(seqs):
        L = len(s)
        for j in range(min(NT, L)):
            k = AAI.get(s[j])
            if k is not None:
                X[i, j * 20 + k] = 1
        for j in range(min(CT, L)):
            k = AAI.get(s[L - 1 - j])
            if k is not None:
                X[i, NT * 20 + j * 20 + k] = 1
        for c in s:
            k = AAI.get(c)
            if k is not None:
                X[i, (NT + CT) * 20 + k] += 1.0 / max(L, 1)
        X[i, -1] = L
    return X


def score(y, p, ytop):
    """Spearman plus retrieval of the true top-100 inside the model's top-500."""
    rho = float(spearmanr(y, p).statistic)
    tt = set(np.argsort(-y)[:100].tolist())
    mt = set(np.argsort(-p)[:500].tolist())
    return dict(spearman=round(rho, 3),
                top100_at_500=round(len(tt & mt) / 100, 3),
                random_top100_at_500=round(500 / len(y), 3),
                auroc=round(roc_auc_score(ytop, p), 3),
                auprc=round(average_precision_score(ytop, p), 3))


def analyse(df, chain):
    d = df[(df["chain"] == chain)]
    wide = d.pivot_table(index="cdr3", columns="round", values="count",
                         aggfunc="sum", fill_value=0)
    rounds = [r for r in sorted(wide.columns) if (chain, r) not in EXCLUDE]
    wide = wide[rounds]
    print(f"\n{'='*74}\n{chain}: rounds {rounds}, {len(wide):,} unique CDR3")

    tot = wide.sum(axis=0)
    for r in rounds:
        c = wide[r][wide[r] > 0]
        print(f"  round {r}: {int(tot[r]):>10,} reads  {len(c):>8,} clonotypes  "
              f"H={shannon(c.values):6.3f}  top-10 share={c.nlargest(10).sum()/c.sum()*100:5.2f}%")

    cpm = wide / tot * 1e6
    first, last = rounds[0], rounds[-1]
    early = rounds[:-2]                       # features
    keep = wide[(wide[first] >= MINCOUNT) | (wide[last] >= MINCOUNT)].index
    c = cpm.loc[keep]
    print(f"\n  clonotypes with >={MINCOUNT} reads in round {first} or {last}: {len(keep):,}")
    print(f"  feature rounds {early} -> target round {last} (disjoint)")

    es = np.log2((c[last] + 1) / (c[first] + 1)).values
    final = np.log1p(c[last].values)
    ytop = (final > np.quantile(final, 0.99)).astype(int)
    idx = np.arange(len(keep))
    itr, ite = train_test_split(idx, test_size=0.3, random_state=SEED)
    out = []

    # ---- A: early rounds -> final-round abundance ----
    Xa = np.log1p(c[early].values)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                               random_state=SEED).fit(Xa[itr], final[itr])
    m = score(final[ite], rf.predict(Xa[ite]), ytop[ite])
    out.append(dict(chain=chain, task="A_early_rounds", **m))
    print(f"  [A ] early rounds {early}          rho={m['spearman']:.3f} "
          f"AUROC={m['auroc']:.3f} top100@500={m['top100_at_500']:.3f}")

    base = c[early[-1]].values                                   # trivial baseline
    m = score(final[ite], base[ite], ytop[ite])
    print(f"  [A0] baseline: round {early[-1]} abundance alone  rho={m['spearman']:.3f} "
          f"AUROC={m['auroc']:.3f} top100@500={m['top100_at_500']:.3f}")
    out.append(dict(chain=chain, task="A0_baseline_last_early_round", **m))

    # ---- B: sequence only -> final enrichment ----
    Xs = featurise(list(keep))
    estop = (es > np.quantile(es, 0.99)).astype(int)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                               random_state=SEED).fit(Xs[itr], es[itr])
    pb = rf.predict(Xs[ite])
    m = score(es[ite], pb, estop[ite])
    out.append(dict(chain=chain, task="B_sequence_only", **m))
    print(f"  [B ] CDR3 sequence only              rho={m['spearman']:.3f} "
          f"AUROC={m['auroc']:.3f} top100@500={m['top100_at_500']:.3f} "
          f"(random {m['random_top100_at_500']:.3f})")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(itr)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                               random_state=SEED).fit(Xs[itr], es[perm])
    m = score(es[ite], rf.predict(Xs[ite]), estop[ite])
    print(f"  [B0] permutation null                rho={m['spearman']:.3f} "
          f"AUROC={m['auroc']:.3f}")
    out.append(dict(chain=chain, task="B0_permutation_null", **m))

    m = score(es[ite], np.array([len(s) for s in keep], float)[ite], estop[ite])
    print(f"  [B1] CDR3 length only                rho={m['spearman']:.3f} "
          f"AUROC={m['auroc']:.3f}")
    out.append(dict(chain=chain, task="B1_length_only", **m))

    # ---- C: sequence + early abundance -> final abundance ----
    Xc = np.hstack([Xs, Xa])
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                               random_state=SEED).fit(Xc[itr], final[itr])
    m = score(final[ite], rf.predict(Xc[ite]), ytop[ite])
    out.append(dict(chain=chain, task="C_sequence_plus_rounds", **m))
    print(f"  [C ] sequence + early rounds         rho={m['spearman']:.3f} "
          f"AUROC={m['auroc']:.3f} top100@500={m['top100_at_500']:.3f}")
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/panning_cmet/cdr3_counts.parquet"
    df = pd.read_parquet(path)
    rows = []
    for chain in ["VH", "VL"]:
        if (df["chain"] == chain).any():
            rows += analyse(df, chain)
    pd.DataFrame(rows).to_csv("results/exp3_enrichment.csv", index=False)
    print("\nwrote results/exp3_enrichment.csv")
