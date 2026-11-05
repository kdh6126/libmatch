"""Is there actually a distribution shift between the two design campaigns?

The prospective task trains on iteration 0 and tests on iteration 1. Before
drawing a schematic that shows two displaced clouds, check that the clouds are
in fact displaced -- in sequence space and in label space.
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp
from sklearn.decomposition import TruncatedSVD

AA = "ACDEFGHIKLMNPQRSTVWY-"
AAIDX = {a: i for i, a in enumerate(AA)}

reg = pd.read_parquet("data/alphaseq/panel_cov_reg.parquet")
ag = reg.matalpha_description.value_counts().idxmax()
d = reg[reg.matalpha_description == ag].copy()
tr = d[d.iteration == 0]
te = d[d.iteration == 1]
te = te[~te.mata_sequence.isin(set(tr.mata_sequence))]
print(f"antigen {ag}: iteration 0 n={len(tr):,}   iteration 1 n={len(te):,} "
      f"(after dropping re-measured)\n")


def onehot(seqs, ref):
    L = len(ref)
    rows, cols = [], []
    for i, s in enumerate(seqs):
        s = s.ljust(L)[:L]
        for j, ch in enumerate(s):
            rows.append(i)
            cols.append(j * 21 + AAIDX.get(ch, AAIDX["-"]))
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                             shape=(len(seqs), L * 21))


ref = d.mata_sequence.value_counts().idxmax()
rng = np.random.default_rng(0)
N = 6000
itr = rng.choice(len(tr), min(N, len(tr)), replace=False)
ite = rng.choice(len(te), min(N, len(te)), replace=False)
sub = pd.concat([tr.iloc[itr], te.iloc[ite]])
X = onehot(sub.mata_sequence.tolist(), ref)

sv = TruncatedSVD(n_components=2, random_state=0).fit(X)
P = sv.transform(X)
lab = sub.iteration.to_numpy()
print(f"sequence length reference {len(ref)}, one-hot {X.shape}")
print(f"explained variance {sv.explained_variance_ratio_.sum():.1%}\n")

a, b = P[lab == 0], P[lab == 1]
ca, cb = a.mean(0), b.mean(0)
spread = np.sqrt(((a - ca) ** 2).sum(1).mean())
sep = np.linalg.norm(ca - cb)
print("SEQUENCE SPACE")
print(f"  centroid separation {sep:.3f}")
print(f"  within-campaign spread {spread:.3f}")
print(f"  separation / spread  {sep / spread:.2f}")
for k in (0, 1):
    print(f"  PC{k + 1} KS test iter0 vs iter1: "
          f"D={ks_2samp(a[:, k], b[:, k]).statistic:.3f}, "
          f"p={ks_2samp(a[:, k], b[:, k]).pvalue:.2e}")

# how far is each iter1 sequence from the nearest iter0 sequence, in mutations?
print("\nMUTATIONAL DISTANCE FROM THE PARENT")
for name, g in [("iteration 0", tr), ("iteration 1", te)]:
    s = g.mata_sequence.sample(min(3000, len(g)), random_state=0)
    dist = [sum(c1 != c2 for c1, c2 in zip(x.ljust(len(ref))[:len(ref)], ref))
            for x in s]
    print(f"  {name}: median {np.median(dist):.0f} mutations, "
          f"90th pct {np.percentile(dist, 90):.0f}, max {max(dist)}")

print("\nLABEL SPACE")
print(f"  iter0 affinity {tr.affinity.mean():.3f} +- {tr.affinity.std():.3f}")
print(f"  iter1 affinity {te.affinity.mean():.3f} +- {te.affinity.std():.3f}")
ks = ks_2samp(tr.affinity.dropna(), te.affinity.dropna())
print(f"  KS D={ks.statistic:.3f}, p={ks.pvalue:.2e}")

np.savez("results/split_shift.npz", P=P, lab=lab,
         aff=sub.affinity.to_numpy())
print("\nwrote results/split_shift.npz")
