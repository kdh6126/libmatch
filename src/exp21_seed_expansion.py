"""Experiment 21 -- seed expansion for the one-hot vs ESM comparison.

The manuscript reports this comparison as a win count ("ESM won 20 of 25
intervals"), which is the form a referee is most likely to challenge: with five
seeds a count can turn on a couple of lucky splits. Here the same comparison is
rerun with twenty independent train/test splits per antigen, so the effect can
be reported as a mean difference with a confidence interval and a paired test
rather than a tally.

The ESM representation is read from the cache built by exp17, so nothing about
the features changes -- only the number of splits. The convolutional network is
not fitted here because it plays no part in this particular comparison.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, ttest_1samp, wilcoxon
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")

AA = "ACDEFGHIKLMNPQRSTVWY-"
AAIDX = {a: i for i, a in enumerate(AA)}
SIZES = [200, 500, 1000, 2000, 4000]
SEEDS = 20
HOLD = 1500
CACHE = "data/esm"

ANTIGENS = [("2DD8_S", "CARDRLLWLRLWYFDVW", "SARS-CoV spike RBD"),
            ("1NCA_N", "CAREFDLFGLLLLFDYW", "influenza N9 neuraminidase")]


def library_is_valid(ag, parent):
    """The parent must carry the same energy here as in the released library.

    Same binary, same antigen, same scoring function produced both, so any
    disagreement means the run that wrote the file did not finish. Checking it
    here stops a broken library from reaching the results silently, which is
    how the 1NCA_N file went unnoticed.
    """
    lib = f"data/absolut/matched_{ag}.parquet"
    if not os.path.exists(lib):
        return False, "library file is absent"
    m = pd.read_parquet(lib)
    row = m[m.cdr3 == parent]
    if not len(row):
        return False, "parent is not in the library"
    orc = f"data/absolut/oracle_{ag}.parquet"
    if not os.path.exists(orc):
        return True, "no oracle file to check against"
    o = pd.read_parquet(orc)
    ecol = [c for c in o.columns if "energy" in c.lower()][0]
    ccol = "cdr3" if "cdr3" in o.columns else o.columns[0]
    ref = o.loc[o[ccol] == parent, ecol]
    if not len(ref):
        return True, "parent absent from the oracle"
    d = abs(float(row.energy.iloc[0]) - float(ref.iloc[0]))
    if d > 1e-3:
        return False, (f"parent energy {float(row.energy.iloc[0]):.2f} disagrees "
                       f"with the oracle {float(ref.iloc[0]):.2f}")
    return True, "parent matches the oracle"


def encode(seqs, ref):
    """Equal lengths here, so position i maps to position i."""
    L = len(ref)
    proj = np.array([list(s.ljust(L)[:L]) for s in seqs])
    var = [j for j in range(L) if len(set(proj[:, j])) > 1]
    X = np.zeros((len(seqs), len(var) * 21), np.float32)
    for c, j in enumerate(var):
        for i, ch in enumerate(proj[:, j]):
            X[i, c * 21 + AAIDX.get(ch, AAIDX["-"])] = 1.0
    return sparse.csr_matrix(X), var


rows = []
used = []
for ag, parent, desc in ANTIGENS:
    ok, why = library_is_valid(ag, parent)
    print(f"{ag}: {'USING' if ok else 'SKIPPING'} -- {why}")
    if not ok:
        continue
    used.append(ag)
    d = pd.read_parquet(f"data/absolut/matched_{ag}.parquet")
    d = d[d.cdr3.str.len() == len(parent)].reset_index(drop=True)
    y = d.energy.to_numpy(np.float32)
    Xs, var = encode(d.cdr3.tolist(), parent)

    path = f"{CACHE}/matched_{ag}_perpos.npz"
    if not os.path.exists(path):
        sys.exit(f"ESM cache missing: {path}. Run exp17 for {ag} first.")
    Xesm = np.load(path)["X"]
    assert len(Xesm) == len(y), f"{ag}: ESM cache has {len(Xesm)} rows, data has {len(y)}"

    print(f"\n=== {ag} ({desc}) ===", flush=True)
    print(f"{len(d):,} sequences, {len(var)} variable positions, "
          f"ESM {Xesm.shape}", flush=True)

    for s in range(SEEDS):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:HOLD], perm[HOLD:]
        yte = y[te]
        for m in SIZES:
            if m > len(pool):
                continue
            tr = pool[:m]
            r = Ridge(alpha=1.0, solver="lsqr").fit(Xs[tr], y[tr])
            p_oh = Xs[te] @ r.coef_ + r.intercept_
            re = Ridge(alpha=50.0, solver="lsqr").fit(Xesm[tr], y[tr])
            p_es = re.predict(Xesm[te])
            a = float(spearmanr(yte, p_oh).statistic)
            b = float(spearmanr(yte, p_es).statistic)
            rows.append(dict(antigen=ag, n_train=m, seed=s,
                             onehot=round(a, 4), esm=round(b, 4),
                             onehot_minus_esm=round(a - b, 4)))
        print(f"  seed {s:2d} done", flush=True)

R = pd.DataFrame(rows)
R.to_csv("results/exp21_seed_expansion.csv", index=False, encoding="utf-8")

# ---- per (antigen, size): paired test across seeds ---------------------------
out = []
for (ag, m), g in R.groupby(["antigen", "n_train"]):
    d = g.onehot_minus_esm.to_numpy()
    t = ttest_1samp(d, 0.0)
    try:
        w = float(wilcoxon(d).pvalue)
    except ValueError:          # all differences identical in sign and value
        w = np.nan
    boot = np.array([np.random.default_rng(7 + i).choice(d, len(d), True).mean()
                     for i in range(10000)])
    out.append(dict(antigen=ag, n_train=m, n_seeds=len(d),
                    mean_diff=round(d.mean(), 4), sd=round(d.std(ddof=1), 4),
                    ci_lo=round(np.percentile(boot, 2.5), 4),
                    ci_hi=round(np.percentile(boot, 97.5), 4),
                    esm_wins=int((d < 0).sum()),
                    p_ttest=float(t.pvalue), p_wilcoxon=w))
S = pd.DataFrame(out)
S.to_csv("results/exp21_per_size_tests.csv", index=False, encoding="utf-8")

pd.set_option("display.width", 200)
print("\n=== per antigen and training size, across "
      f"{SEEDS} splits (negative favours ESM) ===")
print(S.to_string(index=False))

print("\n=== pooled per antigen ===")
for ag, g in R.groupby("antigen"):
    d = g.onehot_minus_esm.to_numpy()
    boot = np.array([np.random.default_rng(11 + i).choice(d, len(d), True).mean()
                     for i in range(10000)])
    print(f"{ag}: mean {d.mean():+.4f} "
          f"[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}], "
          f"ESM won {(d < 0).sum()}/{len(d)} cells")

print("\nwrote results/exp21_seed_expansion.csv and results/exp21_per_size_tests.csv")
