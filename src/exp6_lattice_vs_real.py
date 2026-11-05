"""Experiment 6 -- is the Absolut! lattice easier than reality, and does the
search advantage survive on measured data?

The headline search results (exp2, exp4) run on Absolut!, a 3D-lattice affinity
model. A lattice energy is deterministic and may be a smoother function of
sequence than real binding, which would inflate every number. This experiment
puts the SAME model and the SAME protocol on three arms:

  lattice        Absolut! energies, subsampled to the size of a real library
  lattice+noise  the same, with Gaussian noise matched to the real noise/SD
  real           AlphaSeq measured affinities on coronavirus RBDs

Part A is a learning curve (how much signal per label). Part B repeats the
budget-limited search at a matched library size. If the advantage holds on the
real arm, the lattice is a convenience rather than a crutch.
"""
import argparse, sys, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
from exp2_search import encode as absolut_encode, fit_cnn, cnn_score, run

NOISE_SD = 0.27               # AlphaSeq WT replicate sd, log10 Kd units
LIB_SIZE = 30_000
HOLDOUT = 5_000
AA_GAP = "ACDEFGHIKLMNPQRSTVWY-"


# ---------------------------------------------------------------- loaders
def load_real(antigen, iteration=1):
    """AlphaSeq library: aligned VHH one-hot + tokens, affinity as the oracle.

    Affinity is log10 Kd, so lower is better -- the same orientation as an
    Absolut! energy, and what `run` expects.
    """
    f = np.load("data/alphaseq/features.npz", allow_pickle=True)
    Xab, ab_seqs = f["Xab"], f["ab_seqs"]
    ix = {s: i for i, s in enumerate(ab_seqs)}
    reg = pd.read_parquet("data/alphaseq/panel_cov_reg.parquet")
    d = reg[(reg.matalpha_description == antigen) & (reg.iteration == iteration)]
    d = d.drop_duplicates("mata_sequence")
    rows = np.array([ix[s] for s in d["mata_sequence"]])
    X = Xab[rows]
    # tokens 1..21 over the aligned variable positions; 0 stays free for padding
    tok = (X.reshape(len(X), -1, 21).argmax(2) + 1).astype(np.uint8)
    return d["affinity"].to_numpy().astype(np.float32), sparse.csr_matrix(X), tok, 22


def load_lattice(antigen, size, rng):
    lib = pd.read_parquet("data/absolut/library.parquet")["cdr3"]
    Xs = sparse.load_npz("data/absolut/library_features.npz").tocsr()
    e = np.load("data/absolut/energies.npz")[antigen]
    sub = np.sort(rng.choice(len(lib), size=size, replace=False))
    tok = absolut_encode(lib.values[sub].tolist())
    return e[sub].astype(np.float32), Xs[sub], tok, 21


def add_matched_noise(y, ratio, rng):
    """Gaussian noise whose sd is `ratio` x the label sd of this library."""
    return y + rng.normal(0, ratio * float(y.std()), size=len(y)).astype(np.float32)


# ---------------------------------------------------------------- part A
def learning_curve(y, Xs, tok, vocab, sizes, seeds, tag, rows):
    n = len(y)
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        perm = rng.permutation(n)
        test, pool = perm[:HOLDOUT], perm[HOLDOUT:]
        yte = y[test]
        true_top = set(np.argsort(yte)[:100].tolist())
        for m in sizes:
            if m > len(pool):
                continue
            tr = pool[:m]
            ridge = Ridge(alpha=1.0, solver="lsqr").fit(Xs[tr], y[tr])
            p_ridge = Xs[test] @ ridge.coef_ + ridge.intercept_
            cnn, mu, sd = fit_cnn(tok[tr], y[tr], seed=s, vocab=vocab)
            p_cnn = cnn_score(cnn, tok[test], mu, sd)
            for name, p in [("ridge", p_ridge), ("cnn", p_cnn)]:
                got = len(true_top & set(np.argsort(p)[:500].tolist())) / 100
                rows.append(dict(arm=tag, model=name, n_train=m, seed=s,
                                 spearman=round(float(spearmanr(yte, p).statistic), 4),
                                 top100_at_500=round(got, 3)))
            print(f"  [{tag:14s}] n={m:6,} s{s} ridge rho="
                  f"{rows[-2]['spearman']:+.3f} cnn rho={rows[-1]['spearman']:+.3f}",
                  flush=True)


# ---------------------------------------------------------------- part B
def budget_search(y, Xs, tok, vocab, budgets, seeds, tag, rows):
    n = len(y)
    order = np.argsort(y)
    n1 = max(1, n // 100)                       # top 1% of a 30k library = 300
    top1 = set(order[:n1].tolist())
    opt = float(y.min())
    for budget in budgets:
        for kind in ["random", "oneshot", "active"]:
            for s in seeds:
                t0 = time.time()
                sel = run(kind, y, Xs, tok, budget,
                          np.random.default_rng(97 * s + budget), s, vocab=vocab)
                hits = len(top1 & set(sel.tolist()))
                rows.append(dict(arm=tag, budget=budget, strategy=kind, seed=s,
                                 lib_size=n, n_top1pct=n1, hits=hits,
                                 recall=round(hits / n1, 4),
                                 best=round(float(y[sel].min()), 3),
                                 gap=round(float(y[sel].min()) - opt, 3),
                                 secs=round(time.time() - t0, 1)))
                print(f"  [{tag:14s}] B={budget:5,} {kind:8s} s{s} "
                      f"hits={hits:4d}/{n1} recall={hits/n1:.3f} "
                      f"best={y[sel].min():.3f} [{rows[-1]['secs']}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-antigens", nargs="+",
                    default=["SARS-CoV2_RBD_(6LZG)", "SARS-CoV1_RBD_(6LZG_Mimic)",
                             "WIV1"])
    ap.add_argument("--lattice-antigens", nargs="+",
                    default=["2DD8_S", "1NCA_N", "1ZTX_E"])
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[200, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--budgets", nargs="+", type=int, default=[500, 1000, 2000, 4000])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    lc, bs = [], []

    print("=== real arm (AlphaSeq measured affinity) ===", flush=True)
    ratios = []
    for ag in a.real_antigens:
        y, Xs, tok, vocab = load_real(ag)
        ratios.append(NOISE_SD / float(y.std()))
        tag = f"real:{ag.split('_')[0][:10]}"
        print(f"{ag}: n={len(y):,}  SD={y.std():.3f}  noise/SD={ratios[-1]:.3f}",
              flush=True)
        learning_curve(y, Xs, tok, vocab, a.sizes, seeds, tag, lc)
        budget_search(y, Xs, tok, vocab, a.budgets, seeds, tag, bs)
    ratio = float(np.mean(ratios))
    print(f"\nmean noise/SD on the real arm = {ratio:.3f}\n", flush=True)

    for ag in a.lattice_antigens:
        for noisy in [False, True]:
            rng = np.random.default_rng(11)
            y, Xs, tok, vocab = load_lattice(ag, LIB_SIZE, rng)
            if noisy:
                y = add_matched_noise(y, ratio, np.random.default_rng(12))
            tag = f"lattice{'+noise' if noisy else ''}:{ag}"
            print(f"=== {tag}  n={len(y):,}  SD={y.std():.3f} ===", flush=True)
            learning_curve(y, Xs, tok, vocab, a.sizes, seeds, tag, lc)
            budget_search(y, Xs, tok, vocab, a.budgets, seeds, tag, bs)

    L = pd.DataFrame(lc); B = pd.DataFrame(bs)
    L.to_csv("results/exp6_learning_curve.csv", index=False)
    B.to_csv("results/exp6_budget_search.csv", index=False)
    L["domain"] = L["arm"].str.split(":").str[0]
    B["domain"] = B["arm"].str.split(":").str[0]
    print("\n=== A. learning curve, Spearman by domain ===")
    print(L.pivot_table(index="n_train", columns=["domain", "model"],
                        values="spearman").round(3).to_string())
    print("\n=== B. recall of the true top 1% by domain ===")
    print(B.pivot_table(index="budget", columns=["domain", "strategy"],
                        values="recall").round(3).to_string())
    print("\nwrote results/exp6_learning_curve.csv, results/exp6_budget_search.csv")


if __name__ == "__main__":
    main()
