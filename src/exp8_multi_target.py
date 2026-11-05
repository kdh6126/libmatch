"""Experiment 8 -- widen the real arm beyond the coronavirus RBD panel.

exp6 compared lattice against real on one target family, which leaves the
obvious objection that the conclusion is a property of VHH72 rather than of
measured data. open-alphaseq also carries anti-TIGIT, anti-PD-1 and
anti-HER2 campaigns, each a different parent antibody in a different format,
so the same protocol runs on four independent target families.

Each dataset is featurised on its own terms: variants are aligned to the most
common (wild-type) sequence in that campaign, and the positions that actually
vary become the one-hot block.
"""
import argparse, sys, time
import numpy as np
import pandas as pd
from Bio import Align
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
from exp2_search import fit_cnn, cnn_score, run

AA = "ACDEFGHIKLMNPQRSTVWY-"
AAIDX = {a: i for i, a in enumerate(AA)}
HOLDOUT = 5_000

DATASETS = {
    "CoV2-RBD":  ("YM_1068", "SARS-CoV2_RBD_(6LZG)"),
    "TIGIT":     ("YM_0988", None),
    "PD-1":      ("YM_0985", None),
    "HER2":      ("YM_0990", None),
}

aligner = Align.PairwiseAligner(mode="global", open_gap_score=-10,
                                extend_gap_score=-0.5, match_score=2,
                                mismatch_score=-1)


def project(seq, ref):
    aln = aligner.align(ref, seq)[0]
    out = ["-"] * len(ref)
    for (r0, r1), (q0, q1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(r1 - r0):
            out[r0 + k] = seq[q0 + k]
    return "".join(out)


def build(name, dataset, antigen):
    df = pd.read_parquet(f"data/alphaseq/{dataset}.parquet")
    df = df[df["matalpha_sequence"].notna() & df["alphaseq_affinity"].notna()]
    if antigen:
        df = df[df["matalpha_description"] == antigen]
    else:                                   # the antigen with the most measurements
        antigen = df["matalpha_description"].value_counts().idxmax()
        df = df[df["matalpha_description"] == antigen]
    df = df[df["sufficient_replicate_observations"] & df["above_background"]]
    g = (df.groupby("mata_sequence", as_index=False)
           .agg(affinity=("alphaseq_affinity", "mean"),
                n=("alphaseq_affinity", "size")))
    ref = df["mata_sequence"].value_counts().idxmax()

    proj = np.array([list(project(s, ref)) for s in g["mata_sequence"]])
    var = [j for j in range(proj.shape[1]) if len(set(proj[:, j])) > 1]
    X = np.zeros((len(g), len(var) * 21), np.float32)
    tok = np.zeros((len(g), len(var)), np.uint8)
    for c, j in enumerate(var):
        col = proj[:, j]
        for i, ch in enumerate(col):
            k = AAIDX.get(ch, AAIDX["-"])
            X[i, c * 21 + k] = 1.0
            tok[i, c] = k + 1               # 0 reserved for padding
    y = g["affinity"].to_numpy().astype(np.float32)
    print(f"[{name}] {antigen[:38]:38s} n={len(g):,}  ref_len={len(ref)}  "
          f"variable={len(var)}  SD={y.std():.3f}", flush=True)
    return y, sparse.csr_matrix(X), tok, 22, antigen


def learning_curve(y, Xs, tok, vocab, sizes, seeds, tag, rows):
    n = len(y)
    for s in seeds:
        perm = np.random.default_rng(1000 + s).permutation(n)
        hold = min(HOLDOUT, n // 4)
        test, pool = perm[:hold], perm[hold:]
        yte = y[test]
        true_top = set(np.argsort(yte)[:100].tolist())
        for m in sizes:
            if m > len(pool):
                continue
            tr = pool[:m]
            ridge = Ridge(alpha=1.0, solver="lsqr").fit(Xs[tr], y[tr])
            p_r = Xs[test] @ ridge.coef_ + ridge.intercept_
            cnn, mu, sd = fit_cnn(tok[tr], y[tr], seed=s, vocab=vocab)
            p_c = cnn_score(cnn, tok[test], mu, sd)
            for model, p in [("ridge", p_r), ("cnn", p_c)]:
                got = len(true_top & set(np.argsort(p)[:500].tolist())) / 100
                rows.append(dict(arm=tag, model=model, n_train=m, seed=s,
                                 spearman=round(float(spearmanr(yte, p).statistic), 4),
                                 top100_at_500=round(got, 3)))
            print(f"  [{tag:10s}] n={m:6,} s{s} ridge={rows[-2]['spearman']:+.3f} "
                  f"cnn={rows[-1]['spearman']:+.3f}", flush=True)


def budget_search(y, Xs, tok, vocab, budgets, seeds, tag, rows):
    n = len(y)
    n1 = max(1, n // 100)
    top1 = set(np.argsort(y)[:n1].tolist())
    for budget in budgets:
        if budget > n // 3:
            continue
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
                                 secs=round(time.time() - t0, 1)))
                print(f"  [{tag:10s}] B={budget:5,} {kind:8s} s{s} "
                      f"recall={hits/n1:.3f} [{rows[-1]['secs']}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[200, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--budgets", nargs="+", type=int, default=[500, 1000, 2000, 4000])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    lc, bs = [], []
    for name, (ds, ag) in DATASETS.items():
        y, Xs, tok, vocab, antigen = build(name, ds, ag)
        learning_curve(y, Xs, tok, vocab, a.sizes, seeds, name, lc)
        budget_search(y, Xs, tok, vocab, a.budgets, seeds, name, bs)

    L, B = pd.DataFrame(lc), pd.DataFrame(bs)
    L.to_csv("results/exp8_learning_curve.csv", index=False, encoding="utf-8")
    B.to_csv("results/exp8_budget_search.csv", index=False, encoding="utf-8")
    print("\n=== learning curve, Spearman (ridge) ===")
    print(L[L.model == "ridge"].pivot_table(index="n_train", columns="arm",
                                            values="spearman").round(3).to_string())
    print("\n=== recall of the true top 1% ===")
    print(B.pivot_table(index="budget", columns=["arm", "strategy"],
                        values="recall").round(3).to_string())


if __name__ == "__main__":
    main()
