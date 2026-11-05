"""Experiment 9b -- give the language model its best shot.

exp9 mean-pooled ESM-2 over the whole sequence, which is the wrong thing to do
on a local mutational library: averaging 245 residues washes out the handful of
positions that differ between variants. Here the per-residue embeddings at the
variable positions are kept instead of averaged, which is the featurisation a
reviewer would ask for.

Kept compact by projecting the 480-dim residue embedding down with a PCA fitted
on a sample, so the design matrix stays a few thousand columns rather than
tens of thousands.
"""
import argparse, sys, time, os
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
from exp8_multi_target import build as build_real, project, DATASETS
from exp6_lattice_vs_real import load_lattice

HOLDOUT = 5_000
LAYER = 12
NPC = 48
CACHE = "data/esm"


@torch.no_grad()
def embed_positions(seqs, var, tag, batch=32):
    """Per-residue ESM-2 embeddings at `var` positions, PCA-reduced."""
    path = f"{CACHE}/{tag}_perpos.npz"
    if os.path.exists(path):
        return np.load(path)["X"]
    import esm
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval(); torch.set_num_threads(16)
    bc = alphabet.get_batch_converter()
    vp = np.asarray(sorted(var))
    chunks, t0 = [], time.time()
    for i in range(0, len(seqs), batch):
        pack = [(f"s{j}", s) for j, s in enumerate(seqs[i:i + batch])]
        _, _, toks = bc(pack)
        rep = model(toks, repr_layers=[LAYER])["representations"][LAYER]
        for k, (_, s) in enumerate(pack):
            r = rep[k, 1:len(s) + 1]
            sel = np.clip(vp, 0, r.shape[0] - 1)
            chunks.append(r[sel].numpy().astype(np.float16))
        if (i // batch) % 60 == 0:
            done = i + len(pack)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    [{tag}] {done:,}/{len(seqs):,} ({rate:.0f}/s, "
                  f"eta {(len(seqs)-done)/max(rate,1e-9)/60:.1f}m)", flush=True)
    A = np.stack(chunks)                                    # (n, |var|, 480)
    flat = A.reshape(-1, A.shape[-1]).astype(np.float32)
    sub = np.random.default_rng(0).choice(len(flat), size=min(200_000, len(flat)),
                                          replace=False)
    pca = PCA(n_components=NPC, random_state=0).fit(flat[sub])
    X = pca.transform(flat).reshape(A.shape[0], -1).astype(np.float32)
    np.savez_compressed(path, X=X)
    print(f"    [{tag}] per-position {X.shape} "
          f"(PCA {NPC}, var {pca.explained_variance_ratio_.sum():.2f}) "
          f"in {(time.time()-t0)/60:.1f}m", flush=True)
    return X


def curve(y, blocks, sizes, seeds, tag, rows):
    n = len(y)
    for s in seeds:
        perm = np.random.default_rng(1000 + s).permutation(n)
        hold = min(HOLDOUT, n // 4)
        test, pool = perm[:hold], perm[hold:]
        yte = y[test]
        for m in sizes:
            if m > len(pool):
                continue
            tr = pool[:m]
            got = {}
            for name, X in blocks.items():
                mdl = Ridge(alpha=1.0 if name == "onehot" else 50.0,
                            solver="lsqr").fit(X[tr], y[tr])
                p = (X[test] @ mdl.coef_ + mdl.intercept_
                     if sparse.issparse(X) else mdl.predict(X[test]))
                rho = round(float(spearmanr(yte, p).statistic), 4)
                rows.append(dict(arm=tag, features=name, n_train=m, seed=s,
                                 spearman=rho))
                got[name] = rho
            print(f"  [{tag:14s}] n={m:6,} s{s} " +
                  " ".join(f"{k}={v:+.3f}" for k, v in got.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[200, 500, 1000, 2000, 5000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lattice", nargs="+", default=["2DD8_S"])
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    rows = []

    for name, (ds, ag) in DATASETS.items():
        y, Xs, tok, vocab, antigen = build_real(name, ds, ag)
        df = pd.read_parquet(f"data/alphaseq/{ds}.parquet")
        df = df[df["matalpha_description"] == antigen]
        df = df[df["alphaseq_affinity"].notna()
                & df["sufficient_replicate_observations"] & df["above_background"]]
        seqs = (df.groupby("mata_sequence", as_index=False)
                  .agg(a=("alphaseq_affinity", "mean")))["mata_sequence"].tolist()
        ref = df["mata_sequence"].value_counts().idxmax()
        proj = np.array([list(project(s, ref)) for s in seqs])
        var = [j for j in range(proj.shape[1]) if len(set(proj[:, j])) > 1]
        Xp = embed_positions(seqs, var, f"real_{name}")
        z = np.load(f"{CACHE}/real_{name}.npz")
        curve(y, {"onehot": Xs, "esm_pooled": z["var"], "esm_perpos": Xp},
              a.sizes, seeds, f"real:{name}", rows)

    lib = pd.read_parquet("data/absolut/library.parquet")["cdr3"]
    for ag in a.lattice:
        yl, Xl, tokl, _ = load_lattice(ag, 30_000, np.random.default_rng(11))
        idx = np.sort(np.random.default_rng(11).choice(len(lib), 30_000,
                                                       replace=False))
        seqs = lib.values[idx].tolist()
        L = max(len(s) for s in seqs)
        Xp = embed_positions(seqs, list(range(min(L, 26))), f"lattice_{ag}")
        z = np.load(f"{CACHE}/lattice_{ag}.npz")
        curve(yl, {"onehot": Xl, "esm_pooled": z["var"], "esm_perpos": Xp},
              a.sizes, seeds, f"lattice:{ag}", rows)

    R = pd.DataFrame(rows)
    R.to_csv("results/exp9b_esm_perpos.csv", index=False, encoding="utf-8")
    R["domain"] = R["arm"].str.split(":").str[0]
    print("\n=== Spearman by feature set ===")
    print(R.pivot_table(index=["domain", "n_train"], columns="features",
                        values="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
