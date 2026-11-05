"""Experiment 9 -- does a protein language model change the verdict?

The rank inversion in exp6/exp8 (ridge beats a max-pooling CNN on measured
data, the CNN wins on the lattice past ~5k labels) is only interesting if it
is not an artefact of using two weak featurisations. So the same protocol is
repeated with ESM-2 embeddings, on both the real targets and the matched
lattice libraries.

Two ESM featurisations are compared, since mean-pooling a 245-residue scFv can
wash out the handful of positions that actually differ between variants:
  esm_mean  mean over all residues
  esm_var   mean over the positions that vary within this library
"""
import argparse, sys, time, os
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
from exp8_multi_target import build as build_real, project, DATASETS
from exp6_lattice_vs_real import load_lattice, add_matched_noise, NOISE_SD

HOLDOUT = 5_000
LAYER = 12
CACHE = "data/esm"


def embedder():
    import esm
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()
    torch.set_num_threads(16)
    bc = alphabet.get_batch_converter()
    return model, bc


@torch.no_grad()
def embed(seqs, var_pos, tag, batch=32):
    """Mean-pooled ESM-2 representation, whole sequence and variable positions."""
    path = f"{CACHE}/{tag}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return z["mean"], z["var"]
    os.makedirs(CACHE, exist_ok=True)
    model, bc = embedder()
    M, V = [], []
    t0 = time.time()
    vp = np.asarray(sorted(var_pos)) if var_pos is not None else None
    for i in range(0, len(seqs), batch):
        chunk = [(f"s{j}", s) for j, s in enumerate(seqs[i:i + batch])]
        _, _, toks = bc(chunk)
        rep = model(toks, repr_layers=[LAYER])["representations"][LAYER]
        for k, (_, s) in enumerate(chunk):
            r = rep[k, 1:len(s) + 1]                    # drop BOS
            M.append(r.mean(0).numpy())
            if vp is not None:
                sel = vp[vp < r.shape[0]]
                V.append(r[sel].mean(0).numpy() if len(sel) else r.mean(0).numpy())
        if (i // batch) % 40 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    [{tag}] {done:,}/{len(seqs):,} ({rate:.0f}/s, "
                  f"eta {(len(seqs)-done)/max(rate,1e-9)/60:.1f}m)", flush=True)
    M = np.vstack(M).astype(np.float32)
    V = np.vstack(V).astype(np.float32) if V else M
    np.savez_compressed(path, mean=M, var=V)
    print(f"    [{tag}] cached {M.shape} in {(time.time()-t0)/60:.1f}m", flush=True)
    return M, V


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
            for name, X in blocks.items():
                alpha = 1.0 if name == "onehot" else 10.0
                mdl = Ridge(alpha=alpha, solver="lsqr").fit(X[tr], y[tr])
                p = (X[test] @ mdl.coef_ + mdl.intercept_
                     if sparse.issparse(X) else mdl.predict(X[test]))
                rows.append(dict(arm=tag, features=name, n_train=m, seed=s,
                                 spearman=round(float(spearmanr(yte, p).statistic), 4)))
            got = {r["features"]: r["spearman"] for r in rows[-len(blocks):]}
            print(f"  [{tag:14s}] n={m:6,} s{s} " +
                  " ".join(f"{k}={v:+.3f}" for k, v in got.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[200, 500, 1000, 2000, 5000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lattice", nargs="+", default=["2DD8_S", "1NCA_N"])
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    rows = []

    print("=== real targets ===", flush=True)
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
        M, V = embed(seqs, var, f"real_{name}")
        curve(y, {"onehot": Xs, "esm_mean": M, "esm_var": V},
              a.sizes, seeds, f"real:{name}", rows)

    print("\n=== lattice (matched size) ===", flush=True)
    lib = pd.read_parquet("data/absolut/library.parquet")["cdr3"]
    for ag in a.lattice:
        rng = np.random.default_rng(11)
        yl, Xl, tokl, vocabl = load_lattice(ag, 30_000, rng)
        # load_lattice draws with rng(11); reproduce the same subsample for sequences
        rng2 = np.random.default_rng(11)
        idx = np.sort(rng2.choice(len(lib), size=30_000, replace=False))
        seqs = lib.values[idx].tolist()
        M, V = embed(seqs, None, f"lattice_{ag}")
        curve(yl, {"onehot": Xl, "esm_mean": M, "esm_var": V},
              a.sizes, seeds, f"lattice:{ag}", rows)
        yn = add_matched_noise(yl, NOISE_SD / float(yl.std()) * 0 + 0.45,
                               np.random.default_rng(12))
        curve(yn, {"onehot": Xl, "esm_mean": M, "esm_var": V},
              a.sizes, seeds, f"lattice+noise:{ag}", rows)

    R = pd.DataFrame(rows)
    R.to_csv("results/exp9_esm.csv", index=False, encoding="utf-8")
    R["domain"] = R["arm"].str.split(":").str[0]
    print("\n=== Spearman by feature set ===")
    print(R.pivot_table(index=["domain", "n_train"], columns="features",
                        values="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
