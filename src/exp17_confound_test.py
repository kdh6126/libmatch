"""Experiment 17 -- the decisive confound test.

exp15 built a lattice library with the same structure as the real ones: all
single mutants of one parent CDRH3 plus random double and triple mutants,
scored by the Absolut! binary. So this library is synthetic (lattice labels)
but structurally matched to the measured data (one parent, aligned positions).

Running one-hot against ESM here separates two explanations for the
featurisation result we reported:

  if ESM wins  -> the earlier inversion tracked library STRUCTURE, not the
                  lattice; the benchmark-validity claim must be narrowed
  if one-hot wins -> the inversion tracked the LATTICE itself, and the claim
                  stands
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
from exp2_search import fit_cnn, cnn_score

AA = "ACDEFGHIKLMNPQRSTVWY-"
AAIDX = {a: i for i, a in enumerate(AA)}
PARENT = "CARDRLLWLRLWYFDVW"   # default; overridden by --parent
SIZES = [200, 500, 1000, 2000, 4000]
SEEDS = 5
HOLD = 1500
CACHE = "data/esm"
NPC = 48


def encode(seqs, ref):
    """Same length here, so alignment is the identity -- position i to position i."""
    L = len(ref)
    proj = np.array([list(s.ljust(L)[:L]) for s in seqs])
    var = [j for j in range(L) if len(set(proj[:, j])) > 1]
    X = np.zeros((len(seqs), len(var) * 21), np.float32)
    tok = np.zeros((len(seqs), len(var)), np.uint8)
    for c, j in enumerate(var):
        for i, ch in enumerate(proj[:, j]):
            k = AAIDX.get(ch, AAIDX["-"])
            X[i, c * 21 + k] = 1.0
            tok[i, c] = k + 1
    return sparse.csr_matrix(X), tok, var


@torch.no_grad()
def esm_perpos(seqs, var, tag, batch=64):
    path = f"{CACHE}/{tag}_perpos.npz"
    if os.path.exists(path):
        return np.load(path)["X"]
    import esm
    from sklearn.decomposition import PCA
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval(); torch.set_num_threads(16)
    bc = alphabet.get_batch_converter()
    vp = np.asarray(sorted(var))
    chunks = []
    for i in range(0, len(seqs), batch):
        pack = [(f"s{j}", s) for j, s in enumerate(seqs[i:i + batch])]
        _, _, toks = bc(pack)
        rep = model(toks, repr_layers=[12])["representations"][12]
        for k, (_, s) in enumerate(pack):
            r = rep[k, 1:len(s) + 1]
            chunks.append(r[np.clip(vp, 0, r.shape[0] - 1)].numpy().astype(np.float16))
        if (i // batch) % 20 == 0:
            print(f"    ESM {i + len(pack):,}/{len(seqs):,}", flush=True)
    A = np.stack(chunks)
    flat = A.reshape(-1, A.shape[-1]).astype(np.float32)
    sub = np.random.default_rng(0).choice(len(flat), min(200_000, len(flat)),
                                          replace=False)
    pca = PCA(n_components=NPC, random_state=0).fit(flat[sub])
    X = pca.transform(flat).reshape(A.shape[0], -1).astype(np.float32)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, X=X)
    return X


_ap = argparse.ArgumentParser()
_ap.add_argument("--antigen", default="2DD8_S")
_ap.add_argument("--parent", default=PARENT)
_ap.add_argument("--out", default=None)
_a, _ = _ap.parse_known_args()
PARENT = _a.parent
d = pd.read_parquet(f"data/absolut/matched_{_a.antigen}.parquet")
d = d[d.cdr3.str.len() == len(PARENT)].reset_index(drop=True)
y = d.energy.to_numpy(np.float32)
print(f"구조를 맞춘 격자 라이브러리: {len(d):,}개 (모항체 {PARENT})")
print(f"에너지 {y.min():.2f} ~ {y.max():.2f}, SD {y.std():.3f}")

Xs, tok, var = encode(d.cdr3.tolist(), PARENT)
print(f"가변 위치 {len(var)}개")
Xesm = esm_perpos(d.cdr3.tolist(), var, f"matched_{_a.antigen}")
print(f"ESM 위치별 표현 {Xesm.shape}\n")

rows = []
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
        c, mu, sd = fit_cnn(tok[tr], y[tr], seed=s, vocab=22)
        p_cnn = cnn_score(c, tok[te], mu, sd)
        rho = {k: float(spearmanr(yte, v).statistic)
               for k, v in [("onehot", p_oh), ("esm", p_es), ("cnn", p_cnn)]}
        rows.append(dict(n_train=m, seed=s, **{k: round(v, 4) for k, v in rho.items()},
                         onehot_minus_esm=round(rho["onehot"] - rho["esm"], 4)))
        print(f"  n={m:5,} s{s}  onehot={rho['onehot']:+.3f}  esm={rho['esm']:+.3f}  "
              f"cnn={rho['cnn']:+.3f}   차이(oh-esm)={rho['onehot']-rho['esm']:+.3f}",
              flush=True)

R = pd.DataFrame(rows)
R.to_csv(_a.out or f"results/exp17_confound_{_a.antigen}.csv", index=False, encoding="utf-8")
piv = R.groupby("n_train")[["onehot", "esm", "cnn", "onehot_minus_esm"]].mean().round(3)
piv["승자"] = np.where(piv["onehot_minus_esm"] > 0, "one-hot", "ESM")
print("\n=== 구조를 맞춘 격자에서의 결과 ===")
print(piv.to_string())
w = (R.onehot_minus_esm > 0).sum()
print(f"\none-hot 승 {w}/{len(R)} 구간, 평균 차이 {R.onehot_minus_esm.mean():+.4f}")
print("\n[해석] 이전 격자(무관한 CDR3 690만)에서는 one-hot이 5/5 구간 승이었다.")
print("       여기서도 one-hot이 이기면 → 격자 자체의 성질 (주장 유지)")
print("       여기서 ESM이 이기면      → 라이브러리 구조 탓 (주장 축소 필요)")
