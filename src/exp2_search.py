"""Budget-constrained search of a fixed antibody library -- the OPAL analogue.

Library : Absolut!'s 6,917,820 murine CDRH3 sequences (fixed and designed,
          exactly like a synthetic phage-display library)
Antigen : one infectious-disease target
Oracle  : lattice binding energy, treated as an expensive measurement

Question: with a budget of B oracle calls out of ~7e6 library members, how good
a binder can you find, and does spending the budget iteratively beat spending
it all at once?

Screening is hierarchical, matching the proposed platform:
  stage 1  sparse ridge scores the entire library      (~1 s per full scan)
  stage 2  a max-pooling CNN reranks the stage-1 head  (semantics: best 11-mer)
"""
import argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.linear_model import Ridge

AA = "ACDEFGHIKLMNPQRSTVWY"
AAIDX = {a: i + 1 for i, a in enumerate(AA)}
MAXLEN = 26
RERANK = 200_000
torch.set_num_threads(16)


def encode(seqs):
    s = np.frombuffer("".join(x.ljust(MAXLEN)[:MAXLEN] for x in seqs).encode(),
                      dtype=np.uint8).reshape(-1, MAXLEN)
    lut = np.zeros(256, np.uint8)
    for a, i in AAIDX.items():
        lut[ord(a)] = i
    return lut[s]


class SliceCNN(nn.Module):
    def __init__(self, emb=32, ch=96, vocab=21):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(ch, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.conv(self.emb(x).transpose(1, 2)).amax(2)).squeeze(-1)


def fit_cnn(X, y, epochs=None, bs=128, seed=0, target_steps=4000, vocab=21):
    # fixed gradient-step budget so run time does not blow up with the sample size
    if epochs is None:
        epochs = int(np.clip(target_steps / max(1, len(X) / bs), 15, 80))
    torch.manual_seed(seed)
    m = SliceCNN(vocab=vocab)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    lf = nn.MSELoss()
    Xt = torch.from_numpy(X).long()
    mu, sd = float(y.mean()), float(y.std() + 1e-8)
    yt = torch.from_numpy(((y - mu) / sd).astype(np.float32))
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            k = perm[i:i + bs]
            opt.zero_grad(); lf(m(Xt[k]), yt[k]).backward(); opt.step()
    m.eval()
    return m, mu, sd


@torch.no_grad()
def cnn_score(m, Xenc, mu, sd, bs=100_000):
    out = np.empty(len(Xenc), np.float32)
    for i in range(0, len(Xenc), bs):
        out[i:i + bs] = m(torch.from_numpy(Xenc[i:i + bs]).long()).numpy() * sd + mu
    return out


def propose(measured_idx, energy, Xs, Xenc, k, rng, seed, use_cnn=True, vocab=21):
    """Hierarchical screen: ridge over the whole library, CNN over its head."""
    idx = np.asarray(measured_idx)
    y = energy[idx]
    ridge = Ridge(alpha=1.0, solver="lsqr").fit(Xs[idx], y)
    s1 = Xs @ ridge.coef_ + ridge.intercept_
    s1[idx] = np.inf
    n_head = min(RERANK, len(s1) - 1)          # small libraries have no head to cut
    head = np.argpartition(s1, n_head)[:n_head]
    if not use_cnn or len(idx) < 200:
        return head[np.argsort(s1[head])[:k]]
    m, mu, sd = fit_cnn(Xenc[idx], y, seed=seed, vocab=vocab)
    s2 = cnn_score(m, Xenc[head], mu, sd)
    return head[np.argsort(s2)[:k]]


def run(kind, energy, Xs, Xenc, budget, rng, seed, n_rounds=4, explore=0.2, vocab=21):
    N = energy.shape[0]
    if kind == "random":
        return rng.choice(N, size=budget, replace=False)

    n_init = max(200, budget // n_rounds)
    measured = list(rng.choice(N, size=n_init, replace=False))
    seen = set(measured)
    rounds = 1 if kind == "oneshot" else n_rounds - 1
    for r in range(rounds):
        take = (budget - n_init) // rounds if r < rounds - 1 else budget - len(measured)
        if take <= 0:
            break
        n_exp = int(take * (1 - explore)) if kind == "active" else take
        pick = propose(measured, energy, Xs, Xenc, n_exp, rng, seed + r, vocab=vocab)
        for i in pick:
            if i not in seen:
                seen.add(i); measured.append(i)
        while len(measured) < n_init + sum(
                (budget - n_init) // rounds for _ in range(r + 1)) and take > n_exp:
            cand = int(rng.integers(N))
            if cand not in seen:
                seen.add(cand); measured.append(cand)
    return np.asarray(measured[:budget])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--antigens", nargs="+", required=True)
    ap.add_argument("--budgets", nargs="+", type=int, default=[1000, 5000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--strategies", nargs="+", default=["random", "oneshot", "active"])
    ap.add_argument("--out", default="results/exp2_search.csv")
    a = ap.parse_args()

    lib = pd.read_parquet("data/absolut/library.parquet")["cdr3"].tolist()
    Xs = sparse.load_npz("data/absolut/library_features.npz").tocsr()
    Xenc = encode(lib)
    E = np.load("data/absolut/energies.npz")
    print(f"library {len(lib):,}  sparse {Xs.shape}  encoded {Xenc.shape}", flush=True)

    rows = []
    for ag in a.antigens:
        energy = E[ag]
        order = np.argsort(energy)
        top001 = set(order[:max(1, len(energy) // 10_000)].tolist())
        top01 = set(order[:len(energy) // 1_000].tolist())
        opt = float(energy.min())
        for budget in a.budgets:
            for kind in a.strategies:
                for s in range(a.seeds):
                    rng = np.random.default_rng(97 * s + budget)
                    t0 = time.time()
                    sel = run(kind, energy, Xs, Xenc, budget, rng, s)
                    e = energy[sel]
                    r = dict(antigen=ag, budget=budget, strategy=kind, seed=s,
                             best_energy=round(float(e.min()), 2),
                             gap=round(float(e.min()) - opt, 2),
                             best_pct=round(float((energy < e.min()).mean() * 100), 6),
                             n_top001pct=len(top001 & set(sel.tolist())),
                             n_top01pct=len(top01 & set(sel.tolist())),
                             secs=round(time.time() - t0, 1))
                    rows.append(r)
                    print(f"  {ag} B={budget:6d} {kind:8s} s{s} best={r['best_energy']:8.2f}"
                          f" (opt {opt:.2f}) pct={r['best_pct']:.5f}"
                          f" hits@0.01%={r['n_top001pct']:4d} @0.1%={r['n_top01pct']:5d}"
                          f" [{r['secs']}s]", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}\n")
    print(out.groupby(["budget", "strategy"])[
        ["best_energy", "gap", "best_pct", "n_top001pct", "n_top01pct", "secs"]]
        .mean().round(4).to_string())


if __name__ == "__main__":
    main()
