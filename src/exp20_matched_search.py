"""Experiment 20 -- how large is the search advantage once structure is matched?

exp2 reported a very large advantage for guided search over random sampling on
the lattice (378 of the top 691 at budget 2,000, versus ~0 for random). exp6/exp8
found a much smaller advantage on measured data (2.9-3.4x). The lattice library
there was 6.9M unrelated CDRH3s; the measured libraries were mutational
neighbourhoods of ~30k.

Two things differ at once again -- structure and size. This holds the oracle
fixed (Absolut!) and uses the structurally matched mutational library from
exp15, at sizes comparable to the measured libraries. If the advantage drops to
the measured range, the headline number was about library structure, not about
the benchmark.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from exp17_confound_test import encode, PARENT
from exp8_multi_target import build as build_real, DATASETS
from exp2_search import run

BUDGETS = [200, 500, 1000]
SEEDS = 5


def sweep(y, Xs, tok, vocab, tag, rows):
    n = len(y)
    n1 = max(1, n // 100)                      # top 1%
    top1 = set(np.argsort(y)[:n1].tolist())
    opt = float(y.min())
    for b in BUDGETS:
        if b > n // 3:
            continue
        for kind in ["random", "oneshot", "active"]:
            for s in range(SEEDS):
                sel = run(kind, y, Xs, tok, b,
                          np.random.default_rng(97 * s + b), s, vocab=vocab)
                hits = len(top1 & set(sel.tolist()))
                rows.append(dict(arm=tag, lib_size=n, budget=b, strategy=kind,
                                 seed=s, hits=hits, n_top1pct=n1,
                                 recall=round(hits / n1, 4),
                                 best=round(float(y[sel].min()), 3),
                                 gap=round(float(y[sel].min()) - opt, 3)))
        r = pd.DataFrame(rows[-3 * SEEDS:])
        m = r.groupby("strategy")["recall"].mean()
        print(f"  [{tag:22s}] B={b:5,}  능동={m.get('active', 0):.3f} "
              f"일회성={m.get('oneshot', 0):.3f} 무작위={m.get('random', 0):.3f}"
              f"  → 능동/무작위 = {m.get('active',0)/max(m.get('random',1e-9),1e-9):.1f}배",
              flush=True)


rows = []

# ---- structurally matched lattice ----
d = pd.read_parquet("data/absolut/matched_2DD8_S.parquet")
d = d[d.cdr3.str.len() == len(PARENT)].reset_index(drop=True)
yl = d.energy.to_numpy(np.float32)
Xl, tokl, _ = encode(d.cdr3.tolist(), PARENT)
print(f"구조를 맞춘 격자 {len(yl):,}개 (모항체 변이체, 오라클은 Absolut!)")
sweep(yl, Xl, tokl, 22, "matched_lattice", rows)

# ---- measured libraries, subsampled to a comparable size ----
for name, (ds, ag) in DATASETS.items():
    y, Xs, tok, vocab, antigen = build_real(name, ds, ag)
    sub = np.sort(np.random.default_rng(4242).permutation(len(y))[:len(yl)])
    print(f"\n실측 {name} — {len(yl):,}개로 맞춤 (전체 {len(y):,})")
    sweep(y[sub], Xs[sub], tok[sub], vocab, f"real:{name}", rows)

R = pd.DataFrame(rows)
R.to_csv("results/exp20_matched_search.csv", index=False, encoding="utf-8")
R["domain"] = np.where(R.arm.str.startswith("real"), "real", "matched_lattice")

print("\n=== 상위 1% 회수율 (라이브러리 크기·구조를 맞춘 뒤) ===")
piv = R.pivot_table(index="budget", columns=["domain", "strategy"],
                    values="recall").round(3)
print(piv.to_string())

print("\n=== 능동 / 무작위 배수 ===")
for dom in ["matched_lattice", "real"]:
    s = R[R.domain == dom]
    for b in sorted(s.budget.unique()):
        a = s[(s.budget == b) & (s.strategy == "active")].recall.mean()
        r = s[(s.budget == b) & (s.strategy == "random")].recall.mean()
        print(f"  {dom:16s} B={b:5,}  {a/max(r,1e-9):5.1f}배")
print("\n[참고] exp2의 무관한 CDR3 690만 라이브러리에서는 이 배수가 수백~수천이었다.")
print("wrote results/exp20_matched_search.csv")
