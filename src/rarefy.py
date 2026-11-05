"""Diversity comparisons across panning rounds need equal sequencing depth.

VH round 4 was sequenced to 2.6M reads against 1.2M for round 3, which alone
raises the observed Shannon entropy and lowers the top-10 share. Every round is
therefore subsampled to the shallowest round before the statistics are taken.
"""
import numpy as np
import pandas as pd

def rarefied_stats(counts, depth, reps=5, seed=0):
    """Shannon H and top-10 share after subsampling `counts` to `depth` reads."""
    rng = np.random.default_rng(seed)
    c = np.asarray(counts, dtype=np.int64)
    c = c[c > 0]
    p = c / c.sum()
    Hs, shares = [], []
    for _ in range(reps):
        s = rng.multinomial(depth, p)
        s = s[s > 0]
        q = s / s.sum()
        Hs.append(float(-(q * np.log(q)).sum()))
        shares.append(float(np.sort(s)[-10:].sum() / s.sum() * 100))
    return float(np.mean(Hs)), float(np.std(Hs)), float(np.mean(shares))


def round_table(cnt, chain, exclude=(("VL", 1),), min_count=5):
    d = cnt[cnt.chain == chain]
    w = d.pivot_table(index="cdr3", columns="round", values="count",
                      aggfunc="sum", fill_value=0)
    rounds = [r for r in sorted(w.columns) if (chain, r) not in exclude]
    depths = {r: int(w[r].sum()) for r in rounds}
    # drop clonotypes seen fewer than `min_count` times: most singletons are
    # PCR / sequencing error variants and they dominate the raw entropy
    filt = {r: w[r].values[w[r].values >= min_count] for r in rounds}
    common = min(int(v.sum()) for v in filt.values())
    rows = []
    for r in rounds:
        H, sd, share = rarefied_stats(filt[r], common)
        rows.append(dict(chain=chain, round=r, reads=depths[r],
                         min_count=min_count, rarefied_to=common,
                         shannon=round(H, 3), shannon_sd=round(sd, 3),
                         top10_share=round(share, 2),
                         clonotypes_kept=int(len(filt[r])),
                         singleton_frac=round(float((w[r].values[w[r].values > 0] == 1).mean()), 3)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cnt = pd.read_parquet("data/panning_cmet/cdr3_counts.parquet")
    out = pd.concat([round_table(cnt, c) for c in ["VH", "VL"]], ignore_index=True)
    out.to_csv("results/panning_diversity_rarefied.csv", index=False)
    print(out.to_string(index=False))
