"""Is the prospective (iter0 -> iter1) drop a modelling failure or an assay batch effect?

Sequences measured in BOTH campaigns give a direct read on assay reproducibility
across batches. If that agreement is high, the drop is genuine generalisation
failure under design-distribution shift.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

reg = pd.read_parquet("data/alphaseq/panel_cov_reg.parquet")
ANT = [a for a, n in reg["matalpha_description"].value_counts().items() if n > 1000]

rows = []
print("=== assay reproducibility on sequences measured in both campaigns ===")
for ag in ANT:
    a = reg[(reg["matalpha_description"] == ag) & (reg["iteration"] == 0)]
    b = reg[(reg["matalpha_description"] == ag) & (reg["iteration"] == 1)]
    m = a.merge(b, on="mata_sequence", suffixes=("_0", "_1"))
    if len(m) < 50:
        print(f"  {ag[:32]:32s} only {len(m)} shared sequences -- skipped")
        continue
    rho = spearmanr(m["affinity_0"], m["affinity_1"]).statistic
    r = pearsonr(m["affinity_0"], m["affinity_1"]).statistic
    bias = float((m["affinity_1"] - m["affinity_0"]).mean())
    sd = float((m["affinity_1"] - m["affinity_0"]).std())
    rows.append(dict(antigen=ag, n_shared=len(m), spearman=round(rho, 3),
                     pearson=round(r, 3), batch_shift=round(bias, 3),
                     sd_of_diff=round(sd, 3)))
    print(f"  {ag[:32]:32s} n={len(m):5d} rho={rho:.3f} r={r:.3f} "
          f"shift={bias:+.3f} sd(diff)={sd:.3f}")

print("\n=== design-distribution shift: iter0 vs iter1 affinity ===")
for ag in ANT:
    a = reg[(reg["matalpha_description"] == ag) & (reg["iteration"] == 0)]["affinity"]
    b = reg[(reg["matalpha_description"] == ag) & (reg["iteration"] == 1)]["affinity"]
    if len(a) < 500 or len(b) < 500:
        continue
    print(f"  {ag[:32]:32s} iter0 med={a.median():.3f} iqr={a.quantile(.75)-a.quantile(.25):.3f}"
          f" | iter1 med={b.median():.3f} iqr={b.quantile(.75)-b.quantile(.25):.3f}"
          f" | shift={b.median()-a.median():+.3f}")

pd.DataFrame(rows).to_csv("results/exp1_batch_control.csv", index=False)
print("\nwrote results/exp1_batch_control.csv")
