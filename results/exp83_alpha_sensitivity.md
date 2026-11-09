# Ridge penalty sensitivity

Pre-registered log grid around the fixed penalties: one-hot [0.1, 1.0, 10.0, 100.0], ESM [5.0, 50.0, 500.0]. Splits follow the rule used throughout (`default_rng(1000 + s)`, 1500 held out), with 20 seeds. The grid was fixed before the results were seen and no penalty is re-selected from it.

The language model is ahead wherever the difference is negative. At the fixed penalties it is ahead in 21 of 35 (antigen, budget) cells, and in 14 of 14 at the budgets of 200 and 500 where the article makes the claim.

## Whether the fixed penalties favoured the conclusion

Mean held-out Spearman over the seven antigens, by penalty. The fixed values are one-hot 1 and ESM 50.

| Labels | onehot a=0.1 | onehot a=1 | onehot a=10 | onehot a=100 | esm a=5 | esm a=50 | esm a=500 |
|---|---|---|---|---|---|---|---|
| 200 | 0.7558 | 0.7903 | 0.7457 | 0.6961 | 0.8497 | 0.8415 | 0.7807 |
| 500 | 0.8470 | 0.8695 | 0.8157 | 0.7353 | 0.8839 | 0.8844 | 0.8203 |
| 1000 | 0.9017 | 0.9052 | 0.8647 | 0.7604 | 0.9086 | 0.9069 | 0.8483 |
| 2000 | 0.9221 | 0.9225 | 0.9024 | 0.7912 | 0.9246 | 0.9211 | 0.8753 |
| 4000 | 0.9309 | 0.9309 | 0.9240 | 0.8320 | 0.9333 | 0.9297 | 0.8999 |

One-hot does best at its fixed penalty of 1 at every budget, so the fixed value did not handicap it. ESM does slightly better at 5 than at its fixed 50 at 200 labels, so the fixed value if anything understates its advantage where that advantage is largest. The direction reverses only at a=500 for ESM, which is the edge of the grid and costs it far more than its neighbours; the matching edge for one-hot, a=100, costs one-hot more still. An edge penalty that cripples one side is a property of the penalty, not of the comparison.

## Per antigen and budget

The last two columns give one-hot its best penalty on the grid and ESM its worst, both chosen post hoc on the same held-out set, which is an advantage neither would have in practice.

| Antigen | Labels | one-hot (a=1) | ESM (a=50) | diff | one-hot best | ESM worst | diff |
|---|---|---|---|---|---|---|---|
| 1ADQ_A | 200 | +0.743 | +0.780 | -0.036 | +0.743 (a=1) | +0.717 (a=500) | +0.027 |
| 1NCA_N | 200 | +0.800 | +0.854 | -0.054 | +0.800 (a=1) | +0.797 (a=500) | +0.003 |
| 1OSP_O | 200 | +0.806 | +0.867 | -0.061 | +0.806 (a=1) | +0.828 (a=500) | -0.022 |
| 1ZTX_E | 200 | +0.873 | +0.929 | -0.056 | +0.884 (a=0.1) | +0.864 (a=500) | +0.020 |
| 2DD8_S | 200 | +0.711 | +0.748 | -0.036 | +0.711 (a=1) | +0.673 (a=500) | +0.039 |
| 2Q8A_A | 200 | +0.788 | +0.872 | -0.083 | +0.788 (a=1) | +0.816 (a=500) | -0.027 |
| 3VRL_C | 200 | +0.810 | +0.841 | -0.031 | +0.810 (a=1) | +0.771 (a=500) | +0.039 |
| 1ADQ_A | 500 | +0.804 | +0.831 | -0.026 | +0.804 (a=1) | +0.757 (a=500) | +0.047 |
| 1NCA_N | 500 | +0.884 | +0.894 | -0.010 | +0.884 (a=1) | +0.836 (a=500) | +0.048 |
| 1OSP_O | 500 | +0.878 | +0.897 | -0.019 | +0.878 (a=1) | +0.853 (a=500) | +0.026 |
| 1ZTX_E | 500 | +0.957 | +0.959 | -0.002 | +0.965 (a=0.1) | +0.902 (a=500) | +0.063 |
| 2DD8_S | 500 | +0.802 | +0.820 | -0.018 | +0.802 (a=1) | +0.725 (a=500) | +0.077 |
| 2Q8A_A | 500 | +0.878 | +0.903 | -0.026 | +0.878 (a=1) | +0.853 (a=500) | +0.025 |
| 3VRL_C | 500 | +0.883 | +0.887 | -0.003 | +0.883 (a=1) | +0.816 (a=500) | +0.067 |
| 1ADQ_A | 1000 | +0.844 | +0.858 | -0.014 | +0.844 (a=1) | +0.789 (a=500) | +0.055 |
| 1NCA_N | 1000 | +0.915 | +0.913 | +0.003 | +0.915 (a=1) | +0.860 (a=500) | +0.056 |
| 1OSP_O | 1000 | +0.915 | +0.916 | -0.002 | +0.915 (a=1) | +0.872 (a=500) | +0.043 |
| 1ZTX_E | 1000 | +0.982 | +0.974 | +0.007 | +0.984 (a=0.1) | +0.926 (a=500) | +0.057 |
| 2DD8_S | 1000 | +0.852 | +0.859 | -0.007 | +0.852 (a=1) | +0.768 (a=500) | +0.083 |
| 2Q8A_A | 1000 | +0.914 | +0.919 | -0.004 | +0.914 (a=1) | +0.876 (a=500) | +0.038 |
| 3VRL_C | 1000 | +0.915 | +0.910 | +0.005 | +0.915 (a=1) | +0.846 (a=500) | +0.069 |
| 1ADQ_A | 2000 | +0.870 | +0.875 | -0.006 | +0.870 (a=1) | +0.822 (a=500) | +0.047 |
| 1NCA_N | 2000 | +0.931 | +0.926 | +0.004 | +0.931 (a=1) | +0.884 (a=500) | +0.047 |
| 1OSP_O | 2000 | +0.931 | +0.929 | +0.002 | +0.931 (a=0.1) | +0.891 (a=500) | +0.040 |
| 1ZTX_E | 2000 | +0.988 | +0.982 | +0.005 | +0.988 (a=0.1) | +0.948 (a=500) | +0.041 |
| 2DD8_S | 2000 | +0.878 | +0.879 | -0.001 | +0.878 (a=1) | +0.811 (a=500) | +0.067 |
| 2Q8A_A | 2000 | +0.930 | +0.930 | +0.001 | +0.931 (a=0.1) | +0.898 (a=500) | +0.033 |
| 3VRL_C | 2000 | +0.930 | +0.926 | +0.004 | +0.930 (a=1) | +0.873 (a=500) | +0.056 |
| 1ADQ_A | 4000 | +0.883 | +0.886 | -0.003 | +0.883 (a=1) | +0.855 (a=500) | +0.028 |
| 1NCA_N | 4000 | +0.939 | +0.935 | +0.003 | +0.939 (a=1) | +0.905 (a=500) | +0.034 |
| 1OSP_O | 4000 | +0.939 | +0.937 | +0.002 | +0.939 (a=0.1) | +0.910 (a=500) | +0.029 |
| 1ZTX_E | 4000 | +0.990 | +0.986 | +0.004 | +0.990 (a=0.1) | +0.965 (a=500) | +0.025 |
| 2DD8_S | 4000 | +0.893 | +0.893 | +0.000 | +0.893 (a=1) | +0.852 (a=500) | +0.041 |
| 2Q8A_A | 4000 | +0.936 | +0.935 | +0.001 | +0.936 (a=0.1) | +0.914 (a=500) | +0.022 |
| 3VRL_C | 4000 | +0.938 | +0.935 | +0.003 | +0.938 (a=1) | +0.899 (a=500) | +0.038 |
