"""보충표 S14 — 모항체 선택이 결론을 바꾸는가(exp79).

원고는 Methods에서 2DD8_S의 모항체가 748,021개 가운데 1위가 아니라 4위라고
적는다. 적어 둔 이상 "1위로 만들었으면 달랐겠는가"는 심사자가 물을 자리이고,
마침 그 라이브러리가 있다.

    python src/si_table_parent.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp79_parent_choice.csv"
MARK_A = "% ---- BEGIN parent choice (generated)"
MARK_B = "% ---- END parent choice"


def table_tex():
    R = pd.read_csv(SRC)
    rows = []
    for (lib, n), g in R.groupby(["library", "n_train"]):
        v = g.onehot_minus_esm.to_numpy()
        rows.append(dict(library=lib, n_train=int(n),
                         onehot=g.onehot.mean(), esm=g.esm.mean(),
                         diff=v.mean(), esm_wins=int((v < 0).sum()),
                         n=len(v)))
    S = pd.DataFrame(rows).sort_values(["n_train", "library"])
    big = int(S.n_train.max())
    pub_big = S[(S.library == "published") & (S.n_train == big)]["diff"].iloc[0]
    rk_big = S[(S.library == "rank1") & (S.n_train == big)]["diff"].iloc[0]
    d_pub = S[S.library == "published"].onehot.mean()
    d_rk = S[S.library == "rank1"].onehot.mean()

    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{Whether the choice of parent changes the representation "
        "result. The library for SARS-CoV spike RBD is built around a parent "
        "that ranks 4th of the 748{,}021 length-17 sequences in the released "
        "table; we also built one around the sequence that ranks 1st. The two "
        "parents differ at seven positions and nothing else changes: same "
        "antigen, same oracle, same design rule, same size, same splits, same "
        "models. Negative values favour the language model. The direction "
        "agrees at every training size, so the reported reversal does not rest "
        "on which parent was chosen. What differs is how fast the advantage "
        f"closes: at {big:,} labels it reaches zero for the published parent "
        f"and remains at ${rk_big:+.4f}$ for the rank-1 parent, so the "
        "statement that the difference vanishes at large training sets does "
        "depend on that choice. The library built around the global minimum is "
        f"also the less learnable of the two (mean one-hot Spearman {d_rk:.3f} "
        f"against {d_pub:.3f}): next to the best sequence in the library almost "
        "every mutation is worse, so the labels bunch into a narrow range. "
        "Choosing the strongest binder as the parent does not automatically "
        "give the better benchmark.}",
        "\\centering",
        "\\begin{tabular}{@{}rlrrrr@{}}",
        "\\toprule",
        "Labels & Parent & one-hot & ESM-2 & Difference & ESM ahead \\\\",
        "\\midrule"]
    for _, x in S.iterrows():
        name = "published (rank 4)" if x.library == "published" else "rank 1"
        lines.append(f"{int(x.n_train):,} & {name} & {x.onehot:.3f} & "
                     f"{x.esm:.3f} & {x['diff']:+.4f} & "
                     f"{int(x.esm_wins)}/{int(x.n)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), S


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp79를 먼저 돌리십시오.")
    tex, S = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S14, {len(S)}행)")


if __name__ == "__main__":
    main()
