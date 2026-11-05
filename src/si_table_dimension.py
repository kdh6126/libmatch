"""보충표 S11 — 설계 공간의 차원만 바꾸면 무엇이 달라지는가(exp76).

본문은 정합 후에도 탐색 우위가 남는다고 적고, 그 몫이 대리 모형의 적합도를
따라간다고 적는다. 그 사슬의 뒷마디 -- 자리 수가 적합도를 정한다 -- 를 한
항원 안에서 확인한 표다.

두 가지를 함께 싣는다. 자리를 무작위로 여러 번 뽑은 판(exp76)이 결과이고,
자리를 "서열을 가장 많이 담는" 쪽으로 고른 판(exp74)은 그렇게 고르면 왜
안 되는지를 보여 준다. 실패한 설계를 숨기면 독자가 같은 함정에 빠진다.

    python src/si_table_dimension.py
"""
import io
import os
import sys

import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp76_random_positions.csv"
SRC74 = "results/exp74_ratios.csv"
MARK_A = "% ---- BEGIN dimension (generated)"
MARK_B = "% ---- END dimension"


def table_tex():
    R = pd.read_csv(SRC).sort_values(["antigen", "k"], ascending=[True, False])
    gain = (R[R.k == R.k.min()].rho.mean() - R[R.k == R.k.max()].rho.mean())
    spread = R[R.k < R.k.max()].rho_sd.mean()
    n_ag = R.antigen.nunique()
    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{Whether narrowing the design space makes a library easier to "
        "learn. Within one matched library the number of varying positions was "
        "reduced from 15 to 11 and 8, holding the antigen, the oracle, the "
        "library size and the mixture of single, double and triple mutants "
        "fixed. The positions were drawn at random and the result averaged over "
        "twelve draws, so which positions are chosen cancels; the surrogate is "
        "the same ridge regression used elsewhere, fitted on 400 labels over "
        "twenty splits. Held-out Spearman rises as the space narrows in all "
        f"{n_ag} antigens, by {gain:+.3f} on average from 15 positions to 8. "
        f"Two limits apply. The draw-to-draw standard deviation is {spread:.3f}, "
        "so no single antigen is conclusive and the agreement of all "
        f"{n_ag} is the evidence (sign test $p = 0.125$). And the effect is an "
        "order of magnitude smaller than the gap it is meant to explain: the "
        "matched lattice and the measured libraries differ by 0.2 to 0.4 in the "
        "same quantity. Dimension has the right sign and the wrong size; the "
        "remainder is presumably measurement noise, whose cost we measure "
        "separately as 0.105.}",
        "\\centering",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Antigen & Varying positions & Position sets & Held-out Spearman & "
        "Range over draws \\\\",
        "\\midrule"]
    for ag, g in R.groupby("antigen", sort=True):
        for i, (_, x) in enumerate(g.iterrows()):
            name = ag.replace("_", "\\_") if i == 0 else ""
            sd = "" if x.rho_sd == 0 else f" $\\pm$ {x.rho_sd:.3f}"
            rng = ("---" if x.rho_sd == 0
                   else f"{x.rho_lo:.3f} to {x.rho_hi:.3f}")
            lines.append(f"{name} & {int(x.k)} & {int(x.draws)} & "
                         f"{x.rho:.3f}{sd} & {rng} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]

    if os.path.exists(SRC74):
        d = pd.read_csv(SRC74)
        d = d[d.metric == "recall_1pct"].sort_values(["arm", "k"],
                                                     ascending=[True, False])
        lines += [
            "",
            "\\begin{table}[h]",
            "\\caption{The same question asked of guided search, and why it "
            "cannot be answered that way. Running the search comparison on the "
            "subsets above gives no monotone trend in any antigen. Two features "
            "of the design account for it. The common subset size forces a "
            "budget of 400 out of about 1,460, so random search already "
            "recovers a quarter of the top 1\\% and the ratio cannot exceed "
            "about 3.7; the highest cell reaches an active recovery of 0.90 and "
            "is against that ceiling. And in this table the position sets were "
            "chosen to contain as many sequences as possible rather than at "
            "random, so the identity of the positions varies with their number. "
            "The preceding table removes both problems by measuring "
            "learnability, which has no ceiling, over randomly drawn position "
            "sets. We report this attempt because the ceiling is easy to walk "
            "into.}",
            "\\centering",
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Antigen & Varying positions & Active & Random & Ratio (95\\% CI) "
            "\\\\",
            "\\midrule"]
        for arm, g in d.groupby("arm", sort=True):
            for i, (_, x) in enumerate(g.iterrows()):
                name = (arm.split(":")[1].replace("_", "\\_") if i == 0 else "")
                lines.append(f"{name} & {int(x.k)} & {x.active:.3f} & "
                             f"{x['random']:.3f} & {x.ratio:.2f} "
                             f"({x.ci_lo:.2f}--{x.ci_hi:.2f}) \\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines += ["\\end{tabular}", "\\end{table}"]
    lines.append(MARK_B)
    return "\n".join(lines), R


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp76을 먼저 돌리십시오.")
    tex, R = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S11, 항원 {R.antigen.nunique()}종)")


if __name__ == "__main__":
    main()
