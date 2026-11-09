"""보충표 — ridge 벌점이 표현 비교의 결론을 만들었는가(exp83).

심사자 4가 정규화를 고른 홀드아웃의 구성을 물었고, 답은 그런 절차가 없다는
것이다. 그러면 다음 질문이 남는다 -- 고정값 두 개가 결론을 만든 것은 아닌가.

표 하나로 답한다. 고정값 주위의 미리 정한 로그 격자에서 두 표현의 held-out
Spearman 평균을 예산별로 적고, 고정 벌점에서 언어 모형이 앞선 항원 수를 함께
적는다. 고정값이 어느 쪽에 유리했는지가 한눈에 보인다.

    python src/si_table_alpha.py
"""
import io
import os
import sys

import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp83_alpha_sensitivity.csv"
SUM = "results/exp83_alpha_summary.csv"
MARK_A = "% ---- BEGIN alpha sensitivity (generated)"
MARK_B = "% ---- END alpha sensitivity"
ALPHA_ONEHOT = [0.1, 1.0, 10.0, 100.0]
ALPHA_ESM = [5.0, 50.0, 500.0]
FIXED_ONEHOT = 1.0
FIXED_ESM = 50.0


def table_tex():
    R = pd.read_csv(SRC)
    S = pd.read_csv(SUM)
    G = R.groupby(["n_train", "rep", "alpha"]).rho.mean()
    sizes = sorted(R.n_train.unique())
    seeds = int(R.seed.nunique())
    n_ag = int(R.antigen.nunique())

    ahead = {int(m): int(g.esm_ahead_fixed.sum())
             for m, g in S.groupby("n_train")}
    small = [m for m in sizes if m <= 500]
    n_small = sum(ahead[m] for m in small)

    head = ("\\multicolumn{%d}{c}{One-hot} & \\multicolumn{%d}{c}{ESM-2} & \\\\"
            % (len(ALPHA_ONEHOT), len(ALPHA_ESM)))
    cols = " & ".join(["$\\alpha=%g$" % a for a in ALPHA_ONEHOT]
                      + ["$\\alpha=%g$" % a for a in ALPHA_ESM])

    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{Whether the fixed ridge penalties produced the "
        "representation result. The penalties are constants in the code "
        "($\\alpha = 1$ for one-hot, $\\alpha = 50$ for ESM-2) and are not "
        "tuned for antigens, label budgets, splits or seeds, so there is no "
        "selection procedure to report; what can be reported is whether the "
        "two values favoured the conclusion. The independent unit is the "
        "antigen: each cell is the held-out Spearman correlation averaged "
        "over the %d matched libraries and %d splits, with the splits, "
        "held-out size, training sizes and seeds identical to those used "
        "throughout---only the penalty changes. This grid is a sensitivity "
        "analysis and nothing more: it was fixed before the results were "
        "seen, and no penalty in any reported analysis was re-selected or "
        "replaced from it. One-hot is best at its own fixed value at every "
        "budget, so that value did not handicap it; ESM-2 is slightly better "
        "at $\\alpha = 5$ than at its fixed 50 at 200 labels, so the fixed "
        "value understates its advantage where that advantage is largest. The "
        "last column counts the libraries in which ESM-2 leads at the fixed "
        "penalties: all %d of %d at the two budgets where the article makes "
        "the claim, falling to a tie as the budget grows, which is the "
        "decay the article reports. At the largest ESM-2 penalty examined, "
        "performance declined and the direction of the comparison could "
        "reverse; at the largest one-hot penalty examined the decline is "
        "larger still. Absolute penalties are not comparable between the two "
        "representations, whose feature scales differ, so each row is read "
        "within a representation rather than across them.}"
        % (n_ag, seeds, n_small, n_ag * len(small)),
        "\\label{tab:alpha}",
        "\\centering",
        # 캡션 마지막 줄이 짧게 끝나면 표의 \toprule 이 그 줄과 같은
        # 높이에 그려져 밑줄처럼 보인다. 한 줄만큼 떼어 놓는다.
        "\\vspace{4pt}",
        "\\begin{tabular}{@{}l" + "r" * (len(ALPHA_ONEHOT) + len(ALPHA_ESM))
        + "r@{}}",
        "\\toprule",
        "& " + head,
        "Labels & " + cols + " & ESM-2 ahead \\\\",
        "\\midrule"]

    for m in sizes:
        cells = ["%.3f" % G[(m, "onehot", a)] for a in ALPHA_ONEHOT]
        cells += ["%.3f" % G[(m, "esm", a)] for a in ALPHA_ESM]
        lines.append("%s & %s & %d/%d \\\\"
                     % (f"{m:,}", " & ".join(cells), ahead[int(m)], n_ag))

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), n_small, n_ag * len(small)


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    for p in (SRC, SUM):
        if not os.path.exists(p):
            sys.exit(f"{p} 가 없습니다. exp83을 먼저 돌리십시오.")
    tex, n_small, d_small = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (벌점 민감도 표 · 200·500 라벨에서 "
          f"{n_small}/{d_small} 칸 방향 유지)")


if __name__ == "__main__":
    main()
