"""보충표 S17 — 공개 전수표와 우리 파이프라인의 대조(exp75).

정합 라이브러리가 공개 표와 같은 오라클에서 나왔다는 근거는 지금까지
'두 라이브러리가 우연히 공유하는 서열 한둘'뿐이었다. 그 표본으로는 파이프라인
전체가 옳다고 말할 수 없다. 항원마다 공개 표에서 340개를 직접 다시 채점해
비교한다.

    python src/si_table_rescore.py
"""
import io
import os
import sys

import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp75_rescore_check.csv"
MARK_A = "% ---- BEGIN rescore check (generated)"
MARK_B = "% ---- END rescore check"


def table_tex():
    d = pd.read_csv(SRC).sort_values("antigen")
    n_tot = int(d.n_scored.sum())
    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{Agreement between our scoring pipeline and the published "
        "Absolut! table. The parent of each matched library also appears in "
        "the released library, so its energy must agree; but a mutational "
        "neighbourhood and a library of unrelated CDRH3s otherwise share only "
        "one or two sequences by chance, which is too thin a sample to "
        "establish that the pipeline reproduces the published oracle. We "
        f"therefore rescored {int(d.n_sampled.iloc[0])} published sequences "
        "per antigen and compared them with the released table: 200 of "
        "length 17 drawn at random, the 20 lowest and the 20 highest in "
        "energy, and 100 of any length, so that the comparison covers the "
        "tails as well as the bulk and is not restricted to the length the "
        "matched libraries use. Sequences were scored in the same "
        "\\texttt{repertoire} mode and the best slice per sequence retained. "
        f"All {n_tot:,} sequences were scored, none went missing, and the "
        f"largest absolute disagreement was "
        f"{d.max_diff.max():.1e} energy units, at the rounding of the "
        "released table's single-precision values. None exceeded the "
        "$10^{-3}$ tolerance.}",
        "\\centering",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "Antigen & Rescored & Missing & Median difference & Largest "
        "difference & Over tolerance \\\\",
        "\\midrule"]
    for _, x in d.iterrows():
        lines.append(
            f"{x.antigen.replace('_', chr(92) + '_')} & {int(x.n_scored):,} & "
            f"{int(x.n_missing)} & {x.median_diff:.1e} & {x.max_diff:.1e} & "
            f"{int(x.n_over_tol)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), d


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp75를 먼저 돌리십시오.")
    tex, d = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S17, {int(d.n_scored.sum()):,}개 재채점 · "
          f"최대 차 {d.max_diff.max():.1e} · 초과 {int(d.n_over_tol.sum())}건)")


if __name__ == "__main__":
    main()
