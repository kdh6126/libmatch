"""보충표 S9 — 탐색 우위를 시드 20개와 신뢰구간으로.

본문의 탐색 headline("정합하면 3.3배")은 시드 5개의 평균 비율이었고 구간이
없었다. 표현 비교에서 정합 전 팔이 3~5회였던 것과 같은 문제이고, 구간 없이
3.3 대 3.0을 "일치한다"고 읽을 수는 없다. exp72가 같은 설계로 시드만 늘려
다시 잰 값을 싣는다.

    python src/si_table_search_ci.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp72_ratios.csv"
MARK_A = "% ---- BEGIN search ci (generated)"
MARK_B = "% ---- END search ci"


def _fmt(x, nd=2, dash="---"):
    return dash if not np.isfinite(x) else f"{x:.{nd}f}"


def table_tex():
    d = pd.read_csv(SRC).sort_values(["budget", "arm"])
    lattice = d[d.arm.str.startswith("matched:")]
    real = d[d.arm.str.startswith("real:")]
    at1k = d[d.budget == 1000]
    lo = at1k[at1k.arm.str.startswith("real:")].ratio
    hi = at1k[at1k.arm.str.startswith("matched:")].ratio
    # 배수의 분자가 1에 가까우면 그 배수는 값이 아니라 하한이다. 몇 개가
    # 그런지 세어 캡션에 적는다 -- 숨기면 심사자가 표에서 직접 본다.
    mt = at1k[at1k.arm.str.startswith("matched:")]
    n_matched = len(mt)
    n_sat = int((mt.active >= 0.95).sum())
    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{The search advantage measured with twenty seeds and a "
        "bootstrap interval on the reported quantity. Library, budgets, "
        "strategies, the top-1\\% rule and the seed rule are those of the first "
        "pass; only the number of seeds and the reporting changed. The ratio is "
        "of mean recovery, so the interval resamples seeds and recomputes the "
        "ratio of means rather than averaging per-seed ratios, which are "
        "undefined whenever random search recovers nothing. At a budget of 200 "
        "the guided strategies spend the whole budget on their initial random "
        "sample and never fit a model, so a ratio of 1.00 there is a property "
        "of the design and not a result. A ratio whose numerator is close to 1 "
        "is a lower bound rather than a value, because the numerator cannot "
        f"rise much further: at a budget of 1{{,}}000, {n_sat} of the "
        f"{n_matched} matched libraries recover at least 95\\% of the top "
        "1\\%. "
        f"At a budget of 1{{,}}000 the measured targets span {_fmt(lo.min())} to {_fmt(lo.max())}-fold "
        f"and the matched lattice {_fmt(hi.min())} to {_fmt(hi.max())}; only the "
        f"lowest matched library overlaps that range. Matching library size and "
        "mutational structure therefore narrows this difference by three orders "
        "of magnitude without closing it.}",
        "\\centering",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "Library & Budget & Active & Random & Ratio & 95\\% CI \\\\",
        "\\midrule"]
    for block in (lattice, real):
        for _, x in block.iterrows():
            name = x.arm.split(":", 1)[1].replace("_", "\\_")
            kind = ("matched lattice" if x.arm.startswith("matched:")
                    else "measured")
            ci = (f"{_fmt(x.ci_lo)} to {_fmt(x.ci_hi)}"
                  if np.isfinite(x.ci_lo) else "---")
            lines.append(
                f"{name} ({kind}) & {int(x.budget):,} & {x.active:.3f} & "
                f"{x['random']:.3f} & {_fmt(x.ratio)} & {ci} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), d


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp72를 먼저 돌리십시오.")
    tex, d = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S9, {len(d)} 칸)")
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
