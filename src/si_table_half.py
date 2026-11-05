"""보충표 S16 — 상위 1%의 절반을 찾는 데 드는 측정 횟수(exp80).

이 분야가 보고하는 양은 무작위 대비 배수인데, 그 양에는 천장이 있다. 정합
격자에서는 예산 1,000에서 일곱 중 다섯이 상위 1%의 95% 이상을 회수하므로
배수가 값이 아니라 하한이 된다. 천장이 없는 양으로 같은 이야기를 한다.

    python src/si_table_half.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp80_budget_to_half.csv"
MARK_A = "% ---- BEGIN budget half (generated)"
MARK_B = "% ---- END budget half"


def table_tex():
    d = pd.read_csv(SRC)
    act = d[d.strategy == "active"].copy()
    rnd = d[d.strategy == "random"]
    mt = act[act.domain == "matched lattice"].dropna(subset=["median"])
    me = act[act.domain == "measured"].dropna(subset=["median"])
    n_never_rand = int(rnd["median"].isna().sum())

    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{Measurements needed for guided search to recover half of "
        "the top 1\\%. The ratio over random sampling, which method papers "
        "report, has a ceiling: once guided search has found nearly all of the "
        "top fraction its numerator cannot rise, and on the matched lattice "
        "that happens within the budgets a measured campaign would use. This "
        "quantity does not saturate. It is read off each seed's recovery "
        "curve over the budgets tested (200, 300, 500 and 1{,}000) by linear "
        "interpolation in log budget, and seeds that never reach half within "
        "that range are counted rather than extrapolated. Guided search needs "
        f"{mt['median'].min():.0f} to {mt['median'].max():.0f} measurements on "
        f"the matched libraries against {me['median'].min():.0f} and "
        f"{me['median'].max():.0f} on two of the measured targets, with the "
        "other two not reaching half within 1{,}000. Random sampling reaches "
        f"half in none of the {n_never_rand} arms.}}",
        "\\centering",
        "\\begin{tabular}{@{}llrrr@{}}",
        "\\toprule",
        "Library & Strategy & Measurements & Interquartile & Seeds reaching "
        "half \\\\",
        "\\midrule"]
    for _, x in d.sort_values(["strategy", "domain", "median"]).iterrows():
        name = x.arm.split(":", 1)[1].replace("_", "\\_")
        kind = "matched" if x.domain == "matched lattice" else "measured"
        val = ("$>$1{,}000" if pd.isna(x["median"])
               else f"{x['median']:,.0f}")
        iqr = ("---" if pd.isna(x.lo) else f"{x.lo:,.0f}--{x.hi:,.0f}")
        lines.append(f"{name} ({kind}) & {x.strategy} & {val} & {iqr} & "
                     f"{int(x.reached)}/{int(x.seeds)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), mt, me


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp80을 먼저 돌리십시오.")
    tex, mt, me = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S16, 정합 {mt['median'].min():.0f}~"
          f"{mt['median'].max():.0f} · 실측 {me['median'].min():.0f}~"
          f"{me['median'].max():.0f})")


if __name__ == "__main__":
    main()
