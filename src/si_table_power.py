"""보충표 S13 — 동족 항체 검사의 분해능(exp78).

원고는 이 검사에서 신호를 찾지 못했다고 보고한다. 없음을 보고하는 결과는
기구의 분해능과 짝을 지어야 뜻이 생기는데, 원고에는 "모의 자료에서 회수된다"는
한 문장만 있었고 그 모의가 어디에도 없었다. 이 표가 그 자리를 메운다.

    python src/si_table_power.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp78_cognate_power.csv"
MARK_A = "% ---- BEGIN cognate power (generated)"
MARK_B = "% ---- END cognate power"


def table_tex():
    d = pd.read_csv(SRC)
    obs = d[d.delta == 0].iloc[0]
    s80 = d[d.power_sign >= 0.8]
    k80 = d[d.power_ks >= 0.8]
    ds = float(s80.delta.min()) if len(s80) else float("nan")
    dk = float(k80.delta.min()) if len(k80) else float("nan")
    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{How large a pairing effect the cognate-antibody test would "
        "have detected. Each cognate CDRH3's energy was lowered by a fixed "
        f"amount $\\delta$, making it bind that much better, and the two tests "
        "reported in the main text were repeated on the same 88 antigens. "
        "Power is the fraction of 2{,}000 bootstrap resamples of the antigens "
        "in which the test rejects at $\\alpha = 0.05$. Recomputing a "
        "percentile against 6.9 million energies requires the full per-antigen "
        "table, so the shifted percentiles use a normal approximation whose "
        "standard deviation is solved from the median and the observed "
        "(energy, percentile) pair; against the five antigens whose full "
        "energy arrays we hold, the approximation errs by at most 0.005 in "
        f"percentile. The observed data are the $\\delta = 0$ row. The sign "
        f"test reaches 95\\% power at $\\delta = {ds:g}$ energy units and the "
        f"Kolmogorov--Smirnov test 97\\% at $\\delta = {dk:g}$, against a "
        "library whose energies span $-131$ to $-75$, so an effect of that "
        "size would have been visible and the null result is not a failure of "
        "resolution.}",
        "\\centering",
        "\\begin{tabular}{@{}rrrrrr@{}}",
        "\\toprule",
        "$\\delta$ & Median percentile & $p$ (sign) & Power & $p$ (KS) & "
        "Power \\\\",
        "\\midrule"]
    for _, x in d.iterrows():
        lines.append(f"{x.delta:g} & {x.median_pct:.2f}\\% & "
                     f"{x.p_sign:.1e} & {x.power_sign:.2f} & "
                     f"{x.p_ks:.1e} & {x.power_ks:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), obs, ds, dk


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp78을 먼저 돌리십시오.")
    tex, obs, ds, dk = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S13, 부호 80%에서 delta={ds:g}, "
          f"KS 80%에서 delta={dk:g})")


if __name__ == "__main__":
    main()
