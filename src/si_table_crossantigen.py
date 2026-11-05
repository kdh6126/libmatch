"""보충표 S15 — 오라클 에너지가 항원 사이에서 얼마나 닮았는가(exp81).

본문은 격자 오라클이 어느 항체가 어느 항원에 붙는지를 담고 있지 않다고
보고하고, 그 근거로 동족 항체의 백분위가 균일하다는 것과 접촉 부위가 항원의
모양으로 정해진다는 것을 든다. 둘 다 간접 증거다.

직접 증거는 이것이다 -- 같은 서열 691만 개를 여러 항원에 채점한 값이 서로
얼마나 닮았는가. 닮아 있다면 오라클이 재는 것의 큰 부분은 '이 항원에 붙는가'가
아니라 '일반적으로 잘 붙는가'다.

    python src/si_table_crossantigen.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp81_cross_antigen.csv"
SRC2 = "results/exp81_top_overlap.csv"
MARK_A = "% ---- BEGIN cross antigen (generated)"
MARK_B = "% ---- END cross antigen"


def table_tex():
    C = pd.read_csv(SRC)
    O = pd.read_csv(SRC2) if os.path.exists(SRC2) else None
    c17 = C.pearson_len17.dropna() if "pearson_len17" in C.columns else []
    same = C[(C.a == "1ZTX_E") & (C.b == "2Q8A_A")]

    fold = ""
    if O is not None and len(O):
        g = O[O.top == O.top.max()]
        med = float(g.overlap.median())
        exp = float(g.expected.iloc[0])
        if exp > 0 and med > 0:
            fold = (f" Their top {int(O.top.max()):,} binders overlap by a "
                    f"median of {med:.0f} sequences where independence "
                    f"predicts {exp:.3f}, a factor of {med / exp:,.0f}.")

    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{How much the oracle's energies for different antigens "
        "resemble one another. The same 6.9 million CDRH3 sequences are scored "
        "against each antigen, so the two columns of any pair are a paired "
        "measurement on identical sequences; we sample 500{,}000 of them. "
        "Lattice energy falls with sequence length, which would inflate the "
        "correlation on its own, so the last column repeats it within length "
        f"17 alone. Correlations remain between {C.pearson.min():.2f} and "
        f"{C.pearson.max():.2f} overall"
        + (f" and between {c17.min():.2f} and {c17.max():.2f} at fixed length"
           if len(c17) else "")
        + f".{fold} For one pair the single best length-17 binder of 748{{,}}021 "
        "is the same sequence. This is the direct form of the boundary the "
        "main text draws from the cognate-antibody test: a large part of what "
        "the oracle measures is whether a sequence binds well in general "
        "rather than whether it binds this antigen, which is also why no "
        "antibody stands out against its own.}",
        "\\centering",
        "\\begin{tabular}{@{}llrrr@{}}",
        "\\toprule",
        "Antigen A & Antigen B & Pearson & Spearman & Pearson, length 17 \\\\",
        "\\midrule"]
    for _, x in C.iterrows():
        v17 = ("---" if pd.isna(x.get("pearson_len17", np.nan))
               else f"{x.pearson_len17:+.3f}")
        lines.append(f"{x.a.replace('_', chr(92) + '_')} & "
                     f"{x.b.replace('_', chr(92) + '_')} & "
                     f"{x.pearson:+.3f} & {x.spearman:+.3f} & {v17} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), C


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp81을 먼저 돌리십시오.")
    tex, C = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S15, 쌍 {len(C)}개)")


if __name__ == "__main__":
    main()
