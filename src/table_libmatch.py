"""본문 표 — libmatch 가 네 쌍에 대해 내는 판정.

도구가 무엇을 하는지는 출력으로 보이는 편이 설명보다 짧다. 네 쌍을 고른 기준은
이 논문에서 멀어지는 순서다. 첫 쌍은 같은 항원에 대한 두 라이브러리로 이
논문이 해부한 비교이고, 둘째는 실측 대 합성, 셋째와 넷째는 항체와 무관한 공개
심층변이주사다. 넷째까지 같은 도구가 돌고 같은 축이 어긋난다는 것이 요점이다.

    python src/table_libmatch.py
"""
import io
import os
import sys

import pandas as pd

sys.path.insert(0, "libmatch")
import libmatch as LM                                           # noqa: E402

MS = "manuscript/manuscript.tex"
MARK_A = "% ---- BEGIN libmatch table (generated)"
MARK_B = "% ---- END libmatch table"
D = "libmatch/demo"

PAIRS = [
    ("Matched vs released lattice", "same antigen, two recipes",
     ("matched_2DD8_S.csv", None, None), ("released_2DD8_S.csv", None, None)),
    ("Measured vs matched lattice", "yeast display against simulation",
     ("alphaseq_measured.csv", "sequence", "label"),
     ("matched_2DD8_S.csv", None, None)),
    ("ProteinGym RASH vs matched", "non-antibody deep mutational scan",
     ("proteingym_RASH_HUMAN_Bandaru_2017.csv", None, None),
     ("matched_2DD8_S.csv", None, None)),
    ("ProteinGym TPK1 vs matched", "non-antibody deep mutational scan",
     ("proteingym_TPK1_HUMAN_Weile_2017.csv", None, None),
     ("matched_2DD8_S.csv", None, None)),
]
AXES = ["size", "structure", "dimension", "noise", "budget"]


def wt_for(fn):
    p = os.path.join(D, fn.replace(".csv", ".wt.txt"))
    return LM._read_wt(p) if os.path.exists(p) else None


def run(a, b):
    da = LM.read_table(os.path.join(D, a[0]), a[1], a[2], wt_for(a[0]))
    db = LM.read_table(os.path.join(D, b[0]), b[1], b[2], wt_for(b[0]))
    A = LM.describe(da, a[0])
    B = LM.describe(db, b[0])
    return LM.compare(A, B), A, B


def cell(row):
    """어긋났으면 크기를, 맞았으면 체크를. 표가 숫자로 말하게 한다."""
    if row.status == LM.MATCH:
        return "\\checkmark"
    note = str(row.note)
    if "x apart" in note:
        return "$" + note.split("x apart")[0].strip() + "\\times$"
    if "difference" in note:
        return note.replace("difference ", "$\\Delta$ ")
    if "cannot be aligned" in note:
        return "n/a"
    if "no replicates" in note:
        return "n/a"
    if "only one side" in note:
        return "differs"
    if "little overlap" in note:
        return "narrow"
    return "mismatch"


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    rows, n_mis = [], []
    for name, sub, a, b in PAIRS:
        cmp_df, A, B = run(a, b)
        d = {r.axis: cell(r) for _, r in cmp_df.iterrows()}
        k = int((cmp_df.status == LM.MISMATCH).sum())
        n_mis.append(k)
        rows.append((name, sub, d, k))
        print(f"  {name:<30} {k}/5 어긋남  "
              + " ".join(f"{ax}={d[ax]}" for ax in AXES))

    L = [MARK_A, "\\begin{table}[t]",
         "\\caption{What the tool reports for four dataset pairs, ordered by "
         "distance from the data it was developed on. A tick marks an axis the "
         "two sides share; a mismatch is given as the factor separating them, "
         "or as the reason the axis cannot be compared. The first row is the "
         "comparison this paper dissects. The last two are deep mutational "
         "scans of non-antibody proteins distributed as mutation notation, "
         "which the tool expands against the wild-type sequence; nothing in "
         "the five properties is specific to antibodies. Across all four the "
         "dimension axis is mismatched, by 5.9 to 16.1-fold, while structure "
         "agrees wherever both sides are neighbourhoods. The noise axis is "
         "reported as unavailable whenever a released file carries one score "
         "per sequence, which is the usual case.}",
         "\\centering", "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
         "Comparison & & Size & Structure & Dimension & Noise & Budget \\\\",
         "\\midrule"]
    for name, sub, d, k in rows:
        L.append(f"{name} & & {d['size']} & {d['structure']} & "
                 f"{d['dimension']} & {d['noise']} & {d['budget']} \\\\")
        L.append(f"\\quad\\textit{{{sub}}} & & & & & & \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\label{tab:libmatch}",
          "\\end{table}", MARK_B]
    tex = "\n".join(L)

    s = io.open(MS, encoding="utf-8").read()
    if MARK_A in s:
        i, j = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:i] + tex + s[j:]
    else:
        anchor = "\\subsection*{Four differences appear when conditions are "\
                 "not matched}"
        if anchor not in s:
            sys.exit("삽입 위치를 찾지 못했습니다.")
        s = s.replace(anchor, tex + "\n\n" + anchor)
    io.open(MS, "w", encoding="utf-8", newline="\n").write(s)
    print(f"\nwrote {MS}  (표 추가 · 어긋난 축 {min(n_mis)}~{max(n_mis)}/5)")


if __name__ == "__main__":
    main()
