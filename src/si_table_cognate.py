"""Supplementary Table S3 — the denominator of the cognate-antibody test.

The main text says 88 of 103 antigens support the test. A reader cannot judge
that number without knowing what happened to the other 15, and whether the
exclusions are correlated with the result. Four of the categories below were
found by fixing gates that had been rejecting valid data, so the counts moved
during the work; the table reports the final disposition and the reason for
each class, and names the antigens wherever the class is small enough to list.

Written into manuscript/supplementary_information.tex, replacing any
previous copy of the table, so re-running is safe.

    python src/si_table_cognate.py
"""
import io
import os
import sys

import pandas as pd

SI = "manuscript/supplementary_information.tex"
MARK_A = "% ---- BEGIN cognate disposition (generated)"
MARK_B = "% ---- END cognate disposition"


def rows():
    d = pd.read_csv("results/exp62_native_fulltable.csv")
    q = pd.read_csv("results/exp62_quarantine.csv")
    pre = pd.read_csv("results/exp67_epitope_precheck.csv")
    ids = [x.strip() for x in io.open("data/absolut/antigen_ids.txt",
                                      encoding="utf-8") if x.strip()]
    bad = pre[pre.verdict == "전제 실패"]
    prem = set(bad.antigen.astype(str))

    ok = d[d.status == "ok"]
    short = d[d.status == "too_short"]
    scored = ok[~ok.antigen.astype(str).isin(prem)]
    reduced = scored[scored.coverage_pct.fillna(100) < 99]
    main = scored[scored.coverage_pct.fillna(100) >= 99]

    # 전제 실패는 원인이 둘이라 나눠 적는다. 하나로 묶으면 "서열이 없다"와
    # "항체가 다른 사슬에 붙는다"가 같은 칸에 들어가는데, 둘은 다른 문제다.
    noseq = [a for a in prem
             if "PDB에 없음" in str(bad[bad.antigen == a].detail.iloc[0])]
    nopair = sorted(prem - set(noseq))

    def names(xs, cap=4):
        # 항원 이름에는 밑줄이 들어 있다(2Q8A_A). LaTeX에서 그대로 두면 수식
        # 첨자로 읽혀 컴파일이 멈춘다.
        xs = sorted(str(x).replace("_", "\\_") for x in xs)
        return ", ".join(xs[:cap]) + (", \\ldots" if len(xs) > cap else "")

    return [
        ("Scored, full published table", len(main),
         "used in the main analysis", ""),
        ("Scored, published table covers 97.5--97.6\\% of the library",
         len(reduced), "reported separately with exact bounds",
         names(reduced.antigen)),
        ("Cognate CDRH3 shorter than the 11-residue scoring window",
         len(short), "the oracle returns no value", names(short.antigen)),
        ("Antibody in the source entry does not contact the antigen chain",
         len(nopair), "the pair is not a binder of this antigen",
         names(nopair)),
        ("Antigen sequence absent from the named PDB entry", len(noseq),
         "provenance cannot be established", names(noseq)),
        ("Two distinct antibodies bound to the same antigen",
         int((q.reason.astype(str).str.contains("둘 이상")).sum()),
         "no single cognate CDRH3", ""),
        # 사유 문구는 관문을 고치면서 바뀐다("중쇄 후보 없음" ->
        # "중쇄는 있으나 보존 Cys가 없음"). 문구 하나에 맞추면 조용히 0이 되고
        # 합계가 어긋나므로 둘 다 받는다.
        ("Conserved cysteine absent from the heavy chain",
         int((q.reason.astype(str)
              .str.contains("Cys|중쇄 후보 없음", regex=True)).sum()),
         "not expressible in the oracle's convention", ""),
        ("Published table archive corrupt at source", 1,
         "size and checksum match the server, but the archive is truncated",
         "4N9G\\_C"),
    ], len(ids)


def table_tex():
    rs, total = rows()
    n = sum(r[1] for r in rs)
    lines = [MARK_A,
             "\\begin{table}[h]",
             "\\caption{Disposition of the 103 Absolut! antigens in the "
             "cognate-antibody test. Exclusions are listed with their reason "
             "so that the denominator can be checked; none of the classes is "
             "defined by the outcome of the test.}",
             "\\centering",
             "\\begin{tabular}{@{}p{7.4cm}rp{4.4cm}@{}}",
             "\\toprule",
             "Disposition & $n$ & Reason \\\\",
             "\\midrule"]
    for name, k, why, who in rs:
        if not k:
            continue
        extra = f" ({who})" if who else ""
        lines.append(f"{name}{extra} & {k} & {why} \\\\")
    lines += ["\\midrule",
              f"Total & {n} & \\\\",
              "\\bottomrule",
              "\\end{tabular}",
              "\\end{table}",
              MARK_B]
    if n != total:
        print(f"  [경고] 합계 {n}이 항원 목록 {total}과 다릅니다. "
              "빠진 갈래가 있습니다.")
    return "\n".join(lines), n, total


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    tex, n, total = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S3, 합계 {n}/{total})")


if __name__ == "__main__":
    main()
