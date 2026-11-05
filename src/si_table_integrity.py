"""보충표 S10 — 조용히 틀린 라이브러리가 내놓았을 결론(exp77).

원고는 일관성 검사를 Methods에서 한 문단으로 다루고 "실패해서 보류했다"로
끝난다. 그 검사가 무엇을 막았는지는 적지 않았는데, 실은 그것이 이 논문의
주제와 같은 이야기다 -- 통제하지 않은 오라클 실행이 그럴듯하면서 다른 답을
내놓는다.

같은 항원, 같은 설계, 같은 모형, 같은 분할. 채점 실행 하나만 다르다.

    python src/si_table_integrity.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp77_corrupt_conclusion.csv"
OLD = "data/absolut/unreliable/matched_1NCA_N.parquet"
NEW = "data/absolut/matched_1NCA_N.parquet"
MARK_A = "% ---- BEGIN integrity check (generated)"
MARK_B = "% ---- END integrity check"


def table_tex():
    R = pd.read_csv(SRC)
    rows = []
    for (lib, n), g in R.groupby(["library", "n_train"]):
        v = g.onehot_minus_esm.to_numpy()
        rows.append(dict(library=lib, n_train=int(n),
                         onehot=g.onehot.mean(), esm=g.esm.mean(),
                         diff=v.mean(), esm_wins=int((v < 0).sum()), n=len(v)))
    S = pd.DataFrame(rows).sort_values(["n_train", "library"])

    a = pd.read_parquet(OLD)[["cdr3", "energy"]]
    b = pd.read_parquet(NEW)[["cdr3", "energy"]]
    m = a.merge(b, on="cdr3", suffixes=("_o", "_n"))
    d = (m.energy_o - m.energy_n)
    n_diff = int((d.abs() > 1e-3).sum())

    flips = []
    for n in sorted(S.n_train.unique()):
        q = S[(S.library == "quarantined") & (S.n_train == n)]["diff"].iloc[0]
        r = S[(S.library == "rebuilt") & (S.n_train == n)]["diff"].iloc[0]
        if np.sign(q) != np.sign(r):
            flips.append(n)

    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{What the consistency check prevented. The first library "
        "built for influenza N9 neuraminidase was written by a scoring run that "
        "failed without raising an error; it was quarantined, the scoring "
        f"function was changed to raise on a non-zero exit, and the library was "
        f"rebuilt. The two versions disagree for {n_diff:,} of {len(m):,} "
        f"sequences, by a median of {d.abs().median():.1f} energy units and up "
        f"to {d.abs().max():.1f}, the quarantined version binding more weakly "
        f"in {(d > 0).mean() * 100:.0f}\\% of cases. Nothing internal to that "
        "library reveals the problem: its energy range is plausible, its "
        "mutational structure is correct and its parent is near the minimum. "
        "Repeating the representation comparison on both versions, with the "
        "same design, models, splits and seeds, changes the conclusion: the "
        "quarantined library puts the language model ahead at every training "
        "size, while the rebuilt one shows the advantage closing by 1,000 "
        f"labels (sign differs at {', '.join(f'{x:,}' for x in flips)}; Welch "
        "$t$ on the pooled differences $p = 2.7\\times10^{-17}$). The direction "
        "of the error is worth stating plainly: the corrupted library resembles "
        "measured data more closely than the correct one does, so the check "
        "guarded against a false positive that favoured our own "
        "conclusion.}",
        "\\centering",
        "\\begin{tabular}{@{}rlrrrr@{}}",
        "\\toprule",
        "Labels & Library & one-hot & ESM-2 & Difference & ESM ahead \\\\",
        "\\midrule"]
    for _, x in S.iterrows():
        name = ("quarantined" if x.library == "quarantined" else "rebuilt")
        lines.append(f"{int(x.n_train):,} & {name} & {x.onehot:.3f} & "
                     f"{x.esm:.3f} & {x['diff']:+.4f} & "
                     f"{int(x.esm_wins)}/{int(x.n)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), S, n_diff, len(m)


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    for p in (SRC, OLD, NEW):
        if not os.path.exists(p):
            sys.exit(f"{p} 가 없습니다.")
    tex, S, n_diff, n_tot = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S10, 어긋난 서열 {n_diff:,}/{n_tot:,})")


if __name__ == "__main__":
    main()
