"""보충표 S20 — 일곱 모델의 순위가 격자에서 실측으로 옮겨지는가(exp82).

벤치마크가 방법을 충실히 줄 세우는지는 이 분야가 벤치마크에 기대하는 바로 그
성질이다. Absolut! 원 논문은 상관 0.9 를 보고했지만 근거는 고전 분류기로 수행한
이진 분류 과제 하나였고, 그 뒤 벤치마크가 쓰인 범위보다 좁다. 이 논문도 본문에서
두 표현만 비교했다면 같은 비판을 받는다.

exp16 이 실측 표적 넷에 돌린 일곱 모델을 정합 격자 일곱 항원에 같은 정의로
돌리고, 학습 크기마다 두 팔의 모델 순위를 맞춰 본다.

    python src/si_table_transfer.py
"""
import io
import os
import sys

import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp82_transfer_summary.csv"
LAT = "results/exp82_zoo_transfer.csv"
MEA = "results/exp16_model_zoo.csv"
MARK_A = "% ---- BEGIN zoo transfer (generated)"
MARK_B = "% ---- END zoo transfer"


def esc(s):
    return str(s).replace("_", chr(92) + "_")


def table_tex():
    d = pd.read_csv(SRC)
    L = pd.read_csv(LAT)
    M = pd.read_csv(MEA)
    n_ag = L.antigen.nunique()
    n_tg = M.target.nunique()
    n_md = len(set(L.model) & set(M.model))
    hi = d.loc[d.rho.idxmax()]
    lo = d.loc[d.rho.idxmin()]

    out = [MARK_A, "\\begin{table}[h]",
           "\\caption{Whether the benchmark ranks methods as the measured data "
           f"do. The same {n_md} models were fitted on the {n_ag} matched "
           f"lattice libraries and on the {n_tg} measured targets, from one "
           "definition shared by both runs so that no hyperparameter can "
           "differ between the arms. At each training size the models are "
           "ordered by mean held-out Spearman within each arm and the two "
           "orderings compared. Intervals resample antigens and targets "
           "separately, since those are the two sources of variation in the "
           f"means being ranked. Agreement is high at {int(hi.n_train)} "
           f"labels, where the orderings correlate at {hi.rho:+.3f} and both "
           "arms put the same model first, and falls monotonically to "
           f"{lo.rho:+.3f} at {int(lo.n_train):,}, where the best model on the "
           "lattice is no longer the best on measured data. The original "
           "benchmark reported a correlation of 0.9 for method accuracies, "
           "obtained for one binary classification task with classical "
           "classifiers; these numbers locate the budget at which that "
           "agreement holds rather than contradicting it. Sizes above "
           f"{int(d.n_train.max()):,} are not comparable because the matched "
           "libraries hold 6{,}000 sequences.}",
           "\\centering", "\\begin{tabular}{@{}rlllc@{}}", "\\toprule",
           "Training size & Rank correlation & Best on lattice & "
           "Best on measured & Same \\\\", "\\midrule"]
    for _, r in d.iterrows():
        out.append(f"{int(r.n_train):,} & {r.rho:+.3f} "
                   f"[{r.lo:+.3f}, {r.hi:+.3f}] & {esc(r.best_lattice)} & "
                   f"{esc(r.best_measured)} & "
                   f"{'yes' if r.same_best else 'no'} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(out), d, n_ag, n_tg, n_md


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    for f in (SRC, LAT, MEA):
        if not os.path.exists(f):
            sys.exit(f"{f} 가 없습니다.")
    tex, d, n_ag, n_tg, n_md = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        i, j = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:i] + tex + s[j:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S20 · 모델 {n_md} · 격자 {n_ag}항원 · "
          f"실측 {n_tg}표적)")
    print(f"  상관 {d.rho.max():+.3f} → {d.rho.min():+.3f} · "
          f"최고 모델 일치 {int(d.same_best.sum())}/{len(d)}")


if __name__ == "__main__":
    main()
