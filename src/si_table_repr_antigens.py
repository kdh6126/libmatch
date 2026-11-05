"""보충표 S8 — 표현 비교를 정합 항원 여러 종으로 넓힌 결과.

본문의 표현 비교는 정합 항원 하나(2DD8_S)에 기대고 있었다. 심사가 이 자리를
"항원 하나"라고 짚을 것이고 그 말이 맞다. exp71이 exp68이 만든 정합
라이브러리마다 같은 규약을 적용한 값을 이 표에 싣는다.

통합 평균만 싣지 않는다. 학습 크기별로 갈라 놓아야 우위가 라벨 예산의
함수라는 것이 보이고, 그 모양이 항원마다 같은지도 보인다.

    python src/si_table_repr_antigens.py
"""
import io
import os
import sys

import pandas as pd

SI = "manuscript/supplementary_information.tex"
PER_AG = "results/exp71_per_antigen.csv"
PER_SZ = "results/exp71_per_size.csv"
MARK_A = "% ---- BEGIN repr many antigens (generated)"
MARK_B = "% ---- END repr many antigens"


def table_tex():
    S = pd.read_csv(PER_AG).sort_values("antigen")
    P = pd.read_csv(PER_SZ).sort_values(["antigen", "n_train"])
    n_esm = int((S.ci_hi < 0).sum())
    n_oh = int((S.ci_lo > 0).sum())
    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{The representation comparison on every matched library that "
        "passed the consistency check, each measured with the protocol used for "
        "2DD8\\_S: twenty independent train/test splits, the same training "
        "sizes, the same two models and the same seeds. Only the antigen "
        "changes. Negative values favour the language model. The advantage is a "
        "function of the label budget rather than of the antigen: it is largest "
        "at 200 labels and decays monotonically, so the ordering of the two "
        f"representations is not a fixed property of the matched library. "
        f"Intervals exclude zero in favour of the language model for {n_esm} of "
        f"{len(S)} antigens and in favour of one-hot for {n_oh}.}}",
        "\\centering",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "Antigen & Labels & Mean & 95\\% CI & ESM ahead & $p$ (paired $t$) \\\\",
        "\\midrule"]
    for ag, g in P.groupby("antigen", sort=True):
        for i, (_, x) in enumerate(g.iterrows()):
            name = ag.replace("_", "\\_") if i == 0 else ""
            lines.append(
                f"{name} & {int(x.n_train):,} & {x.mean_diff:+.4f} & "
                f"{x.ci_lo:+.4f} to {x.ci_hi:+.4f} & "
                f"{int(x.esm_wins)}/{int(x.cells)} & {x.p_ttest:.1e} \\\\")
        s = S[S.antigen == ag].iloc[0]
        lines.append(
            f"\\textit{{pooled}} & {int(s.cells)} cells & {s.mean_diff:+.4f} & "
            f"{s.ci_lo:+.4f} to {s.ci_hi:+.4f} & "
            f"{int(s.esm_wins)}/{int(s.cells)} & {s.p_ttest:.1e} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(lines), S, P


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    for p in (PER_AG, PER_SZ):
        if not os.path.exists(p):
            sys.exit(f"{p} 가 없습니다. exp71을 먼저 돌리십시오.")
    tex, S, P = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S8, 항원 {len(S)}종 · 칸 {len(P)}개)")
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
