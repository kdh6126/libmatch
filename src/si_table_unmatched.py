"""보충표 S7 — 정합 전 팔을 정합 후와 같은 규약으로 잰 결과.

본문 표 1의 첫 열("정합 전")이 근거로 삼는 자료다. 이 표가 없으면 "정합 전에
one-hot이 앞섰다"가 몇 번 돌린 결과인지 독자가 알 수 없고, 실제로 처음에는
3~5회였다. exp70이 20회로 다시 잰 값을 여기에 싣는다.

정합 후(표 S2)와 나란히 읽히도록 같은 열 구성을 쓴다.

    python src/si_table_unmatched.py
"""
import io
import os
import sys

import pandas as pd

SI = "manuscript/supplementary_information.tex"
SRC = "results/exp70_per_size_tests.csv"
POOLED = "results/exp70_unmatched_seeds.csv"
MARK_A = "% ---- BEGIN unmatched seeds (generated)"
MARK_B = "% ---- END unmatched seeds"


def table_tex():
    d = pd.read_csv(SRC)
    r = pd.read_csv(POOLED)
    pooled = r.onehot_minus_esm.mean()
    wins = int((r.onehot_minus_esm > 0).sum())
    lines = [
        MARK_A,
        "\\begin{table}[h]",
        "\\caption{The unmatched arm of the representation comparison, "
        "measured with the protocol used for the matched arm: twenty "
        "independent train/test splits, the same training sizes, the same "
        "models and the same seeds. Positive values favour one-hot. The "
        "advantage is present at every size and widens with training data, so "
        "the effect that matching removes is not an artefact of the three to "
        "five splits used in the first pass.}",
        "\\centering",
        "\\begin{tabular}{@{}rrrrrr@{}}",
        "\\toprule",
        "Training labels & Splits & Mean & 95\\% CI & one-hot wins & "
        "$p$ (paired $t$) \\\\",
        "\\midrule"]
    for _, x in d.iterrows():
        lines.append(
            f"{int(x.n_train):,} & {int(x.n_seeds)} & {x.mean_diff:+.4f} & "
            f"{x.ci_lo:+.4f} to {x.ci_hi:+.4f} & "
            f"{int(x.onehot_wins)}/{int(x.n_seeds)} & "
            f"{x.p_ttest:.1e} \\\\")
    lines += [
        "\\midrule",
        f"Pooled & {len(r)} cells & {pooled:+.4f} & & {wins}/{len(r)} & \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        MARK_B]
    return "\n".join(lines), pooled, wins, len(r)


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp70을 먼저 돌리십시오.")
    tex, pooled, wins, n = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (표 S7, 통합 {pooled:+.4f}, one-hot {wins}/{n})")


if __name__ == "__main__":
    main()
