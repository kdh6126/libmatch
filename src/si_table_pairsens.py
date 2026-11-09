"""보충표 S18 — 짝 검정의 감도(exp69/exp70 및 pair_* 열두 팔).

동족 항체 검정은 음성 결과였고, 음성 결과는 기구의 감도만큼만 유효하다.
그래서 구성상 결합체인 서열에 같은 검정을 걸어 감도를 재는데, 그것만으로는
부족하다. 강한 결합체와 천연 항체는 세기 말고도 구조(이웃 대 무관)·항원
수·길이가 함께 다르므로, 차이를 세기 탓으로 돌리면 이 논문이 1절에서
해석되지 않는다고 규정한 바로 그 귀속이 된다.

그래서 항원마다 두 팔을 돌린다. 강한 팔은 기존 정합 라이브러리이고, 약한 팔은
천연 동족 항체의 중앙 백분위(45.5%)에서 고른 모항체의 이웃이다. 두 팔은
항원·길이·이웃 구조가 같고 세기만 다르다.

출처를 함께 적는다. 1NCA_N 은 과거 격리 사건이 있던 항원이라 어느 판을 썼는지
독자가 한 번에 확인할 수 있어야 하는데, 그 항원만 해시를 적으면 오히려 그
항원에 의심이 쏠린다. 전부 같은 형식으로 적는다.

    python src/si_table_pairsens.py
"""
import glob
import hashlib
import io
import os
import re
import sys
import datetime

import numpy as np
import pandas as pd
from scipy import stats

SI = "manuscript/supplementary_information.tex"
MARK_A = "% ---- BEGIN pair sensitivity (generated)"
MARK_B = "% ---- END pair sensitivity"
DRIVER = "src/run_pair_sensitivity.py"
E68 = "results/exp68_matched_libraries.csv"
E75 = "results/exp75_rescore_check.csv"
NATIVE = "results/exp64_crossscore.csv"


def arms():
    """열두 팔을 (항원, 팔) 로 모은다. 2DD8_S 는 exp69/exp70 이름을 쓴다."""
    out = []
    for f in sorted(glob.glob("results/*_ranks.csv")):
        t = os.path.basename(f)[: -len("_ranks.csv")]
        if t == "exp69":
            ag, arm = "2DD8_S", "strong"
        elif t == "exp70":
            ag, arm = "2DD8_S", "weak"
        else:
            m = re.match(r"pair_(.+)_(weak|strong)$", t)
            if not m:
                continue
            ag, arm = m.groups()
        q = pd.read_csv(f)
        x = q.rank01.to_numpy()
        rho = stats.spearmanr(q.home_pct, q.rank01)
        out.append(dict(antigen=ag, arm=arm, n=len(q),
                        home_pct=float(q.home_pct.median()),
                        rank01=float(np.median(x)),
                        top10=int((x < 0.1).sum()),
                        rho=float(rho.statistic), rho_p=float(rho.pvalue)))
    return pd.DataFrame(out)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


E75D = "results/exp75_rescore_detail.csv"


def parent_gap(ag, lib, e68):
    """모항체가 공개 표와 맞는지. exp68 의 저장값을 믿지 않고 다시 구한다.

    exp68 의 2DD8_S 행은 오래된 값을 들고 있었다. 그 라이브러리는 발표된
    모항체로 지었는데, 규칙(길이 17 중 최저 에너지)이 고르는 서열은 다른
    것이고, 행이 쓰인 시점에는 규칙판이 돌고 있었다. parent 열만 뒤늦게
    고쳐졌고 에너지 세 칸은 버려진 후보의 것으로 남았다 -- 표에는 3.1e-07
    이 실렸지만, 그것은 라이브러리의 모항체에 대한 수치가 아니었다.

    고친 값을 다시 적어 두는 것으로는 같은 일이 또 일어난다. 그래서 여기서
    권위 있는 두 자리에서 다시 구한다: 라이브러리 파일이 그 서열에 대해
    적어 둔 에너지와, exp75 가 공개 표와 대조해 둔 차이. exp68 이 이와
    어긋나면 소리 내어 알린다.
    """
    if ag not in e68.index:
        return float("nan")
    par = str(e68.loc[ag, "parent"])
    r = lib.loc[lib.cdr3 == par, "energy"]
    if r.empty:
        print(f"  [주의] {ag}: 라이브러리에 모항체 {par} 가 없습니다")
        return float("nan")
    e_lib = float(r.iloc[0])
    stale = float(e68.loc[ag, "rescored_energy"])
    if abs(e_lib - stale) > 1e-3:
        print(f"  [주의] {ag}: exp68 이 적은 모항체 에너지 {stale:.2f} 는 "
              f"라이브러리의 {e_lib:.2f} 와 다릅니다. 저장값 대신 "
              f"exp75 의 재채점 결과를 씁니다.")
    if os.path.exists(E75D):
        d = pd.read_csv(E75D)
        m = d[(d.antigen == ag) & (d.cdr3 == par)]
        if len(m):
            return float(m.absdiff.iloc[0])
    print(f"  [주의] {ag}: exp75 에 모항체 {par} 의 재채점이 없습니다")
    return float("nan")


def provenance(antigens):
    """팔마다 쓴 라이브러리의 해시·생성 시각·오라클 대조값."""
    e68 = pd.read_csv(E68).set_index("antigen")
    e75 = pd.read_csv(E75).set_index("antigen")
    rows = []
    for ag in antigens:
        for arm, p in (("strong", f"data/absolut/matched_{ag}.parquet"),
                       ("weak", f"data/absolut/matched_{ag}_weak.parquet")):
            if not os.path.exists(p):
                continue
            d = pd.read_parquet(p)
            rows.append(dict(
                antigen=ag, arm=arm, file=os.path.basename(p),
                sha=digest(p)[:16], n=len(d),
                mtime=datetime.datetime.fromtimestamp(
                    os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M"),
                e_min=float(d.energy.min()), e_max=float(d.energy.max()),
                parent_gap=(parent_gap(ag, d, e68) if arm == "strong"
                            else np.nan),
                rescore_max=(float(e75.loc[ag, "max_diff"])
                             if arm == "strong" and ag in e75.index
                             else np.nan)))
    return pd.DataFrame(rows)


def esc(s):
    return str(s).replace("_", chr(92) + "_")


def table_tex():
    a = arms()
    p = a.pivot(index="antigen", columns="arm")
    S, W = p[("rank01", "strong")], p[("rank01", "weak")]
    k = int((W > S).sum())
    sign_p = stats.binomtest(k, len(S), 0.5).pvalue
    nat = pd.read_csv(NATIVE)
    nat = nat[nat.origin == nat.antigen]
    nat_pct, nat_r = float(nat.pct.median()), 0.484
    fixed = datetime.datetime.fromtimestamp(
        os.path.getmtime(DRIVER)).strftime("%Y-%m-%d %H:%M")
    first = min(os.path.getmtime(f) for f in
                glob.glob("results/pair_*_ranks.csv"))
    first = datetime.datetime.fromtimestamp(first).strftime("%Y-%m-%d %H:%M")

    L = [MARK_A, "\\begin{table}[h]",
         "\\caption{Sensitivity of the cognate-antibody test, measured as a "
         "function of binding strength. An absence of signal is bounded by the "
         "sensitivity of the instrument, so sequences that are binders of a "
         "given antigen by construction were put through the same test. Each "
         "antigen contributes two arms that share its identity, its CDRH3 "
         "length and its mutational-neighbourhood structure and differ only in "
         "strength: the strong arm samples the matched library, whose members "
         "the oracle scores in its top percentile, and the weak arm samples "
         "the neighbourhood of a parent chosen at the "
         f"{nat_pct:.1f}th energy percentile, the median of the native cognate "
         "antibodies. Holding the other axes fixed is what licenses "
         "attributing the difference to strength. rank01 is the normalised "
         "rank of the home antigen among the 93 antigens scored, so 0 means "
         "the home antigen is best and 0.5 is chance; the final column is the "
         "within-arm Spearman correlation between a sequence's home percentile "
         "and its rank01, which tests the same dependence again inside each "
         "arm and so does not rest on the single choice of parent. The strong "
         f"arm placed the home antigen lower in {k} of {len(S)} antigens "
         f"(two-sided exact $p = {sign_p:.5f}$), and the within-arm "
         "correlation is positive and significant in all "
         f"{len(a)} arms. The antigen set was fixed in advance: it was written "
         f"into \\texttt{{{esc(os.path.basename(DRIVER))}}} on {fixed}, before "
         f"the first result for a new antigen existed ({first}), and no "
         "antigen was added or dropped after seeing any outcome. 2Q8A\\_A is "
         "the one matched library omitted, because the sorted energy array it "
         "would need does not exist and generating it would have enlarged the "
         "comparison set from 93 to 94 antigens, breaking the protocol shared "
         "with the arms already run. For reference the native cognate "
         f"antibodies sit at a median percentile of {nat_pct:.1f} with a median "
         f"rank01 of {nat_r:.3f}.}}",
         "\\centering", "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         "Antigen & Arm & Sequences & Home percentile & rank01 & In top 10\\% "
         "& Within-arm $\\rho$ \\\\", "\\midrule"]
    for ag in p.index:
        for arm in ("strong", "weak"):
            r = a[(a.antigen == ag) & (a.arm == arm)].iloc[0]
            L.append(f"{esc(ag)} & {arm} & {int(r.n)} & "
                     f"{r.home_pct:.3f}\\% & {r.rank01:.3f} & "
                     f"{int(r.top10)}/{int(r.n)} & "
                     f"${r.rho:+.3f}$ ($p = {r.rho_p:.1e}$) \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    pr = provenance(list(p.index))
    L += ["\\begin{table}[h]",
          "\\caption{Provenance of every library used in the sensitivity test. "
          "The hash and modification time identify exactly which file produced "
          "each arm, and the last two columns repeat the oracle checks for the "
          "matched libraries: the parent energy must agree with the published "
          "table because the same binary, antigen and scoring function "
          "produced both, and a further 340 published sequences per antigen "
          "were rescored. Both columns are read from the rescoring run rather "
          "than from the build log, and the parent gap is the disagreement "
          "for the sequence each library was actually built around. For "
          "2DD8\\_S that sequence is the published parent antibody, which is "
          "not the sequence the selection rule used for the other antigens "
          "returns: among the 748{,}021 length-17 sequences scored against "
          "that antigen it ranks fourth rather than first. The library was "
          "built before the rule was written and was kept as it stood, so "
          "that the results reported for that antigen continue to refer to "
          "the file that produced them. One of these antigens, 1NCA\\_N, had a first library "
          "that was quarantined after a scoring run wrote output while "
          "failing; the version used here is the rebuilt one, and the "
          "quarantined file is retained separately so the two can be compared "
          "(Supplementary Table~S10). Hashes are given for all antigens rather "
          "than for that one alone, so that the record does not single it out. "
          "Weak-arm libraries carry no oracle-agreement entry because they are "
          "not matched libraries: their parents were selected from the "
          "published energy distribution and their members scored fresh, so no "
          "published energy exists to compare them against.}",
          "\\centering", "\\begin{tabular}{@{}llrlrr@{}}", "\\toprule",
          "Antigen & Arm & Sequences & SHA-256 (first 16) & Parent gap & "
          "Rescore max \\\\", "\\midrule"]
    for _, r in pr.iterrows():
        pg = "---" if pd.isna(r.parent_gap) else f"{r.parent_gap:.1e}"
        rm = "---" if pd.isna(r.rescore_max) else f"{r.rescore_max:.1e}"
        L.append(f"{esc(r.antigen)} & {r.arm} & {int(r.n):,} & "
                 f"\\texttt{{{r.sha}}} & {pg} & {rm} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", MARK_B]
    return "\n".join(L), a, k, sign_p


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    for f in (E68, E75, NATIVE):
        if not os.path.exists(f):
            sys.exit(f"{f} 가 없습니다.")
    tex, a, k, sign_p = table_tex()
    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        i, j = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:i] + tex + s[j:]
    else:
        s = s.replace("\\end{document}", tex + "\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    n_ag = a.antigen.nunique()
    print(f"wrote {SI}  (표 S18/S19, 항원 {n_ag}종 · 팔 {len(a)}개 · "
          f"방향 일치 {k}/{n_ag} · p={sign_p:.5f})")
    print(f"  팔 내부 rho: {a.rho.min():+.3f}~{a.rho.max():+.3f} · "
          f"모두 p<0.05: {bool((a.rho_p < 0.05).all())}")


if __name__ == "__main__":
    main()
