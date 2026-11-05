"""원고가 말하는 수치를 결과표와 대조한다.

audit_claims.py 는 예전 실험(exp2, exp20, exp9b)을 본다. 정합 항원이 일곱으로
늘고 시드가 20개가 되면서 본문의 수치가 새 표에서 나오게 됐으므로, 그쪽을 보는
대조가 따로 필요하다.

손으로 옮겨 적은 수치는 반드시 한 번은 어긋난다. 실제로 이 검사가 잡은 것들:

  "우위가 2,000~4,000에서 0에 이른다"  -- 4,000에서 한 항원은 여전히 0을
                                       제외했고 다른 항원은 부호가 넘어가 있었다
  "세 정합 라이브러리"                  -- 실험은 이미 다섯이었다
  전체 상관 +0.93                       -- 군집 분리를 다시 말하는 것이었다

원고에서 숫자를 직접 읽어 비교한다. 읽기 전에 공백을 하나로 접으므로, 문장을
어디서 줄바꿈하든 패턴이 따라온다 -- 줄바꿈을 정규식에 넣으려다 소스에 제어문자를
집어넣은 적이 있어서, 아예 그럴 일이 없게 만든다.

    python src/audit_new_claims.py
    python src/audit_new_claims.py release/manuscript/manuscript.tex
"""
import argparse
import glob
import io
import os
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_MAIN = "manuscript/manuscript.tex"
NL = chr(10)
OK, BAD, NOTE, SKIP = "맞음", "어긋남", "참고", "건너뜀"
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9}


def flat(s):
    """공백을 하나로 접는다. 패턴이 줄바꿈을 몰라도 되게."""
    return re.sub(r"\s+", " ", s)


def num(x):
    return float(str(x).replace("{,}", "").replace(",", ""))


def find(pat, text, group=1):
    m = re.search(pat, text)
    return num(m.group(group)) if m else None


def word(pat, text, group=1):
    m = re.search(pat, text)
    return WORDS.get(m.group(group)) if m else None


class Audit:
    def __init__(self):
        self.bad = 0

    def check(self, label, claimed, actual, tol=0.02):
        if claimed is None or (isinstance(claimed, float)
                               and not np.isfinite(claimed)):
            print(f"  [{'못 찾음':6s}] {label}")
            self.bad += 1
            return False
        good = abs(claimed - actual) <= tol
        if not good:
            self.bad += 1
        print(f"  [{OK if good else BAD:6s}] {label}: 원고 {claimed:g} · "
              f"자료 {actual:g}")
        return good

    def fail(self, msg):
        self.bad += 1
        print(f"  [{BAD:6s}] {msg}")

    def note(self, msg):
        print(f"  [{NOTE:6s}] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manuscript", nargs="?", default=DEFAULT_MAIN)
    a = ap.parse_args()
    for x in (sys.stdout, sys.stderr):
        try:
            x.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.exists(a.manuscript):
        sys.exit(f"원고를 찾을 수 없습니다: {a.manuscript}")
    raw = io.open(a.manuscript, encoding="utf-8").read()
    t = flat(raw)
    A = Audit()
    skipped = []

    # ---------------------------------------------------------------- 탐색
    print("=== 탐색 우위 (exp72) ===")
    R = pd.read_csv("results/exp72_ratios.csv")
    bud = find(r"At (\d+) none of the seven does", t) or 500.0
    r = R[R.budget == bud]
    m = r[r.arm.str.startswith("matched")]
    e = r[r.arm.str.startswith("real")]
    A.note(f"본문이 배수를 주는 예산 {int(bud):,}")
    A.check("정합 최소",
            find(r"ratio is ([\d.]+) to [\d.]+-fold on the matched", t),
            m.ratio.min(), 0.06)
    A.check("정합 최대",
            find(r"ratio is [\d.]+ to ([\d.]+)-fold on the matched", t),
            m.ratio.max(), 0.06)
    A.check("실측 최소",
            find(r"against ([\d.]+) to [\d.]+-fold across the four", t),
            e.ratio.min(), 0.06)
    A.check("실측 최대",
            find(r"against [\d.]+ to ([\d.]+)-fold across the four", t),
            e.ratio.max(), 0.06)

    k = R[(R.budget == 1000) & R.arm.str.startswith("matched")]
    A.check("1,000에서 회수율 0.95 이상인 항원 수",
            word(r"(one|two|three|four|five|six|seven) of the seven matched "
                 r"libraries already recover", t),
            float((k.active >= 0.95).sum()), 0.5)
    ov = m[(m.ratio <= e.ratio.max()) & (m.ratio >= e.ratio.min())]
    A.note(f"예산 {int(bud):,}에서 실측 범위와 겹치는 정합 항원 {len(ov)}종")
    if len(ov):
        A.fail("원고는 겹치지 않는다고 적는데 겹칩니다.")

    # ------------------------------------------------------ 절반 도달 예산
    print("=== 절반 도달 예산 (exp80) ===")
    H = pd.read_csv("results/exp80_budget_to_half.csv")
    ha = H[H.strategy == "active"]
    hm = ha[ha.domain == "matched lattice"].dropna(subset=["median"])
    he = ha[ha.domain == "measured"]
    A.check("정합 최소 측정 횟수",
            find(r"needs (\d+) to \d+ measurements", t)
            or find(r"That is (\d+) to \d+ on the seven", t),
            float(hm["median"].min()), 5)
    A.check("정합 최대 측정 횟수",
            find(r"needs \d+ to (\d+) measurements", t)
            or find(r"That is \d+ to (\d+) on the seven", t),
            float(hm["median"].max()), 5)
    reached = sorted(he["median"].dropna().round(0).astype(int).tolist())
    A.note(f"실측 도달 {reached} · 미도달 {int(he['median'].isna().sum())}종")

    # ------------------------------------------------- 라이브러리의 설계
    print("=== 정합 라이브러리의 설계 ===")
    for lib in sorted(glob.glob("data/absolut/matched_*.parquet")):
        ag = os.path.basename(lib)[len("matched_"):-len(".parquet")]
        dd = pd.read_parquet(lib)
        seqs = dd.cdr3.tolist()
        L = len(seqs[0])
        arr = np.array([list(x) for x in seqs])
        par = "".join(pd.Series(arr[:, j]).mode()[0] for j in range(L))
        order = np.array([sum(p != q for p, q in zip(x, par)) for x in seqs])
        n1 = int((order == 1).sum())
        nv = len({j for x in seqs for j in range(L) if x[j] != par[j]})
        odd = int((order > 3).sum())
        # 약한 팔은 세기 대조용이라 6,000개가 아니라 300개다. 그 차이는 설계이지
        # 결함이 아니므로 규격을 갈라 잰다. 한 규격으로 재면 의도한 차이가
        # 어긋남으로 보고되고, 그런 보고가 쌓이면 감사를 믿지 않게 된다.
        weak = ag.endswith("_weak")
        n_want = 300 if weak else 6000
        good = (len(dd) == n_want and L == 17 and n1 == 285 and nv == 15
                and odd == 0)
        if not good:
            A.bad += 1
        kind = "약한 팔" if weak else "정합"
        print(f"  [{(OK if good else BAD):6s}] {ag} ({kind}): {len(dd):,}개 "
              f"· 길이 {L} · 단일변이 {n1} · 변이 자리 {nv} "
              f"· 4차 이상 {odd}")

    # ------------------------------------------------------------ 항원 수
    print("=== 항원 수 일치 ===")
    n71 = pd.read_csv("results/exp71_per_antigen.csv").antigen.nunique()
    r72 = pd.read_csv("results/exp72_ratios.csv")
    n72 = r72[r72.arm.str.startswith("matched:")].arm.nunique()
    said = word(r"(one|two|three|four|five|six|seven) matched librar", t)
    A.note(f"본문 {said}종 · exp71 {n71}종 · exp72 {n72}종")
    if said != n71 or said != n72:
        A.fail("본문과 실험의 항원 수가 다릅니다. 네 실험(71, 72, 73, 76)을 "
               "같은 집합으로 다시 돌리고 본문을 갱신하십시오.")
        pat = re.compile(r"\b(one|two|three|four|five|six|seven)\b"
                         r"(?=[^.]{0,60}(matched|antigen|librar))")
        hits = [(i + 1, ln.strip()) for i, ln in enumerate(raw.split(NL))
                if pat.search(ln) and "parental" not in ln]
        print(f"  {'':8s} 수를 적은 자리 {len(hits)}곳 (무관한 것 포함):")
        for i, ln in hits:
            print(f"  {'':10s} {i:4d}  {ln[:86]}")

    # -------------------------------------------------------- 표현 비교
    print("=== 표현 비교 (exp71) ===")
    S = pd.read_csv("results/exp71_per_antigen.csv")
    P = pd.read_csv("results/exp71_per_size.csv")
    # 원고는 절댓값이 작은 쪽에서 큰 쪽으로 적는다("ran from -0.005 to -0.022").
    # 그러면 첫 값이 mean_diff 의 최댓값(0에 가까운 쪽)이다.
    A.check("통합 범위의 첫 값(0에 가까운 쪽)",
            find(r"ran from \$-0\.0(\d\d)\$ to", t) and
            -find(r"ran from \$-0\.0(\d\d)\$ to", t) / 1000,
            S.mean_diff.max(), 0.002)
    A.check("통합 범위의 끝 값(가장 큰 우위)",
            find(r"ran from \$-0\.0\d\d\$ to \$-0\.0(\d\d)\$", t) and
            -find(r"ran from \$-0\.0\d\d\$ to \$-0\.0(\d\d)\$", t) / 1000,
            S.mean_diff.min(), 0.002)
    a200 = P[P.n_train == 200].mean_diff
    A.check("200라벨 최소",
            find(r"between \$-0\.0(\d\d)\$ and", t) and
            -find(r"between \$-0\.0(\d\d)\$ and", t) / 1000,
            a200.max(), 0.002)
    A.check("200라벨 최대",
            find(r"between \$-0\.0\d\d\$ and \$-0\.0(\d\d)\$", t) and
            -find(r"between \$-0\.0\d\d\$ and \$-0\.0(\d\d)\$", t) / 1000,
            a200.min(), 0.002)
    big = P[P.n_train == P.n_train.max()]
    neg = int((big.ci_hi < 0).sum())
    zero = int(((big.ci_lo <= 0) & (big.ci_hi >= 0)).sum())
    pos = int((big.ci_lo > 0).sum())
    A.note(f"{int(big.n_train.iloc[0]):,}라벨에서 ESM 우위 {neg}종, "
           f"0 포함 {zero}종, one-hot 우위 {pos}종")
    A.check("큰 라벨에서 one-hot이 앞선 항원 수",
            word(r"one-hot is ahead in (one|two|three|four|five|six|seven) "
                 r"of the seven", t), float(pos), 0.5)

    # ---------------------------------------------------------- 남은 몫
    print("=== 남은 몫의 상관 (exp73) ===")
    D = pd.read_csv("results/exp73_why_residual.csv").dropna(subset=["ratio"])
    me = D[D.domain == "measured"]
    r_me = (float(np.corrcoef(me.rho, me.ratio)[0, 1]) if len(me) > 2
            else float("nan"))
    A.check("실측 군집 안 r", find(r"\$r = ([\d.]+)\$, \$p = ", t), r_me, 0.01)
    A.note(f"전체 {len(D)}팔 Pearson "
           f"{float(np.corrcoef(D.rho, D.ratio)[0, 1]):+.2f} · 실측 군집 "
           f"{len(me)}팔 {r_me:+.2f}")
    lo = D[D.domain == "measured"].n_var
    hi = D[D.domain == "matched lattice"].n_var
    A.note(f"자리 수 격자 {int(hi.min())}~{int(hi.max())}, "
           f"실측 {int(lo.min())}~{int(lo.max())} (원고는 15 대 59~89)")

    # -------------------------------------------------------------- 차원
    print("=== 차원 (exp76) ===")
    W = pd.read_csv("results/exp76_random_positions.csv")
    gain = float(W[W.k == W.k.min()].rho.mean()
                 - W[W.k == W.k.max()].rho.mean())
    A.check("15자리→8자리 이득",
            find(r"by \$\+0\.0(\d\d)\$ on average from 15", t) and
            find(r"by \$\+0\.0(\d\d)\$ on average from 15", t) / 1000,
            gain, 0.002)
    n_ag = W.antigen.nunique()
    A.check("부호 검정 p", find(r"sign test \$p = ([\d.]+)\$; Supplementary "
                              r"Table~S11", t), 2.0 ** -n_ag, 0.001)

    # ------------------------------------------------------------ 무결성
    print("=== 무결성 (exp77) ===")
    p_old = "data/absolut/unreliable/matched_1NCA_N.parquet"
    p_new = "data/absolut/matched_1NCA_N.parquet"
    if os.path.exists(p_old) and os.path.exists(p_new):
        o = pd.read_parquet(p_old)[["cdr3", "energy"]]
        n = pd.read_parquet(p_new)[["cdr3", "energy"]]
        mm = o.merge(n, on="cdr3", suffixes=("_o", "_n"))
        d = (mm.energy_o - mm.energy_n).abs()
        A.check("어긋난 서열 수",
                find(r"for (5\d\d\d) of", t.replace("{,}", "")),
                float((d > 1e-3).sum()), 1)
        A.check("중앙 차이", find(r"median of ([\d.]+) energy units", t),
                float(d.median()), 0.05)
    else:
        skipped.append("무결성(exp77) — 라이브러리 parquet 이 없습니다")
        print(f"  [{SKIP:6s}] 라이브러리 parquet 이 없습니다")

    print()
    for x in skipped:
        print("건너뜀:", x)
    print("어긋난 항목 없음" if not A.bad else f"어긋난 항목 {A.bad}개")
    return 1 if A.bad else 0


if __name__ == "__main__":
    sys.exit(main())
