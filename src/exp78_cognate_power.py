"""실험 78 — 동족 항체 검사는 얼마나 작은 효과까지 잡아내는가.

원고의 Discussion 이 "모의 자료에 짝짓기 효과가 있으면 이 절차가 그것을
회수한다"고 적어 두었는데, 그 모의가 어디에도 없다. Methods 에도 보충자료에도
없고 결과표도 없다. 음성 결과를 보고하는 논문에서 감도 주장은 심사자가 가장
먼저 확인하는 자리이므로, 근거 없는 문장을 두느니 재는 편이 낫다.

없음을 보고하려면 기구의 분해능을 함께 보고해야 한다. "신호가 없다"는 "이만한
신호였다면 보였을 것이다"와 짝을 지어야 뜻이 생긴다.

재는 방법. 관측된 88종의 동족 항체 에너지를 delta 만큼 낮춘다(더 잘 결합하게
만든다). 그러면 각 항원의 백분위가 내려간다. 그 상태에서 원고가 쓰는 검정을
그대로 돌려, 어느 delta 부터 기각되는지 본다.

백분위를 다시 계산하려면 각 항원의 에너지 분포가 필요한데 전수표는 450MB
짜리다. 대신 두 가지를 쓴다.

  정확   에너지 배열을 들고 있는 6종은 실제로 옮겨 다시 센다.
  근사   나머지는 정규 근사를 쓴다. 표에 중앙값과 (에너지, 백분위) 쌍이 있으므로
         표준편차가 풀린다: sigma = (e - m) / Phi^-1(p).

근사가 얼마나 맞는지는 6종에서 둘을 나란히 놓아 확인한다. 확인 없이 근사를
쓰면 그것은 근사가 아니라 짐작이다.

    python src/exp78_cognate_power.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SRC = "results/exp62_native_fulltable.csv"
ENERGIES = "data/absolut/energies.npz"
DELTAS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
N_BOOT = 2000
ALPHA = 0.05
OUT = "results/exp78_cognate_power.csv"
REPORT = "results/exp78_cognate_power.md"
NL = chr(10)


def load():
    """본문의 주분석과 같은 88종을 쓴다.

    status == ok 만으로는 93종이 된다. 본문의 88종은 거기에 두 가지를 더
    뺀 것이다 -- exp67 이 전제 실패로 판정한 항원(항체가 벤치마크가 쓰는 사슬에
    붙지 않는 경우 등)과, 공개표 겹침이 99% 미만인 항원. 감도를 다른 집합에서
    재면 그 값은 본문의 음성 결과에 대한 감도가 아니다.
    """
    d = pd.read_csv(SRC)
    d = d[d.status == "ok"].copy()
    pre = "results/exp67_epitope_precheck.csv"
    if os.path.exists(pre):
        q = pd.read_csv(pre)
        bad = set(q[q.verdict == "전제 실패"].antigen.astype(str))
        n0 = len(d)
        d = d[~d.antigen.astype(str).isin(bad)]
        print(f"  전제 실패 {n0 - len(d)}종 제외")
    n0 = len(d)
    d = d[d.coverage_pct.fillna(100) >= 99]
    if n0 != len(d):
        print(f"  겹침 99% 미만 {n0 - len(d)}종 제외")
    d = d[d.pct_better.notna() & d.native_energy.notna()
          & d.ref_median.notna()]
    p = (d.pct_better / 100.0).clip(1e-9, 1 - 1e-9)
    # e = m + sigma * Phi^-1(p)  =>  sigma = (e - m) / Phi^-1(p)
    z = stats.norm.ppf(p)
    d["sigma"] = (d.native_energy - d.ref_median) / z
    bad = ~np.isfinite(d.sigma) | (d.sigma <= 0)
    if bad.any():
        print(f"  표준편차를 풀 수 없는 항원 {int(bad.sum())}종을 뺍니다 "
              f"(백분위가 정확히 50%이거나 부호가 맞지 않습니다)")
        d = d[~bad]
    return d.reset_index(drop=True)


def shifted_pct_normal(d, delta):
    """delta 만큼 낮춘 동족 에너지의 백분위(정규 근사)."""
    z = (d.native_energy - delta - d.ref_median) / d.sigma
    return stats.norm.cdf(np.asarray(z, float))


def exact_check(d, deltas):
    """에너지 배열을 가진 항원에서 정확한 값과 근사를 나란히 놓는다."""
    if not os.path.exists(ENERGIES):
        return None
    E = np.load(ENERGIES)
    rows = []
    for _, x in d.iterrows():
        if x.antigen not in E.files:
            continue
        e = E[x.antigen]
        e = e[np.isfinite(e)]
        for delta in deltas:
            exact = float((e < x.native_energy - delta).mean())
            approx = float(stats.norm.cdf(
                (x.native_energy - delta - x.ref_median) / x.sigma))
            rows.append(dict(antigen=x.antigen, delta=delta,
                             exact=round(exact, 5), approx=round(approx, 5),
                             err=round(abs(exact - approx), 5)))
    return pd.DataFrame(rows) if rows else None


def tests(p):
    """원고가 쓰는 두 검정. p 는 0~1 백분위."""
    sign = stats.binomtest(int((p < 0.5).sum()), len(p), 0.5).pvalue
    ks = stats.kstest(p, "uniform").pvalue
    return sign, ks


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다.")
    d = load()
    print(f"동족 항체 검사를 통과한 항원 {len(d)}종 · "
          f"관측 백분위 중앙 {d.pct_better.median():.2f}%")

    chk = exact_check(d, [0, 3, 6, 10])
    if chk is not None:
        print(f"  정규 근사 검증({chk.antigen.nunique()}종, 에너지 배열 보유): "
              f"백분위 최대 오차 {chk.err.max():.4f}, "
              f"중앙 오차 {chk.err.median():.4f}")

    rows = []
    rng = np.random.default_rng(0)
    idx = np.arange(len(d))
    boots = [rng.choice(idx, len(idx), replace=True) for _ in range(N_BOOT)]
    for delta in DELTAS:
        p = shifted_pct_normal(d, delta)
        s_obs, k_obs = tests(p)
        s_hit = k_hit = 0
        for b in boots:
            s, k = tests(p[b])
            s_hit += s < ALPHA
            k_hit += k < ALPHA
        rows.append(dict(delta=delta,
                         median_pct=round(float(np.median(p)) * 100, 2),
                         p_sign=s_obs, p_ks=k_obs,
                         power_sign=round(s_hit / len(boots), 3),
                         power_ks=round(k_hit / len(boots), 3)))
        print(f"  delta {delta:4.1f}  중앙 백분위 {rows[-1]['median_pct']:5.2f}% "
              f"· 부호 p {s_obs:.2e} (검정력 {rows[-1]['power_sign']:.2f}) "
              f"· KS p {k_obs:.2e} (검정력 {rows[-1]['power_ks']:.2f})",
              flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False, encoding="utf-8")
    hit = R[R.power_sign >= 0.8]
    d80 = float(hit.delta.min()) if len(hit) else float("nan")
    hitk = R[R.power_ks >= 0.8]
    k80 = float(hitk.delta.min()) if len(hitk) else float("nan")

    L = ["# exp78 동족 항체 검사의 분해능", "",
         "원고는 격자 오라클이 어느 항체가 어느 항원에 붙는지를 담고 있지 "
         "않다고 보고한다. 없음을 보고하려면 기구의 분해능을 함께 보고해야 "
         "한다 -- '신호가 없다'는 '이만한 신호였다면 보였을 것이다'와 짝을 "
         "지어야 뜻이 생긴다.", "",
         f"관측된 {len(d)}종의 동족 항체 에너지를 delta 만큼 낮추고(더 잘 "
         "결합하게 만들고) 같은 검정을 다시 돌렸다. 검정력은 항원을 재표집한 "
         f"{N_BOOT:,}회에서 유의해지는 비율이다.", "",
         "| delta (에너지 단위) | 중앙 백분위 | 부호 검정 p | 검정력 | "
         "KS p | 검정력 |", "|---|---|---|---|---|---|"]
    for _, x in R.iterrows():
        L.append(f"| {x.delta:g} | {x.median_pct:.2f}% | {x.p_sign:.1e} | "
                 f"{x.power_sign:.2f} | {x.p_ks:.1e} | {x.power_ks:.2f} |")
    L += ["",
          f"부호 검정이 80% 검정력에 이르는 지점은 delta = {d80:g} 에너지 "
          f"단위이고, 그때 동족 항체의 중앙 백분위는 "
          f"{float(R[R.delta == d80].median_pct.iloc[0]):.1f}% 다."
          if np.isfinite(d80) else
          "부호 검정은 시험한 범위 안에서 80% 검정력에 이르지 못한다.",
          f"KS 검정은 delta = {k80:g} 에서 같은 수준에 이른다."
          if np.isfinite(k80) else
          "KS 검정도 이 범위에서는 80%에 이르지 못한다.", "",
          "읽는 법. 이 값보다 큰 짝짓기 효과가 있었다면 이 설계로 보였을 "
          "것이고, 그보다 작은 효과는 보이지 않았을 것이다. 관측값은 delta = 0 "
          "줄이고, 거기서 검정은 균일 분포와 구별되지 않는다.", "",
          "## 근사에 대하여", ""]
    if chk is not None:
        L += [f"백분위를 다시 세려면 항원마다 691만 개의 에너지가 필요한데 "
              f"전수표는 450MB 짜리다. 대신 표에 있는 중앙값과 (에너지, "
              f"백분위) 쌍으로 정규 근사의 표준편차를 풀었다. 에너지 배열을 "
              f"들고 있는 {chk.antigen.nunique()}종에서 정확한 값과 나란히 "
              f"놓으면 백분위 오차가 최대 {chk.err.max():.4f}, 중앙 "
              f"{chk.err.median():.4f} 다.", ""]
        if chk.err.max() > 0.05:
            L += ["이 오차는 작지 않다. 아래 표의 delta 값은 자릿수로만 읽어야 "
                  "한다.", ""]
    else:
        L += ["정규 근사를 검증할 에너지 배열이 없다. 그 확인 없이는 아래 "
              "delta 값을 자릿수로도 신뢰하기 어렵다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
