"""실험 81 — 오라클은 항원을 구별하는가, 아니면 '잘 붙는 서열'을 재는가.

2Q8A_A 의 정합 라이브러리를 만들다 걸린 것에서 출발한다. 길이 17짜리
748,021개 가운데 에너지가 가장 낮은 서열이 1ZTX_E 와 **같았다**. 두 항원은
다르고 에너지도 다른데(-118.27 대 -114.47) 최고 결합체가 같다.

파서 사고인 줄 알았으나 아니다. zip 파서를 거치지 않는 oracle_*.parquet 에서도
같고, 나머지 네 항원은 서로 다른 모항체를 낸다. 실제 성질이다.

그러면 얼마나 실제인지 재야 한다. 논문은 이미 동족 항체의 에너지 백분위가
균일하다고 보고하지만, 그것은 "짝짓기 정보가 없다"의 간접 증거다. 직접 증거는
이것이다 -- 같은 서열 691만 개를 여섯 항원에 대해 채점한 값들이 서로 얼마나
닮았는가.

  상관이 높으면  오라클은 '이 서열이 일반적으로 잘 붙는가'를 재고 있고,
                항원은 그 값을 조금 움직일 뿐이다.
  상관이 낮으면  항원마다 다른 것을 재고 있고, 동족 항체 결과는 다른 이유로
                설명해야 한다.

상위 결합체의 겹침도 함께 센다. 독립이라면 상위 N개 두 집합의 기대 겹침은
N^2/M 이므로(M 은 후보 수), 관측값이 그보다 몇 배인지가 곧 효과 크기다.

    python src/exp81_cross_antigen_energy.py
"""
import io
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ENERGIES = "data/absolut/energies.npz"
LIB = "data/absolut/library.parquet"
TOPS = [10, 100, 1000]
SAMPLE = 500_000          # 상관은 표본으로 충분하다
OUT = "results/exp81_cross_antigen.csv"
OUT2 = "results/exp81_top_overlap.csv"
REPORT = "results/exp81_cross_antigen.md"
NL = chr(10)


def main():
    if not os.path.exists(ENERGIES):
        sys.exit(f"{ENERGIES} 가 없습니다.")
    E = np.load(ENERGIES)
    ags = sorted(E.files)
    print(f"항원 {len(ags)}종: {', '.join(ags)}")

    n = len(E[ags[0]])
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, size=min(SAMPLE, n), replace=False))
    mat = {a: E[a][idx].astype(np.float64) for a in ags}
    bad = {a: int(np.isnan(v).sum()) for a, v in mat.items()}
    if any(bad.values()):
        print("  결측:", {k: v for k, v in bad.items() if v})

    # 길이를 통제한다. 격자 에너지는 서열이 길수록 낮아지므로, 길이를 섞어
    # 재면 상관의 상당 부분이 "둘 다 긴 서열을 좋아한다"일 수 있다. 길이 17
    # 안에서 다시 재서 그 몫을 뺀다.
    L17 = None
    if os.path.exists(LIB):
        lens = pd.read_parquet(LIB)["cdr3"].str.len().to_numpy()[idx]
        L17 = lens == 17
        print(f"  길이 17 표본 {int(L17.sum()):,}개")

    rows = []
    for a, b in itertools.combinations(ags, 2):
        x, y = mat[a], mat[b]
        ok = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        rho = float(spearmanr(x[ok], y[ok]).statistic)
        r17 = np.nan
        if L17 is not None:
            m17 = ok & L17
            if m17.sum() > 100:
                r17 = float(np.corrcoef(x[m17], y[m17])[0, 1])
        rows.append(dict(a=a, b=b, n=int(ok.sum()),
                         pearson=round(r, 4), spearman=round(rho, 4),
                         pearson_len17=round(r17, 4)
                         if np.isfinite(r17) else np.nan))
        print(f"  {a:8s} {b:8s} Pearson {r:+.3f} · Spearman {rho:+.3f}"
              + (f" · 길이17만 {r17:+.3f}" if np.isfinite(r17) else ""))
    C = pd.DataFrame(rows)
    C.to_csv(OUT, index=False, encoding="utf-8")

    # 상위 결합체 겹침. 길이를 가리지 않고 전체에서 센다.
    print()
    over = []
    for N in TOPS:
        top = {a: set(np.argsort(E[a])[:N].tolist()) for a in ags}
        exp_ov = N * N / n
        for a, b in itertools.combinations(ags, 2):
            ov = len(top[a] & top[b])
            over.append(dict(top=N, a=a, b=b, overlap=ov,
                             expected=exp_ov,
                             fold=round(ov / exp_ov, 1) if exp_ov else np.nan))
        g = [x for x in over if x["top"] == N]
        m = np.median([x["overlap"] for x in g])
        print(f"  상위 {N:5,}  겹침 중앙 {m:.0f}/{N:,} · 독립 기대 "
              f"{exp_ov:.5f} · 배수 {m / exp_ov:,.0f}")
    O = pd.DataFrame(over)
    O.to_csv(OUT2, index=False, encoding="utf-8")

    same_best = [(a, b) for a, b in itertools.combinations(ags, 2)
                 if int(np.argmin(E[a])) == int(np.argmin(E[b]))]
    lib = pd.read_parquet(LIB)["cdr3"].values if os.path.exists(LIB) else None

    L = ["# exp81 오라클은 항원을 구별하는가", "",
         "논문은 격자 오라클이 어느 항체가 어느 항원에 붙는지를 담고 있지 "
         "않다고 보고한다. 그 근거는 동족 항체의 에너지 백분위가 균일하다는 "
         "것인데, 그것은 간접 증거다. 직접 증거는 같은 서열 691만 개를 여러 "
         "항원에 대해 채점한 값이 서로 얼마나 닮았는지다.", "",
         f"{len(ags)}종의 항원, 표본 {len(idx):,}개.", "",
         "길이는 통제한다. 격자 에너지는 서열이 길수록 낮아지므로, 길이를 "
         "섞어 재면 상관의 일부가 '둘 다 긴 서열을 좋아한다'일 수 있다. "
         "마지막 열은 길이 17 안에서만 다시 잰 값이다.", "",
         "| 항원 A | 항원 B | Pearson | Spearman | 길이 17만 |",
         "|---|---|---|---|---|"]
    for _, x in C.iterrows():
        v17 = ("—" if pd.isna(x.pearson_len17) else f"{x.pearson_len17:+.3f}")
        L.append(f"| {x.a} | {x.b} | {x.pearson:+.3f} | {x.spearman:+.3f} | "
                 f"{v17} |")
    c17 = C.pearson_len17.dropna()
    L += ["", f"Pearson 중앙 {C.pearson.median():+.3f} "
          f"(범위 {C.pearson.min():+.3f} ~ {C.pearson.max():+.3f}), "
          f"Spearman 중앙 {C.spearman.median():+.3f}."
          + (f" 길이 17 안에서는 중앙 {c17.median():+.3f} "
             f"(범위 {c17.min():+.3f} ~ {c17.max():+.3f})로, 길이를 빼도 "
             f"상관이 남는다." if len(c17) else ""), "",
          "## 상위 결합체의 겹침", "",
          "독립이라면 상위 N개 두 집합의 기대 겹침은 N^2/M 이다"
          f"(M = {n:,}).", "",
          "| 상위 N | 겹침 중앙 | 독립 기대 | 배수 |", "|---|---|---|---|"]
    for N in TOPS:
        g = O[O.top == N]
        med = float(g.overlap.median())
        exp = float(g.expected.iloc[0])
        L.append(f"| {N:,} | {med:.0f} | {exp:.5f} | "
                 + (f"{med / exp:,.0f}배 |" if med > 0 else "0 |"))
    L += [""]
    if same_best:
        names = ", ".join(f"{a}/{b}" for a, b in same_best)
        seq = (lib[int(np.argmin(E[same_best[0][0]]))] if lib is not None
               else "")
        L += [f"최고 결합체가 완전히 같은 항원 쌍이 있다: {names}"
              + (f" (서열 {seq})" if seq else "") + ". 691만 개 가운데 1위가 "
              "일치한다.", ""]
    L += ["읽는 법. 서로 다른 항원에 대한 에너지가 이만큼 닮아 있다면, 오라클이 "
          "재는 것의 큰 부분은 '이 서열이 이 항원에 붙는가'가 아니라 '이 서열이 "
          "일반적으로 잘 붙는가'다. 동족 항체의 백분위가 균일한 것도 같은 "
          "사실의 다른 얼굴이다 -- 항원을 바꿔도 좋은 서열의 순위가 크게 바뀌지 "
          "않으므로, 어느 항체가 제 항원에서 특별히 높은 자리를 차지할 이유가 "
          "없다.", "",
          "이 관찰은 계획한 것이 아니다. 2Q8A_A 의 정합 라이브러리를 만들다가 "
          "모항체가 1ZTX_E 와 같아서 파서 사고를 의심했고, 독립된 파일에서 "
          "확인하니 실제였다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {OUT2}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
