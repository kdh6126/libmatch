"""실험 80 — 상위 1%의 절반을 찾는 데 몇 번 재야 하는가.

이 분야가 보고하는 양은 무작위 대비 배수인데, 정합 격자에서는 그 양이
측정되지 않는다. 예산 1,000에서 유도 탐색의 회수율이 다섯 중 넷에서 0.95를
넘고 하나는 1.00이다. 분자가 천장에 닿으면 배수는 값이 아니라 하한이고, 예산을
500으로 낮춰도 0.80~0.94라 사정이 같다. 유도 탐색이 이 라이브러리를 사실상
풀어 버린다.

천장이 없는 양으로 바꿔 읽는다. 회수율을 예산의 함수로 보고, 그것이 절반에
이르는 예산을 구한다. 회수율은 예산이 늘수록 단조 증가하므로 이 값은 잘
정의되고, "상위 1%의 절반을 찾으려면 몇 번 재야 하는가"로 바로 읽힌다.

exp72 가 예산 200·500·1,000에서 시드마다 회수율을 남겼으므로 새로 돌릴 것은
없다. 시드별 곡선에서 교차점을 구하고 시드에 걸쳐 분포를 낸다. 범위 안에서
절반에 이르지 못하는 팔은 ">1,000"으로 적는다 -- 외삽하지 않는다.

    python src/exp80_budget_to_half.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SRC = "results/exp72_search_seeds.csv"
TARGET = 0.5
OUT = "results/exp80_budget_to_half.csv"
REPORT = "results/exp80_budget_to_half.md"
NL = chr(10)


def crossing(budgets, recalls, target=TARGET):
    """회수율이 target 에 이르는 예산. 로그 예산에서 선형 보간한다.

    회수율은 예산에 대해 대략 로그 선형이므로 로그 축에서 잇는다. 마지막
    예산에서도 target 에 못 미치면 외삽하지 않고 없음을 돌려준다 -- 시험하지
    않은 구간의 값을 만들어 내는 것은 측정이 아니다.
    """
    b = np.asarray(budgets, float)
    r = np.asarray(recalls, float)
    o = np.argsort(b)
    b, r = b[o], r[o]
    if r[-1] < target:
        return np.nan
    if r[0] >= target:
        return float(b[0])
    for i in range(1, len(b)):
        if r[i] >= target:
            lo, hi = r[i - 1], r[i]
            f = (target - lo) / (hi - lo) if hi > lo else 0.0
            return float(np.exp(np.log(b[i - 1])
                                + f * (np.log(b[i]) - np.log(b[i - 1]))))
    return np.nan


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} 가 없습니다. exp72를 먼저 돌리십시오.")
    d = pd.read_csv(SRC)
    col = "recall" if "recall" in d.columns else "recall_1pct"
    rows = []
    for (arm, strat), g in d.groupby(["arm", "strategy"]):
        if strat not in ("active", "random"):
            continue
        per = []
        for seed, gg in g.groupby("seed"):
            gg = gg.sort_values("budget")
            per.append(crossing(gg.budget.to_numpy(), gg[col].to_numpy()))
        per = np.asarray(per, float)
        got = per[np.isfinite(per)]
        rows.append(dict(
            arm=arm, strategy=strat, seeds=len(per),
            reached=int(len(got)),
            median=round(float(np.median(got)), 1) if len(got) else np.nan,
            lo=round(float(np.percentile(got, 25)), 1) if len(got) else np.nan,
            hi=round(float(np.percentile(got, 75)), 1) if len(got) else np.nan))
    R = pd.DataFrame(rows)
    R["domain"] = np.where(R.arm.str.startswith("matched:"),
                           "matched lattice", "measured")
    R = R.sort_values(["strategy", "domain", "median"])
    R.to_csv(OUT, index=False, encoding="utf-8")
    print(R.to_string(index=False))

    act = R[R.strategy == "active"]
    mt = act[act.domain == "matched lattice"].dropna(subset=["median"])
    me = act[act.domain == "measured"].dropna(subset=["median"])
    never = act[act.reached < act.seeds]

    L = ["# exp80 상위 1%의 절반을 찾는 데 드는 측정 횟수", "",
         "무작위 대비 배수는 정합 격자에서 측정되지 않는다. 예산 1,000에서 "
         "유도 탐색의 회수율이 다섯 중 넷에서 0.95를 넘고, 예산을 500으로 "
         "낮춰도 0.80~0.94다. 분자가 천장에 닿으면 배수는 값이 아니라 하한이다.",
         "",
         "천장이 없는 양으로 바꿔 읽는다. 회수율이 절반에 이르는 예산은 "
         "단조 증가하는 곡선 위의 한 점이고, '상위 1%의 절반을 찾으려면 몇 번 "
         "재야 하는가'로 바로 읽힌다. exp72 의 시드별 곡선에서 로그 예산 선형 "
         "보간으로 구했고, 시험한 범위(200~1,000) 안에서 절반에 이르지 못한 "
         "시드는 외삽하지 않고 따로 셌다.", "",
         "| 팔 | 종류 | 전략 | 절반 도달 예산 (중앙) | 사분위 | 도달 시드 |",
         "|---|---|---|---|---|---|"]
    for _, x in R.iterrows():
        val = "&gt;1,000" if not np.isfinite(x["median"]) else f"{x['median']:,.0f}"
        iqr = ("—" if not np.isfinite(x.lo)
               else f"{x.lo:,.0f} ~ {x.hi:,.0f}")
        L.append(f"| {x.arm} | {x.domain} | {x.strategy} | {val} | {iqr} | "
                 f"{int(x.reached)}/{int(x.seeds)} |")
    L += [""]
    if len(mt) and len(me):
        L += [f"유도 탐색으로 절반에 이르는 예산은 정합 격자가 "
              f"{mt['median'].min():,.0f}~{mt['median'].max():,.0f}회, "
              f"실측이 {me['median'].min():,.0f}~{me['median'].max():,.0f}회다.",
              ""]
    if len(never):
        L += ["시험한 범위 안에서 절반에 이르지 못한 팔이 있다. 그 칸은 "
              "'>1,000'으로 적었고, 얼마나 더 필요한지는 이 자료로 말할 수 "
              "없다.", ""]
    L += ["이 지표도 완전하지 않다. 예산을 셋만 재었으므로 보간의 해상도가 "
          "거기까지이고, 정합 격자의 값은 가장 낮은 예산(200)에 가까워 아래쪽 "
          "끝에 눌린다. 더 낮은 예산은 탐색 루틴의 초기 표본 하한(200) 때문에 "
          "잴 수 없다. 자릿수로 읽어야 한다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
