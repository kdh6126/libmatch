"""실험 77 — 조용히 틀린 라이브러리는 어떤 결론을 내놓는가.

1NCA_N의 첫 라이브러리는 격리됐다가 파이프라인을 고친 뒤 다시 만들어 통과했다.
두 판을 나란히 놓으면 옛 판은 6,000개 중 5,909개가 다르고 중앙 7.0 에너지
단위, 최대 23.1만큼 어긋나 있다. 채점 실행이 빈 출력만 남긴 데다 다중 사슬
항원의 코드 경로에서 실패했기 때문이다.

중요한 것은 옛 판이 **그 자체로는 멀쩡해 보였다**는 점이다. 에너지 범위도
그럴듯하고, 단일·이중·삼중 변이 구조도 정확하고, 모항체가 최저에 가깝다.
내부 검사로는 걸리지 않는다. 공개 표와의 대조만이 잡았다.

그러면 그 라이브러리로 분석을 돌렸다면 무엇이 나왔을까. 이 논문의 주장이
"통제하지 않은 비교가 오도한다"이므로, 통제하지 않은 **오라클 실행**이
그럴듯하면서 다른 답을 내놓는다면 그것도 같은 주장의 사례다.

표현 비교(exp71과 같은 규약)를 두 판에 각각 돌려 비교한다.

    python src/exp77_corrupt_conclusion.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, ttest_ind
from sklearn.linear_model import Ridge

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")
import exp71_repr_many_antigens as E71                          # noqa: E402

AG = "1NCA_N"
PARENT = "CAREFDLFGLLLLFDYW"
OLD = "data/absolut/unreliable/matched_1NCA_N.parquet"
NEW = "data/absolut/matched_1NCA_N.parquet"
SIZES = E71.SIZES
SEEDS = E71.SEEDS
HOLD = E71.HOLD
OUT = "results/exp77_corrupt_conclusion.csv"
REPORT = "results/exp77_corrupt_conclusion.md"
NL = chr(10)


def arm(path, tag):
    d = pd.read_parquet(path)
    d = d[d.cdr3.str.len() == len(PARENT)].reset_index(drop=True)
    y = d.energy.to_numpy(np.float32)
    Xs, var = E71.encode(d.cdr3.tolist(), PARENT)
    Xe = E71.embeddings(d.cdr3.tolist(), var, tag)
    rows = []
    for s in range(SEEDS):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:HOLD], perm[HOLD:]
        yte = y[te]
        for m in SIZES:
            if m > len(pool):
                continue
            tr = pool[:m]
            r = Ridge(alpha=1.0, solver="lsqr").fit(Xs[tr], y[tr])
            a = float(spearmanr(yte, Xs[te] @ r.coef_ + r.intercept_).statistic)
            re = Ridge(alpha=50.0, solver="lsqr").fit(Xe[tr], y[tr])
            b = float(spearmanr(yte, re.predict(Xe[te])).statistic)
            rows.append(dict(library=tag, n_train=m, seed=s,
                             onehot=round(a, 4), esm=round(b, 4),
                             onehot_minus_esm=round(a - b, 4)))
    return pd.DataFrame(rows), d


def main():
    for p in (OLD, NEW):
        if not os.path.exists(p):
            sys.exit(f"{p} 가 없습니다.")

    a_old = pd.read_parquet(OLD)[["cdr3", "energy"]]
    a_new = pd.read_parquet(NEW)[["cdr3", "energy"]]
    m = a_old.merge(a_new, on="cdr3", suffixes=("_old", "_new"))
    d = (m.energy_old - m.energy_new)
    print(f"두 판 공통 {len(m):,}개 · 다른 것 {int((d.abs() > 1e-3).sum()):,} · "
          f"중앙 |차| {d.abs().median():.2f} · 최대 {d.abs().max():.2f} · "
          f"옛 판이 더 큰(덜 음수인) 비율 {(d > 0).mean():.1%}")

    R_old, D_old = arm(OLD, f"corrupt_{AG}")
    R_new, D_new = arm(NEW, f"matched_{AG}")
    R = pd.concat([R_old.assign(library="quarantined"),
                   R_new.assign(library="rebuilt")], ignore_index=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    S = []
    for (lib, n), g in R.groupby(["library", "n_train"]):
        v = g.onehot_minus_esm.to_numpy()
        S.append(dict(library=lib, n_train=int(n), mean_diff=round(v.mean(), 4),
                      sd=round(v.std(ddof=1), 4),
                      esm_wins=int((v < 0).sum()), n=len(v),
                      onehot=round(g.onehot.mean(), 4),
                      esm=round(g.esm.mean(), 4)))
    S = pd.DataFrame(S).sort_values(["n_train", "library"])
    print()
    print(S.to_string(index=False))

    L = ["# exp77 조용히 틀린 라이브러리는 어떤 결론을 내놓는가", "",
         f"{AG}의 첫 라이브러리는 채점 실행이 실패했는데도 파일이 남았고, "
         "격리됐다가 파이프라인을 고친 뒤 다시 만들어 통과했다. 두 판은 "
         f"{len(m):,}개 가운데 {int((d.abs() > 1e-3).sum()):,}개가 다르고, "
         f"중앙 {d.abs().median():.1f} 에너지 단위, 최대 {d.abs().max():.1f}만큼 "
         f"어긋나 있다. 옛 판이 더 약하게 결합한 쪽으로 치우쳐 있다"
         f"({(d > 0).mean():.0%}).", "",
         "옛 판은 그 자체로는 멀쩡해 보인다. 에너지 범위가 그럴듯하고, 변이 "
         "구조가 정확하고, 모항체가 최저 부근에 있다. 내부 검사로는 걸리지 "
         "않는다 -- 공개 표와의 대조만이 잡았다.", "",
         "그 라이브러리로 분석했다면 무엇이 나왔는지 본다. 표현 비교를 exp71과 "
         "같은 규약으로 두 판에 각각 돌렸다. 음수는 언어 모형 우위다.", "",
         "| 라벨 | 라이브러리 | one-hot | ESM | 차 | ESM 우세 |",
         "|---|---|---|---|---|---|"]
    for _, x in S.iterrows():
        name = "격리본" if x.library == "quarantined" else "재작성본"
        L.append(f"| {int(x.n_train):,} | {name} | {x.onehot:.3f} | "
                 f"{x.esm:.3f} | {x.mean_diff:+.4f} | "
                 f"{int(x.esm_wins)}/{int(x.n)} |")

    o = S[S.library == "quarantined"].set_index("n_train")
    n = S[S.library == "rebuilt"].set_index("n_train")
    flips = [int(k) for k in o.index
             if np.sign(o.loc[k, "mean_diff"]) != np.sign(n.loc[k, "mean_diff"])]
    t = ttest_ind(R[R.library == "quarantined"].onehot_minus_esm,
                  R[R.library == "rebuilt"].onehot_minus_esm, equal_var=False)
    L += ["", f"부호가 뒤집히는 학습 크기: "
          + (", ".join(f"{k:,}" for k in flips) if flips else "없음") + ".",
          f"두 판의 차이 분포 비교: Welch $t$ p = {t.pvalue:.2g}.", ""]
    if flips:
        L += ["같은 항원, 같은 설계, 같은 모형인데 오라클 실행 하나가 실패한 "
              "것만으로 표현 선호의 결론이 바뀐다. 이 논문의 주장이 '통제하지 "
              "않은 비교가 오도한다'인데, 통제하지 않은 오라클 실행도 같은 "
              "일을 한다.", "",
              "여기에 불편한 대목이 하나 있고, 적어 두는 편이 낫다. 격리본이 "
              "내놓는 패턴 -- 모든 예산에서 언어 모형이 앞선다 -- 은 실측 "
              "데이터에서 나오는 패턴과 같다. 즉 그 조용한 손상은 격자를 실제 "
              "데이터처럼 보이게 만들었고, 그 방향은 이 논문 자신의 주장을 더 "
              "강하게 지지하는 쪽이었다. 검사가 막아 준 것은 남의 오류가 "
              "아니라 우리에게 유리한 거짓 양성이다. 자기 결론을 강화하는 "
              "자료일수록 바깥 기준과 맞춰 봐야 하는 이유가 여기 있다.", "",
              "손상이 잡음처럼 보이지도 않는다. 격리본은 전반적으로 덜 "
              "학습 가능하다(one-hot 0.69~0.81 대 0.80~0.94). 무작위 잡음이 "
              "아니라 구조를 가진 오차이고, 다중 사슬 항원을 일부만 싣고 "
              "채점했다면 정확히 이런 모양이 된다.", ""]
    else:
        L += ["결론의 방향은 바뀌지 않았다. 그래도 크기는 달라지므로, 격리 "
              "판단 자체는 옳다 -- 다만 '결론이 뒤집혔을 것'이라고 말할 근거는 "
              "이 자료에 없고, 그렇게 적어야 한다.", ""]
    L += ["옛 판이 왜 틀렸는지는 따로 기록돼 있다. 채점 실행이 빈 출력만 남겼고, "
          "재채점은 두 번 멈췄으며, 이 항원은 다중 사슬이라 단일 사슬 항원과 "
          "다른 코드 경로를 탄다. 실패가 예외로 오르지 않고 조용히 버려진 것이 "
          "직접 원인이고, 그 뒤 채점 함수가 비정상 종료에 예외를 올리도록 "
          "고쳐졌다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
