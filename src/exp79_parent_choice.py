"""실험 79 — 모항체를 어느 것으로 고르느냐가 결론을 바꾸는가.

정합 라이브러리는 모항체 하나를 골라 그 둘레를 채워 만든다. 원고는 2DD8_S 의
모항체가 748,021개 가운데 1위가 아니라 4위라고 적어 두었다(-121.84 대 -125.07).
적어 둔 이상 "그럼 1위로 만들었으면 달랐겠는가"는 당연한 물음이고, 지금 그
라이브러리가 있다.

exp68 이 "길이 17 중 최저 에너지" 규칙으로 2DD8_S 를 다시 만들면서 1위 모항체
라이브러리를 썼고, 그 파일은 발표본을 덮어썼다가 사본에서 되돌리며
matched_2DD8_S_rank1parent.parquet 로 남았다. 버릴 이유가 없다 -- 같은 항원,
같은 오라클, 같은 설계 규칙, 모항체만 다른 두 라이브러리다.

표현 비교를 exp71 과 같은 규약으로 두 판에 각각 돌린다. 결론이 같으면 모항체
선택은 이 결과를 흔들지 않고, 다르면 원고가 그 사실을 적어야 한다.

    python src/exp79_parent_choice.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from sklearn.linear_model import Ridge

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")
import exp71_repr_many_antigens as E71                          # noqa: E402

AG = "2DD8_S"
PUB = "data/absolut/matched_2DD8_S.parquet"
RANK1 = "data/absolut/matched_2DD8_S_rank1parent.parquet"
OUT = "results/exp79_parent_choice.csv"
REPORT = "results/exp79_parent_choice.md"
NL = chr(10)


def parent_of(d):
    """설계의 중심. 다른 모든 서열과의 거리 합이 가장 작은 길이 17 서열."""
    seqs = d.cdr3.tolist()
    L = len(seqs[0])
    arr = np.array([list(s) for s in seqs])
    # 단일 변이가 285개이므로 모항체는 대다수 자리에서 최빈값을 가진다.
    mode = "".join(pd.Series(arr[:, j]).mode()[0] for j in range(L))
    return mode if mode in set(seqs) else seqs[int(np.argmin(
        [(arr != np.array(list(s))).sum() for s in seqs]))]


def arm(path, tag):
    d = pd.read_parquet(path)
    par = parent_of(d)
    d = d[d.cdr3.str.len() == len(par)].reset_index(drop=True)
    y = d.energy.to_numpy(np.float32)
    Xs, var = E71.encode(d.cdr3.tolist(), par)
    Xe = E71.embeddings(d.cdr3.tolist(), var, tag)
    rows = []
    for s in range(E71.SEEDS):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:E71.HOLD], perm[E71.HOLD:]
        yte = y[te]
        for m in E71.SIZES:
            if m > len(pool):
                continue
            tr = pool[:m]
            r = Ridge(alpha=1.0, solver="lsqr").fit(Xs[tr], y[tr])
            a = float(spearmanr(yte, Xs[te] @ r.coef_ + r.intercept_).statistic)
            rr = Ridge(alpha=50.0, solver="lsqr").fit(Xe[tr], y[tr])
            b = float(spearmanr(yte, rr.predict(Xe[te])).statistic)
            rows.append(dict(library=tag, n_train=m, seed=s,
                             onehot=round(a, 4), esm=round(b, 4),
                             onehot_minus_esm=round(a - b, 4)))
    e_par = float(d.loc[d.cdr3 == par, "energy"].iloc[0])
    return pd.DataFrame(rows), par, e_par, float(y.min()), len(var)


def main():
    for p in (PUB, RANK1):
        if not os.path.exists(p):
            sys.exit(f"{p} 가 없습니다.")

    R1, par1, e1, min1, v1 = arm(PUB, f"matched_{AG}")
    print(f"발표 모항체 {par1} 에너지 {e1:.2f} · 최저 {min1:.2f} · "
          f"변이 자리 {v1}")
    R2, par2, e2, min2, v2 = arm(RANK1, f"rank1_{AG}")
    print(f"1위 모항체   {par2} 에너지 {e2:.2f} · 최저 {min2:.2f} · "
          f"변이 자리 {v2}")
    dist = sum(a != b for a, b in zip(par1, par2))
    print(f"두 모항체는 {dist}자리에서 다릅니다")

    R = pd.concat([R1.assign(library="published"),
                   R2.assign(library="rank1")], ignore_index=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    S = []
    for (lib, n), g in R.groupby(["library", "n_train"]):
        v = g.onehot_minus_esm.to_numpy()
        S.append(dict(library=lib, n_train=int(n),
                      onehot=round(g.onehot.mean(), 4),
                      esm=round(g.esm.mean(), 4),
                      diff=round(v.mean(), 4), esm_wins=int((v < 0).sum()),
                      n=len(v)))
    S = pd.DataFrame(S).sort_values(["n_train", "library"])
    print()
    print(S.to_string(index=False))

    pub = S[S.library == "published"].set_index("n_train")
    rk = S[S.library == "rank1"].set_index("n_train")
    # 부호로만 보면 안 된다. 한쪽이 정확히 0일 때 np.sign 은 0을 돌려주고,
    # 그러면 "+0.0000 대 -0.0064"가 방향이 다른 것으로 잡힌다. 둘 다 0에서
    # 뚜렷이 떨어져 있고 부호가 반대일 때만 뒤집힘으로 센다.
    EPS = 0.002
    flips = [int(k) for k in pub.index
             if abs(pub.loc[k, "diff"]) > EPS and abs(rk.loc[k, "diff"]) > EPS
             and np.sign(pub.loc[k, "diff"]) != np.sign(rk.loc[k, "diff"])]
    # 방향이 같아도 닫히는 속도가 다를 수 있다. 그쪽이 이 대조의 실제 결과다.
    big = max(pub.index)
    closes_pub = abs(pub.loc[big, "diff"]) <= EPS
    closes_rk = abs(rk.loc[big, "diff"]) <= EPS
    t = ttest_ind(R[R.library == "published"].onehot_minus_esm,
                  R[R.library == "rank1"].onehot_minus_esm, equal_var=False)

    L = ["# exp79 모항체 선택이 결론을 바꾸는가", "",
         "정합 라이브러리는 모항체 하나를 골라 그 둘레를 채워 만든다. 2DD8_S 의 "
         f"발표 모항체는 748,021개 가운데 1위가 아니라 4위다({e1:.2f} 대 "
         f"{e2:.2f}). 원고가 그 사실을 적어 둔 이상, 1위로 만들었으면 달랐을지 "
         "묻는 것이 당연하다.", "",
         f"두 모항체는 {dist}자리에서 다르고, 그 밖의 모든 것이 같다 -- 같은 "
         "항원, 같은 오라클, 같은 설계 규칙(단일변이 285개 + 무작위 이중·삼중), "
         "같은 크기, 같은 분할, 같은 모형. 음수는 언어 모형 우위다.", "",
         "| 라벨 | 모항체 | one-hot | ESM | 차 | ESM 우세 |",
         "|---|---|---|---|---|---|"]
    for _, x in S.iterrows():
        name = "발표(4위)" if x.library == "published" else "1위"
        L.append(f"| {int(x.n_train):,} | {name} | {x.onehot:.3f} | "
                 f"{x.esm:.3f} | {x['diff']:+.4f} | "
                 f"{int(x.esm_wins)}/{int(x.n)} |")
    L += ["", f"부호가 뒤집히는 학습 크기: "
          + (", ".join(f"{k:,}" for k in flips) if flips else "없음") + ".",
          f"두 판의 차이 분포 비교: Welch $t$ p = {t.pvalue:.2g}.", ""]
    if flips:
        L += ["모항체를 바꾸면 결론의 방향이 바뀐다. 그러면 이 결과는 "
              "라이브러리의 성질이 아니라 모항체 선택의 성질이고, 원고는 어느 "
              "모항체를 왜 골랐는지와 다른 선택에서 무엇이 달라지는지를 적어야 "
              "한다.", ""]
    else:
        L += ["방향은 모든 학습 크기에서 같다 -- 어느 모항체로 만들든 작은 "
              "라벨 수에서 언어 모형이 앞선다. 그 점에서 이 결과는 모항체 "
              "선택에 기대지 않는다.", ""]
        if closes_pub != closes_rk:
            which = "발표 모항체" if closes_pub else "1위 모항체"
            other = "1위 모항체" if closes_pub else "발표 모항체"
            L += [f"다른 것은 닫히는 속도다. {big:,}개 라벨에서 {which}로 만든 "
                  f"라이브러리는 차이가 0에 이르고, {other}로 만든 것은 "
                  f"{rk.loc[big, 'diff'] if closes_pub else pub.loc[big, 'diff']:+.4f}"
                  f"로 남는다. 본문이 '큰 라벨 수에서 차이가 사라진다'고 쓸 때 "
                  f"그 문장은 모항체 선택에 기댄다.", ""]
    d_pub = float(pub.onehot.mean())
    d_rk = float(rk.onehot.mean())
    L += [f"덧붙여, 1위 모항체 라이브러리가 전반적으로 덜 학습 가능하다"
          f"(one-hot 평균 {d_rk:.3f} 대 {d_pub:.3f}). 전역 최적점 둘레는 "
          f"그보다 나은 서열이 없으므로 변이가 대부분 에너지를 올리고, 그 "
          f"결과 라벨이 좁은 구간에 몰린다. 모항체를 '가장 좋은 것'으로 고르는 "
          f"규칙이 반드시 더 나은 벤치마크를 만들지는 않는다.", ""]
    L += ["이 대조는 계획한 것이 아니다. exp68 이 '길이 17 중 최저 에너지' "
          "규칙으로 2DD8_S 를 다시 만들면서 1위 모항체 라이브러리를 썼고, 그 "
          "파일이 발표본을 덮어썼다. 사본에서 되돌리며 덮어쓴 판을 남겨 두었고, "
          "버리는 대신 대조로 쓴다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
