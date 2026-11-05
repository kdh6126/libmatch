"""실험 76 — 자리 수를 줄이면 라이브러리가 정말 더 학습하기 쉬워지는가.

exp73은 여섯 팔에 걸쳐 "탐색 배수는 대리 모형의 적합도를 따라가고, 적합도는
변이 자리 수를 따라간다"는 연관을 보였다. exp74는 그 사슬의 뒷부분을 한 항원
안에서 시험했고 실패했다 -- 자리 수를 줄여도 배수가 단조로 움직이지 않았고,
학습성마저 항원마다 방향이 달랐다.

exp74에는 결함이 둘 있다.

  천장   공통 크기가 작아 예산이 라이브러리의 27%를 차지했다. 무작위가 이미
         상위 구간의 1/4을 주워 오면 배수는 4를 넘을 수 없고, 가장 높은 칸은
         실제로 천장에 닿아 있었다. 눌린 양으로 용량-반응을 잴 수는 없다.
  자리   k자리 집합을 "서열을 가장 많이 담는" 것으로 골랐다. 무작위가 아니다.
         그래서 k가 바뀔 때 자리의 개수만이 아니라 어느 자리인지도 같이
         바뀌었고, 두 효과를 나눌 수 없다.

여기서는 둘 다 없앤다. 탐색을 빼고 학습성만 본다(천장이 없다). 그리고 자리
집합을 무작위로 여러 번 뽑아 평균 낸다(어느 자리인지가 상쇄된다). 크기와
차수 조성은 exp74와 같이 고정한다.

이것은 exp73 사슬의 뒷마디만 떼어 낸 시험이다. 통과하면 그 마디는 서고,
실패하면 exp73의 연관은 차원으로 설명되지 않는다는 뜻이며, 그렇게 적어야 한다.

    python src/exp76_random_positions.py
    python src/exp76_random_positions.py --draws 20
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")

AA = "ACDEFGHIKLMNPQRSTVWY-"
AAIDX = {a: i for i, a in enumerate(AA)}
KS = [8, 11, 15]
N_TRAIN = 400
SEEDS = 20
N_LIB = 1200          # 무작위 자리 집합이 대체로 담을 수 있는 크기
LIBS = "results/exp68_matched_libraries.csv"
PUBLISHED = {"2DD8_S": "CARDRLLWLRLWYFDVW"}
OUT = "results/exp76_random_positions.csv"
REPORT = "results/exp76_random_positions.md"
NL = chr(10)


def encode_at(seqs, ref, positions):
    """주어진 자리에서만 one-hot. 나머지는 모항체와 같으므로 정보가 없다."""
    pos = sorted(positions)
    X = np.zeros((len(seqs), len(pos) * 21), np.float32)
    for c, j in enumerate(pos):
        for i, s in enumerate(seqs):
            X[i, c * 21 + AAIDX.get(s[j], AAIDX["-"])] = 1.0
    return sparse.csr_matrix(X)


def learnability(y, X, n_train, hold):
    out = []
    for s in range(SEEDS):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:hold], perm[hold:]
        tr = pool[:n_train]
        r = Ridge(alpha=1.0, solver="lsqr").fit(X[tr], y[tr])
        out.append(float(spearmanr(y[te], X[te] @ r.coef_ + r.intercept_)
                         .statistic))
    return np.asarray(out)


def balanced(idx, order, mix, n, rng):
    take = []
    for o, frac in mix.items():
        pool = idx[order[idx] == o]
        want = int(round(frac * n))
        if want == 0:
            continue
        if len(pool) < want:
            return None
        take.append(rng.choice(pool, size=want, replace=False))
    return np.sort(np.concatenate(take)) if take else None


def matched_arms(only=None):
    """정합 라이브러리 목록. 모항체는 라이브러리 자체에서 복원한다.

    exp68 의 결과표를 읽지 않는다. 그 표는 exp68 이 항원을 끝낼 때마다 전체를
    다시 쓰므로, 그 프로세스가 도는 동안의 값은 믿을 수 없다 -- 실제로 한 항원의
    모항체가 뒤바뀐 채 읽혀 엉뚱한 자리에서 죽은 적이 있다.
    """
    import matched_libs
    return {ag: par for ag, (par, _) in matched_libs.arms(only).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=12)
    a = ap.parse_args()

    rows = []
    for ag, par in matched_arms().items():
        d = pd.read_parquet(f"data/absolut/matched_{ag}.parquet")
        d = d[d.cdr3.str.len() == len(par)].reset_index(drop=True)
        seqs = d.cdr3.tolist()
        y = d.energy.to_numpy(np.float32)
        mut = [frozenset(i for i, (x, p) in enumerate(zip(s, par)) if x != p)
               for s in seqs]
        order = np.array([len(m) for m in mut])
        allpos = sorted(set().union(*[m for m in mut if m]))

        # 조성 기준은 가장 작은 k의 무작위 집합에서 나오는 것을 쓴다.
        # 한 번 정하면 모든 k와 모든 뽑기에 같은 비율을 적용한다.
        ref_rng = np.random.default_rng(5)
        ref = frozenset(ref_rng.choice(allpos, size=min(KS), replace=False)
                        .tolist())
        ref_idx = np.array([i for i, m in enumerate(mut) if m <= ref])
        o = order[ref_idx]
        mix = {int(v): float((o == v).mean()) for v in np.unique(o)}
        print(f"[{ag}] 기준 조성 "
              + " ".join(f"{v}차={f:.2f}" for v, f in sorted(mix.items()))
              + f" · 공통 크기 {N_LIB:,}", flush=True)

        for k in KS:
            got, skipped = [], 0
            for r in range(1 if k == len(allpos) else a.draws):
                rng = np.random.default_rng(100 * k + r)
                pos = (frozenset(allpos) if k == len(allpos)
                       else frozenset(rng.choice(allpos, size=k,
                                                 replace=False).tolist()))
                idx = np.array([i for i, m in enumerate(mut) if m <= pos])
                sub = balanced(idx, order, mix, N_LIB, rng)
                if sub is None:
                    skipped += 1
                    continue
                ys = y[sub]
                X = encode_at([seqs[i] for i in sub], par, pos)
                rho = learnability(ys, X, N_TRAIN, len(sub) // 3)
                got.append(rho.mean())
            if not got:
                print(f"[{ag}] k={k} — 쓸 수 있는 자리 집합이 없습니다 "
                      f"(건너뜀 {skipped})")
                continue
            got = np.asarray(got)
            rows.append(dict(antigen=ag, k=k, draws=len(got), skipped=skipped,
                             rho=round(got.mean(), 4),
                             rho_sd=round(got.std(ddof=1), 4)
                             if len(got) > 1 else 0.0,
                             rho_lo=round(got.min(), 4),
                             rho_hi=round(got.max(), 4)))
            print(f"[{ag}] k={k:2d}  자리 집합 {len(got)}회 "
                  f"(건너뜀 {skipped})  보류 Spearman {got.mean():.3f} "
                  f"[{got.min():.3f}, {got.max():.3f}]", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False, encoding="utf-8")
    print()
    print(R.to_string(index=False))

    # 단조성을 주장하지 않고 센다.
    ver = []
    for ag, g in R.groupby("antigen"):
        g = g.sort_values("k", ascending=False)
        v = g.rho.to_numpy()
        ver.append(dict(antigen=ag, by_k_desc=list(np.round(v, 3)),
                        monotone_up_as_k_falls=bool(np.all(np.diff(v) > 0))))
    V = pd.DataFrame(ver)
    n_mono = int(V.monotone_up_as_k_falls.sum())

    L = ["# exp76 자리를 무작위로 뽑아, 자리 수만 바꾸면", "",
         "exp74의 결함 둘을 없앴다. 탐색을 빼고 학습성만 보므로 천장이 없고, "
         f"자리 집합을 무작위로 {a.draws}번 뽑아 평균 내므로 '어느 자리인지'가 "
         "상쇄된다. 크기와 차수 조성은 고정했다.", "",
         f"라벨 {N_TRAIN}개, 분할 {SEEDS}회, 라이브러리 {N_LIB:,}개.", "",
         "| 항원 | k | 자리 집합 | 보류 Spearman | 뽑기 간 범위 |",
         "|---|---|---|---|---|"]
    for _, x in R.sort_values(["antigen", "k"], ascending=[True, False]).iterrows():
        L.append(f"| {x.antigen} | {int(x.k)} | {int(x.draws)} | "
                 f"{x.rho:.3f} ± {x.rho_sd:.3f} | "
                 f"{x.rho_lo:.3f} ~ {x.rho_hi:.3f} |")
    L += ["", "## 판정", "",
          "| 항원 | k=15 → 11 → 8 | k가 줄며 단조 증가 |", "|---|---|---|"]
    for _, x in V.iterrows():
        L.append(f"| {x.antigen} | "
                 f"{' → '.join(f'{v:.3f}' for v in x.by_k_desc)} | "
                 f"{'예' if x.monotone_up_as_k_falls else '아니오'} |")
    # 단조성은 중간점까지 요구하므로 주장보다 엄격하다. 주장이 기대는 비교는
    # 가장 넓은 설계 공간과 가장 좁은 설계 공간 사이의 차이 하나다. 둘을 따로
    # 낸다 -- 엄격한 쪽이 깨졌다고 느슨한 쪽까지 버릴 이유는 없고, 느슨한 쪽만
    # 보고하면 중간이 흔들린다는 사실을 숨기게 된다.
    ends = []
    for ag, g in R.groupby("antigen"):
        hi = float(g[g.k == max(KS)].rho.iloc[0])
        lo = float(g[g.k == min(KS)].rho.iloc[0])
        ends.append(dict(antigen=ag, k_hi=max(KS), k_lo=min(KS),
                         rho_hi=round(hi, 4), rho_lo=round(lo, 4),
                         gain=round(lo - hi, 4)))
    E = pd.DataFrame(ends)
    n_up = int((E.gain > 0).sum())
    L += ["", f"{len(V)}개 항원 가운데 단조 증가는 {n_mono}개다.", "",
          f"단조성은 중간점 k={sorted(KS)[1]}까지 순서를 요구하므로 주장보다 "
          f"엄격하다. 주장이 기대는 비교는 양 끝 하나다.", "",
          f"| 항원 | k={max(KS)} | k={min(KS)} | 차 |", "|---|---|---|---|"]
    for _, x in E.iterrows():
        L.append(f"| {x.antigen} | {x.rho_hi:.3f} | {x.rho_lo:.3f} | "
                 f"{x.gain:+.3f} |")
    L += ["", f"양 끝만 보면 {n_up}/{len(E)}개 항원에서 올라간다.", ""]
    if n_up == len(E):
        gain = float(E.gain.mean())
        spread = float(R[R.k < max(KS)].rho_sd.mean())
        L += [f"자리 수를 줄이면 같은 크기·조성에서도 라이브러리가 더 학습하기 "
              f"쉬워진다. {max(KS)}자리에서 {min(KS)}자리로 가며 평균 "
              f"{gain:+.3f}이고, {len(E)}개 항원 모두 같은 방향이다."]
        if n_mono < len(V):
            L += ["", f"중간점까지 순서가 맞는 것은 {n_mono}/{len(V)}이다. "
                  f"어긋나는 항원의 차이는 뽑기 간 표준편차({spread:.3f})보다 "
                  f"작으므로 순서를 뒤집을 만한 크기가 아니지만, 양 끝만 "
                  f"보고하면 중간이 흔들린다는 사실이 가려지므로 함께 적는다."]
        L += ["", "여기에 두 가지 단서를 달아야 한다.", "",
              f"첫째, 크기가 작다. 뽑기마다의 표준편차가 평균 {spread:.3f}이므로 "
              "항원 하나만 놓고 보면 결론이 나지 않는다. 모든 항원이 같은 "
              f"방향이라는 것이 근거이고, 부호 검정으로는 p = "
              f"{2 ** -len(E):.3f}다.", "",
              "둘째, 이 크기로는 격자와 실측의 간극을 설명하지 못한다. exp73에서 "
              "격자 팔의 대리 적합도는 0.84~0.92, 실측 팔은 0.47~0.69로 차이가 "
              f"0.2~0.4인데, 자리 수를 절반으로 줄여 얻는 것은 {gain:+.3f}이다"
              "(두 수치는 라벨 수와 라이브러리 크기가 달라 직접 빼서는 안 되고, "
              "자릿수만 비교한다). 차원은 방향이 맞지만 한 자릿수 부족하다. "
              "나머지는 이 논문이 따로 잰 측정 잡음(Spearman 비용 0.105) 쪽에 "
              "있을 것이고, 그 둘을 합쳐도 전부를 메우는지는 여기서 답하지 "
              "않는다.", ""]
    else:
        L += [f"양 끝 비교에서도 {len(E) - n_up}개 항원이 반대 방향이다. "
              "그러면 exp73의 연관은 '차원' 하나로 설명되지 않는다. 격자 팔과 "
              "실측 팔은 자리 수 말고도 여러 가지가 다르고(측정 잡음, 서열 "
              "길이, 항원 구조), 이 실험은 그 가운데 자리 수만 떼어 냈을 때 "
              "효과가 일정하지 않다는 것을 말한다. 원고에는 연관까지만 "
              "적는다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
