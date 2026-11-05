"""실험 82 — 벤치마크가 방법 순위를 충실히 옮기는가, 일곱 모델로.

Absolut! 원 논문은 방법 정확도의 Spearman 상관 0.9 를 보고했다. 그 근거는 고전
분류기로 수행한 이진 분류 과제 하나였고, 그 뒤 벤치마크가 쓰인 범위 -- 표현
선택, 구조 선택, 탐색 설계 -- 보다 좁다. 이 논문도 지금까지 본문에서 두 표현만
비교했으므로 같은 비판을 받는다.

그래서 exp16 이 실측 표적 넷에 돌린 일곱 모델을 정합 격자 일곱 항원에 그대로
돌린다. 모델 정의는 model_zoo 에서 가져오므로 두 팔이 같은 하이퍼파라미터를
본다. 그러면 "합성 벤치마크에서의 모델 순위가 실측에서의 순위와 같은가"를 두
개가 아니라 일곱 개로 물을 수 있다.

재는 것은 개별 수치가 아니라 **순위 상관**이다. 그래서 씨드를 exp71 의 20 보다
낮춰도 된다 -- 순위는 평균값보다 빨리 안정된다.

    python src/exp82_zoo_transfer.py
    python src/exp82_zoo_transfer.py --seeds 3 --antigens 2DD8_S 1NCA_N
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from model_zoo import models                                    # noqa: E402
import exp71_repr_many_antigens as E71                          # noqa: E402

OUT = "results/exp82_zoo_transfer.csv"
MEASURED = "results/exp16_model_zoo.csv"
SIZES = [200, 500, 1000, 2000, 4000]
HOLD = 1500
AAIDX = E71.AAIDX


def tokens(seqs, var, ref):
    """자리별 토큰. 0 은 여백이므로 1부터 센다. exp8 과 같은 규약."""
    L = len(ref)
    proj = np.array([list(s.ljust(L)[:L]) for s in seqs])
    tok = np.zeros((len(seqs), len(var)), np.uint8)
    for c, j in enumerate(var):
        for i, ch in enumerate(proj[:, j]):
            tok[i, c] = AAIDX.get(ch, AAIDX["-"]) + 1
    return tok


def load(ag):
    p = f"data/absolut/matched_{ag}.parquet"
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p).dropna(subset=["energy"])
    seqs = d.cdr3.astype(str).tolist()
    y = d.energy.to_numpy(np.float32)
    ref = max(set(seqs), key=seqs.count) if False else seqs[0]
    # 모항체는 자리별 최빈 문자다. 첫 서열을 기준으로 삼으면 변이 자리를
    # 하나 놓칠 수 있다.
    L = len(seqs[0])
    arr = np.array([list(s) for s in seqs])
    ref = "".join(pd.Series(arr[:, j]).mode()[0] for j in range(L))
    Xs, var = E71.encode(seqs, ref)
    tok = tokens(seqs, var, ref)
    Xesm = E71.embeddings(seqs, var, f"matched_{ag}")
    return y, Xs, tok, Xesm, len(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--antigens", nargs="*", default=None)
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ags = a.antigens or sorted(
        x[len("matched_"):-len(".parquet")]
        for x in os.listdir("data/absolut")
        if x.startswith("matched_") and x.endswith(".parquet")
        and "_weak" not in x and "rank1parent" not in x)
    print(f"정합 격자 {len(ags)}종 · 모델 7 · 크기 {SIZES} · 씨드 {a.seeds}",
          flush=True)

    rows, t0 = [], time.time()
    for ag in ags:
        got = load(ag)
        if got is None:
            print(f"  건너뜀 {ag}: 라이브러리 없음", flush=True)
            continue
        y, Xs, tok, Xesm, nvar = got
        print(f"\n=== {ag}  n={len(y):,}  변이 자리 {nvar}  "
              f"ESM {'있음' if Xesm is not None else '없음'} ===", flush=True)
        for s in range(a.seeds):
            perm = np.random.default_rng(1000 + s).permutation(len(y))
            te, pool = perm[:HOLD], perm[HOLD:]
            yte = y[te]
            for m in SIZES:
                if m > len(pool):
                    continue
                t1 = time.time()
                preds = models(Xs, tok, Xesm, pool[:m], te, y[pool[:m]], s)
                for mn, pr in preds.items():
                    rows.append(dict(antigen=ag, model=mn, n_train=m, seed=s,
                                     spearman=round(
                                         float(spearmanr(yte, pr).statistic),
                                         4)))
                best = max(preds,
                           key=lambda k: spearmanr(yte, preds[k]).statistic)
                print(f"  n={m:5,} s{s}  최고={best:17s} "
                      f"[{time.time()-t1:.0f}s]", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(R)}행 · {(time.time()-t0)/60:.0f}분)")
    report(R)


def report(R):
    """격자에서의 모델 순위가 실측에서의 순위와 같은가."""
    if not os.path.exists(MEASURED):
        print(f"{MEASURED} 가 없어 이전성은 재지 않았습니다.")
        return
    M = pd.read_csv(MEASURED)
    lat = R.pivot_table(index="model", columns="n_train", values="spearman")
    mea = M.pivot_table(index="model", columns="n_train", values="spearman")
    common_n = sorted(set(lat.columns) & set(mea.columns))
    common_m = sorted(set(lat.index) & set(mea.index))
    print(f"\n=== 모델 순위의 이전성 (공통 모델 {len(common_m)}개) ===")
    print(f"{'학습 크기':>8}  {'격자 최고':<18} {'실측 최고':<18} 순위 상관")
    for n in common_n:
        a_ = lat.loc[common_m, n].dropna()
        b_ = mea.loc[common_m, n].dropna()
        idx = sorted(set(a_.index) & set(b_.index))
        if len(idx) < 3:
            continue
        rho = spearmanr(a_[idx], b_[idx]).statistic
        print(f"{n:>8,}  {a_.idxmax():<18} {b_.idxmax():<18} "
              f"{rho:+.3f}  ({len(idx)}개 모델)")
    print("\n격자 평균 Spearman")
    print(lat.round(3).to_string())


if __name__ == "__main__":
    main()
