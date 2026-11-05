"""실험 70 — 정합 '전' 팔을 정합 '후'와 같은 규약으로 다시 잰다.

논문의 주장은 A(정합 전) != B(정합 후)였다가 조건을 맞추니 같아졌다는 것이다.
그런데 두 팔을 같은 자로 재지 않았다. 정합 후는 20회 분할에 부트스트랩 신뢰구간과
Holm 보정을 썼고(exp21), 정합 전은 3~5개 시드의 점추정에 신뢰구간도 검정도 없다.

그러면 "차이가 사라졌다"가 성립하지 않는다. 가장 단순한 대안 -- 정합 전 추정이
애초에 잡음이었고 설명할 역전이 없었다 -- 가 배제되지 않기 때문이다. 심사자가
이 자리를 짚는 것은 당연하다.

여기서는 exp21과 같은 것을 바꾼다. 분할 수, 학습 크기, 모델, 검정, 씨앗까지
같게 두고 라이브러리만 공개(비정합) 쪽으로 바꾼다.

주의해서 읽어야 할 것이 하나 있다. 두 팔은 인코더가 다르고 다를 수밖에 없다 --
모항체도 공통 길이도 없는 라이브러리에는 정렬 one-hot이 존재하지 않는다. 그래서
이 실험은 "정합 전 추정이 잡음이었는가"에 답하지, "인코더를 통제했는가"에
답하지 않는다. 후자는 이 설계로 답할 수 없고, 본문에 그렇게 적었다.

    python src/exp70_unmatched_seeds.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, ttest_1samp, wilcoxon
from sklearn.linear_model import Ridge

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# exp21과 같은 값. 하나라도 다르면 두 팔을 나란히 놓을 수 없다.
SIZES = [200, 500, 1000, 2000, 4000]
SEEDS = 20
HOLD = 1500
AG = "2DD8_S"
CACHE = "data/esm"
OUT = "results/exp70_unmatched_seeds.csv"
OUT2 = "results/exp70_per_size_tests.csv"
REPORT = "results/exp70_unmatched_seeds.md"


def load_unmatched():
    """공개 라이브러리에서 ESM 캐시와 같은 표본을 되살린다.

    캐시(lattice_2DD8_S_perpos.npz)는 exp9b가 만들 때 쓴 표본 순서를 그대로
    담고 있다. 같은 씨앗으로 같은 부분집합을 다시 뽑아야 행이 맞는다 --
    어긋나면 라벨과 특징이 서로 다른 서열을 가리키게 되고, 그때 나오는
    상관계수는 조용히 0 근처가 된다.
    """
    path = f"{CACHE}/lattice_{AG}_perpos.npz"
    if not os.path.exists(path):
        sys.exit(f"ESM 캐시가 없습니다: {path}")
    Xesm = np.load(path)["X"]
    lib = pd.read_parquet("data/absolut/library.parquet")["cdr3"]
    Xs = sparse.load_npz("data/absolut/library_features.npz").tocsr()
    e = np.load("data/absolut/energies.npz")[AG]
    n = len(Xesm)
    # exp9b가 쓴 씨앗은 11이다(exp9b_esm_perpos.py의 load_lattice 호출).
    # 처음에 0으로 짐작했는데, 틀렸으면 라벨과 임베딩이 서로 다른 서열을
    # 가리킨 채 조용히 0 근처 상관을 냈을 것이다. 씨앗은 짐작하지 않는다.
    sub = np.sort(np.random.default_rng(11).choice(len(lib), size=n,
                                                   replace=False))
    return e[sub].astype(np.float32), Xs[sub], Xesm, lib.values[sub]


def main():
    y, Xs, Xesm, seqs = load_unmatched()
    print(f"공개 라이브러리 표본 {len(y):,}개 · one-hot {Xs.shape} · "
          f"ESM {Xesm.shape}")
    if len(Xesm) != len(y):
        sys.exit(f"행 수가 다릅니다: ESM {len(Xesm)} 대 라벨 {len(y)}")
    # 라벨과 특징이 같은 서열을 가리키는지 한 번 확인한다. 표본 규약이
    # 어긋나면 여기서 잡힌다.
    L = np.array([len(s) for s in seqs])
    print(f"  길이 {L.min()}~{L.max()} · 에너지 {y.min():.1f}~{y.max():.1f} "
          f"(SD {y.std():.2f})")

    # 정렬 검사. 표본 씨앗이 어긋나면 라벨과 임베딩이 다른 서열을 가리키고,
    # 그때 나오는 상관은 0 근처가 된다 -- 실행은 끝까지 되고 결과만 틀린다.
    # 그래서 작은 적합을 한 번 돌려 두 특징 모두 신호를 내는지 먼저 본다.
    chk = np.random.default_rng(0).permutation(len(y))
    tr, te = chk[:4000], chk[4000:6000]
    s_oh = float(spearmanr(
        y[te], Xs[te] @ Ridge(alpha=1.0, solver="lsqr")
        .fit(Xs[tr], y[tr]).coef_).statistic)
    s_es = float(spearmanr(
        y[te], Ridge(alpha=50.0, solver="lsqr")
        .fit(Xesm[tr], y[tr]).predict(Xesm[te])).statistic)
    print(f"  정렬 검사 Spearman  one-hot {s_oh:.3f} · ESM {s_es:.3f}")
    if min(abs(s_oh), abs(s_es)) < 0.10:
        sys.exit("정렬 검사 실패 — 둘 중 하나가 신호를 내지 못합니다. "
                 "표본 씨앗이 ESM 캐시와 맞지 않을 수 있습니다.")

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
            p_oh = Xs[te] @ r.coef_ + r.intercept_
            re = Ridge(alpha=50.0, solver="lsqr").fit(Xesm[tr], y[tr])
            p_es = re.predict(Xesm[te])
            a = float(spearmanr(yte, p_oh).statistic)
            b = float(spearmanr(yte, p_es).statistic)
            rows.append(dict(antigen=AG, arm="unmatched", n_train=m, seed=s,
                             onehot=round(a, 4), esm=round(b, 4),
                             onehot_minus_esm=round(a - b, 4)))
        print(f"  seed {s:2d} done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False, encoding="utf-8")

    out = []
    for m, g in R.groupby("n_train"):
        d = g.onehot_minus_esm.to_numpy()
        t = ttest_1samp(d, 0.0)
        try:
            w = float(wilcoxon(d).pvalue)
        except ValueError:
            w = float("nan")
        boot = np.array([np.random.default_rng(7 + i)
                         .choice(d, len(d), True).mean() for i in range(10000)])
        out.append(dict(n_train=m, n_seeds=len(d),
                        mean_diff=round(d.mean(), 4),
                        sd=round(d.std(ddof=1), 4),
                        ci_lo=round(np.percentile(boot, 2.5), 4),
                        ci_hi=round(np.percentile(boot, 97.5), 4),
                        onehot_wins=int((d > 0).sum()),
                        p_ttest=float(t.pvalue), p_wilcoxon=w))
    S = pd.DataFrame(out)
    S.to_csv(OUT2, index=False, encoding="utf-8")
    print()
    print(S.to_string(index=False))

    d = R.onehot_minus_esm.to_numpy()
    boot = np.array([np.random.default_rng(11 + i).choice(d, len(d), True).mean()
                     for i in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    L = ["# exp70 정합 전 팔 — 정합 후와 같은 규약으로", "",
         "논문은 정합 전과 후를 비교하는데 두 팔을 같은 자로 재지 않았다. "
         f"정합 후는 {SEEDS}회 분할·부트스트랩·Holm 보정, 정합 전은 3~5개 "
         "시드의 점추정이었다. 여기서는 분할 수·학습 크기·모델·검정·씨앗을 "
         "exp21과 같게 두고 라이브러리만 공개 쪽으로 바꾼다.", "",
         "| 통계 | 값 |", "|---|---|",
         f"| 분할 | {SEEDS} |",
         f"| 셀 (분할 x 크기) | {len(d)} |",
         f"| one-hot 빼기 ESM 평균 | {d.mean():+.4f} |",
         f"| 95% 신뢰구간 | {lo:+.4f} ~ {hi:+.4f} |",
         f"| one-hot이 이긴 셀 | {int((d > 0).sum())} / {len(d)} |", "",
         "양수는 one-hot 우위다. 구간이 0을 포함하지 않고 양수 쪽이면 정합 전의 "
         "one-hot 우위가 시드 수를 늘려도 남는다는 뜻이고, 그때 비로소 정합 후의 "
         "소멸을 '차이가 사라졌다'로 읽을 수 있다. 구간이 0을 걸치면 애초에 "
         "설명할 차이가 없었던 것이므로 논문의 첫 번째 대비를 다시 써야 한다.", "",
         "두 팔은 인코더가 다르고 다를 수밖에 없다 -- 모항체도 공통 길이도 없는 "
         "라이브러리에는 정렬 one-hot이 존재하지 않는다. 이 실험은 '정합 전 "
         "추정이 잡음이었는가'에 답하지 '인코더를 통제했는가'에 답하지 않는다.",
         ""]
    io.open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print()
    print("\n".join(L))
    print(f"wrote {OUT}\nwrote {OUT2}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
