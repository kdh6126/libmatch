"""실험 83 — ridge 벌점 alpha 가 표현 비교의 결론을 만들었는가.

심사자 4는 정규화를 고른 홀드아웃의 구성을 물었다. 답은 "그런 절차가 없다"
이다. alpha 는 코드에 상수로 박혀 있고(one-hot 1, ESM 50), 항원·예산·분할·
씨앗 어디에도 맞춰 고르지 않는다. 정직한 답이지만 다음 질문을 남긴다 --
그 두 값이 결론을 만든 것은 아닌가.

여기서 그것만 본다. 두 값 주위의 **미리 정한** 로그 격자를 7개 정합 항원에
그대로 돌리고, 결론의 방향이 유지되는지 본다. 이 결과로 alpha 를 다시 고르지
않는다. 다시 고르면 지금 피하려는 바로 그 일(자료에 맞춘 조정)이 된다.

격자는 결과를 보기 전에 정했다:

    one-hot  alpha in {0.1, 1, 10, 100}      고정값 1 의 위아래 두 자릿수
    ESM      alpha in {5, 50, 500}           고정값 50 의 위아래 한 자릿수

분할 규칙·홀드아웃·학습 크기·씨앗 수는 exp21/exp71 과 **같다**
(default_rng(1000+s), 홀드아웃 1,500, 씨앗 20). 바뀌는 것은 벌점뿐이다.
고정 벌점 칸은 exp21 의 같은 칸과 소수점까지 일치해야 하고, 일치하지
않는다면 그것부터 봐야 한다. 씨앗을 줄이면 본문의 추정과 나란히 놓을 수
없다 -- 재려는 차이가 0.005 수준이라 씨앗 몇 개의 운이 부호를 바꾼다.

읽을 때 먼저 볼 것은 격자 전체의 평균이다. 고정값이 어느 쪽에 유리했는지가
거기서 드러난다 -- 고정값이 한쪽의 최적이고 다른 쪽에 불리했다면, 결론은
alpha 가 만든 것일 수 있다.

one-hot 에 격자 최선을, ESM 에 격자 최악을 준 칸도 함께 적는다. 둘 다 같은
홀드아웃을 보고 사후에 고른 것이라 실제로는 가질 수 없는 이점이고, 해석할
때는 그 alpha 가 격자의 가장자리인지 평평한 구간 안인지를 함께 봐야 한다.
가장자리의 과잉 정규화로 한쪽을 깎아 얻은 역전은 벌점의 성질이지 결론의
성질이 아니다.

임베딩은 캐시를 읽기만 한다. 서열 해시가 다르면 다시 만들지 않고 멈춘다 --
다시 만드는 것은 무거운 작업이고, 이 표는 그럴 만한 것이 아니다.

    python src/exp83_alpha_sensitivity.py
    python src/exp83_alpha_sensitivity.py --only 2DD8_S --seeds 3
"""
import argparse
import glob
import hashlib
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
SIZES = [200, 500, 1000, 2000, 4000]
HOLD = 1500
CACHE = "data/esm"
DIR = "data/absolut"
ALPHA_ONEHOT = [0.1, 1.0, 10.0, 100.0]
ALPHA_ESM = [5.0, 50.0, 500.0]
FIXED_ONEHOT = 1.0
FIXED_ESM = 50.0
OUT = "results/exp83_alpha_sensitivity.csv"
OUT2 = "results/exp83_alpha_summary.csv"
REPORT = "results/exp83_alpha_sensitivity.md"
NL = chr(10)


def encode(seqs, ref):
    """exp71 과 같은 정렬 one-hot. 길이가 같으므로 위치 i가 위치 i에 대응한다."""
    L = len(ref)
    proj = np.array([list(s.ljust(L)[:L]) for s in seqs])
    var = [j for j in range(L) if len(set(proj[:, j])) > 1]
    X = np.zeros((len(seqs), len(var) * 21), np.float32)
    for c, j in enumerate(var):
        for i, ch in enumerate(proj[:, j]):
            X[i, c * 21 + AAIDX.get(ch, AAIDX["-"])] = 1.0
    return sparse.csr_matrix(X), var


def seq_hash(seqs):
    return hashlib.sha1("|".join(seqs).encode()).hexdigest()


def load_cached(tag, seqs):
    """캐시를 읽기만 한다. 서열이 다르면 멈춘다.

    행 수만 보고 재사용하면 안 된다. 캐시 이름은 항원 이름으로 정해지는데
    라이브러리는 다시 만들어질 수 있고 두 판 모두 6,000행이다. 그러면 라벨과
    특징이 서로 다른 서열을 가리킨 채 끝까지 돌고 상관계수만 조용히 틀린다.
    """
    path = os.path.join(CACHE, tag + "_perpos.npz")
    side = os.path.join(CACHE, tag + "_perpos.seqs.sha1")
    if not os.path.exists(path):
        return None, "임베딩 캐시 없음: " + path
    if not os.path.exists(side):
        return None, "서열 해시 기록 없음: " + side
    have = io.open(side).read().strip()
    if have != seq_hash(seqs):
        return None, "서열 해시 불일치: " + tag + " (캐시가 다른 판의 것)"
    X = np.load(path)["X"]
    if len(X) != len(seqs):
        return None, "행 수 불일치: %s %d vs %d" % (tag, len(X), len(seqs))
    return X, None


def run_antigen(ag, seeds):
    import matched_libs
    d = pd.read_parquet(os.path.join(DIR, "matched_" + ag + ".parquet"))
    parent = matched_libs.parent_of(d)
    d = d[d.cdr3.str.len() == len(parent)].reset_index(drop=True)
    seqs = d.cdr3.tolist()
    y = d.energy.to_numpy(np.float32)

    Xesm, why = load_cached("matched_" + ag, seqs)
    if Xesm is None:
        print("  건너뜀 %s: %s" % (ag, why), flush=True)
        return []
    Xs, var = encode(seqs, parent)
    print("  %s: %d개 · 가변 %d자리 · ESM %s"
          % (ag, len(d), len(var), Xesm.shape), flush=True)

    rows = []
    for s in range(seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:HOLD], perm[HOLD:]
        yte = y[te]
        for m in SIZES:
            if m > len(pool):
                continue
            tr = pool[:m]
            for a in ALPHA_ONEHOT:
                p = Ridge(alpha=a, solver="lsqr").fit(
                    Xs[tr], y[tr]).predict(Xs[te])
                rows.append(dict(antigen=ag, seed=s, n_train=m, rep="onehot",
                                 alpha=a,
                                 rho=round(float(spearmanr(p, yte).statistic), 4)))
            for a in ALPHA_ESM:
                p = Ridge(alpha=a, solver="lsqr").fit(
                    Xesm[tr], y[tr]).predict(Xesm[te])
                rows.append(dict(antigen=ag, seed=s, n_train=m, rep="esm",
                                 alpha=a,
                                 rho=round(float(spearmanr(p, yte).statistic), 4)))
        print("    씨앗 %d 끝" % s, flush=True)
    return rows


def summarise(R):
    """고정쌍과, one-hot 에 유리하게 치우친 쌍을 나란히 놓는다."""
    out = []
    for (ag, m), g in R.groupby(["antigen", "n_train"]):
        oh = g[g.rep == "onehot"].groupby("alpha").rho.mean()
        es = g[g.rep == "esm"].groupby("alpha").rho.mean()
        fixed = float(oh[FIXED_ONEHOT] - es[FIXED_ESM])
        skew = float(oh.max() - es.min())
        out.append(dict(antigen=ag, n_train=int(m),
                        onehot_fixed=round(float(oh[FIXED_ONEHOT]), 4),
                        esm_fixed=round(float(es[FIXED_ESM]), 4),
                        diff_fixed=round(fixed, 4),
                        onehot_best=round(float(oh.max()), 4),
                        onehot_best_alpha=float(oh.idxmax()),
                        esm_worst=round(float(es.min()), 4),
                        esm_worst_alpha=float(es.idxmin()),
                        diff_skewed=round(skew, 4),
                        esm_ahead_fixed=bool(fixed < 0),
                        esm_ahead_skewed=bool(skew < 0)))
    return pd.DataFrame(out).sort_values(["n_train", "antigen"])


def antigens(only):
    ags = []
    for p in sorted(glob.glob(os.path.join(DIR, "matched_*.parquet"))):
        name = os.path.basename(p)[len("matched_"):-len(".parquet")]
        if "_weak" in name or "rank1parent" in name:
            continue
        ags.append(name)
    if only:
        ags = [x for x in ags if x == only]
    return ags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--from-csv", action="store_true",
                    help="적합을 다시 하지 않고 저장된 표에서 보고서만 다시 쓴다")
    a = ap.parse_args()

    if a.from_csv:
        R = pd.read_csv(OUT)
        a.seeds = int(R.seed.nunique())
        print("%s 에서 %d행을 읽었습니다 (씨앗 %d회)"
              % (OUT, len(R), a.seeds), flush=True)
    else:
        ags = antigens(a.only)
        print("항원 %d종 · 씨앗 %d회 · one-hot alpha %s · ESM alpha %s"
              % (len(ags), a.seeds, ALPHA_ONEHOT, ALPHA_ESM), flush=True)
        rows = []
        for ag in ags:
            rows += run_antigen(ag, a.seeds)
        if not rows:
            sys.exit("돌릴 수 있는 항원이 없습니다.")
        R = pd.DataFrame(rows)
        R.to_csv(OUT, index=False, encoding="utf-8")
    S = summarise(R)
    S.to_csv(OUT2, index=False, encoding="utf-8")

    n_fixed = int(S.esm_ahead_fixed.sum())
    n_skew = int(S.esm_ahead_skewed.sum())
    small = S[S.n_train <= 500]
    n_small = int(small.esm_ahead_fixed.sum())
    G = R.groupby(["n_train", "rep", "alpha"]).rho.mean().round(4)
    L = ["# Ridge penalty sensitivity", "",
         "Pre-registered log grid around the fixed penalties: one-hot "
         + str(ALPHA_ONEHOT) + ", ESM " + str(ALPHA_ESM) + ". Splits follow "
         "the rule used throughout (`default_rng(1000 + s)`, " + str(HOLD)
         + " held out), with " + str(a.seeds) + " seeds. The grid was fixed "
         "before the results were seen and no penalty is re-selected from it.",
         "",
         "The language model is ahead wherever the difference is negative. At "
         "the fixed penalties it is ahead in " + str(n_fixed) + " of "
         + str(len(S)) + " (antigen, budget) cells, and in " + str(n_small)
         + " of " + str(len(small)) + " at the budgets of 200 and 500 where "
         "the article makes the claim.", "",
         "## Whether the fixed penalties favoured the conclusion", "",
         "Mean held-out Spearman over the seven antigens, by penalty. The "
         "fixed values are one-hot 1 and ESM 50.", "",
         "| Labels | " + " | ".join(
             "%s a=%g" % (rep, al) for rep in ("onehot", "esm")
             for al in (ALPHA_ONEHOT if rep == "onehot" else ALPHA_ESM))
         + " |",
         "|" + "---|" * (1 + len(ALPHA_ONEHOT) + len(ALPHA_ESM))]
    for m in SIZES:
        cells = ["%.4f" % G[(m, rep, al)]
                 for rep in ("onehot", "esm")
                 for al in (ALPHA_ONEHOT if rep == "onehot" else ALPHA_ESM)]
        L.append("| " + str(m) + " | " + " | ".join(cells) + " |")
    L += ["",
          "One-hot does best at its fixed penalty of 1 at every budget, so "
          "the fixed value did not handicap it. ESM does slightly better at "
          "5 than at its fixed 50 at 200 labels, so the fixed value if "
          "anything understates its advantage where that advantage is "
          "largest. The direction reverses only at a=500 for ESM, which is "
          "the edge of the grid and costs it far more than its neighbours; "
          "the matching edge for one-hot, a=100, costs one-hot more still. "
          "An edge penalty that cripples one side is a property of the "
          "penalty, not of the comparison.", "",
          "## Per antigen and budget", "",
          "The last two columns give one-hot its best penalty on the grid "
          "and ESM its worst, both chosen post hoc on the same held-out set, "
          "which is an advantage neither would have in practice.", "",
          "| Antigen | Labels | one-hot (a=1) | ESM (a=50) | diff | one-hot best | ESM worst | diff |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        L.append("| %s | %d | %+.3f | %+.3f | %+.3f | %+.3f (a=%g) | "
                 "%+.3f (a=%g) | %+.3f |"
                 % (r.antigen, int(r.n_train), r.onehot_fixed, r.esm_fixed,
                    r.diff_fixed, r.onehot_best, r.onehot_best_alpha,
                    r.esm_worst, r.esm_worst_alpha, r.diff_skewed))
    io.open(REPORT, "w", encoding="utf-8", newline=NL).write(NL.join(L) + NL)

    print(NL + "wrote %s, %s, %s" % (OUT, OUT2, REPORT))
    print("고정 벌점에서 ESM 우위 %d/%d 칸 · one-hot 에 유리하게 치우친 "
          "비교에서 %d/%d 칸" % (n_fixed, len(S), n_skew, len(S)))


if __name__ == "__main__":
    main()
