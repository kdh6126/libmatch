"""실험 71 — 표현 비교를 정합 항원 여러 종으로 넓힌다.

논문에서 가장 약한 자리는 여기다. "정합하면 one-hot 우위가 사라지고 언어 모형이
앞선다"는 결과가 항원 하나(2DD8_S)에 기대고 있고, 두 번째로 만든 라이브러리는
일관성 검사에서 떨어져 보류됐다. 심사자는 이것을 일화라고 부를 것이고, 그 말이
맞다.

exp68이 전수표를 가진 항원마다 정합 라이브러리를 만든다. 여기서는 그 라이브러리
각각에 exp21과 **완전히 같은** 절차를 적용한다 -- 20회 분할, 같은 학습 크기, 같은
두 모형, 같은 씨앗. 바뀌는 것은 항원뿐이다.

읽는 법은 단순하다. 여러 항원에서 같은 방향이 나오면 그 결과는 항원 하나의
성질이 아니다. 방향이 갈리면 그것도 결과이고, 그때는 "정합하면 뒤집힌다"가 아니라
"항원에 따라 다르다"가 참이 된다. 어느 쪽이든 지금보다 낫다.

    python src/exp71_repr_many_antigens.py
    python src/exp71_repr_many_antigens.py --only 1ADQ_A
"""
import argparse
import hashlib
import io
import os
import sys
import time

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

sys.path.insert(0, "src")

# exp21과 같은 값. 하나라도 다르면 나란히 놓을 수 없다.
AA = "ACDEFGHIKLMNPQRSTVWY-"
AAIDX = {a: i for i, a in enumerate(AA)}
SIZES = [200, 500, 1000, 2000, 4000]
SEEDS = 20
HOLD = 1500
CACHE = "data/esm"
THREADS = int(os.environ.get("EXP71_THREADS", "4"))
OUT = "results/exp71_repr_many.csv"
OUT2 = "results/exp71_per_antigen.csv"
OUT3 = "results/exp71_per_size.csv"
REPORT = "results/exp71_repr_many.md"
LIBS = "results/exp68_matched_libraries.csv"
NL = chr(10)

# 이미 같은 규약으로 재어 둔 항원. exp21 의 출력은 이 파일과 열 구성이 같고
# 분할 수·학습 크기·모형·씨앗이 모두 같다 -- 같은 계산을 두 번 하지 않는다.
# 다시 돌려도 같은 값이 나와야 하고, 다르게 나오면 그것부터 봐야 한다.
PRIOR = {"2DD8_S": "results/exp21_seed_expansion.csv"}


def encode(seqs, ref):
    """정렬 one-hot. 길이가 같으므로 위치 i가 위치 i에 대응한다."""
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


def embeddings(seqs, var, tag):
    """exp9b의 절차를 그대로 쓴다. 캐시는 서열이 같을 때만 재사용한다.

    행 수만 보고 재사용하면 안 된다. 캐시 이름이 항원 이름으로 정해지는데
    라이브러리는 다시 만들어질 수 있고, 두 판 모두 6,000행이다. 실제로
    matched_1NCA_N_perpos.npz 는 일관성 검사에서 떨어져 폐기된 옛 판의
    임베딩이고, 지금 exp68 이 같은 이름으로 새 라이브러리를 만들고 있다.
    행 수만 맞춰 보면 라벨과 특징이 서로 다른 서열을 가리킨 채 끝까지 돌고
    상관계수만 조용히 틀린다 -- 그림 1이 대조 실험의 자세로 그려졌던 것과
    같은 종류의 사고다. 그래서 서열 자체의 해시를 옆에 적어 두고 대조한다.
    """
    path = f"{CACHE}/{tag}_perpos.npz"
    side = f"{CACHE}/{tag}_perpos.seqs.sha1"
    want = seq_hash(seqs)
    if os.path.exists(path):
        have = (io.open(side).read().strip()
                if os.path.exists(side) else None)
        if have == want:
            X = np.load(path)["X"]
            print(f"    임베딩 재사용 {path}  {X.shape}", flush=True)
            return X
        why = "서열 해시가 다릅니다" if have else "해시 기록이 없습니다"
        print(f"    {why} — 캐시를 버리고 다시 만듭니다", flush=True)
        os.replace(path, path + ".stale")
    sys.path.insert(0, "src")
    import torch
    import exp9b_esm_perpos as E9
    # exp9b 는 스레드를 16으로 고정한다. 이 기계에서는 그만큼이 없고, 지금은
    # exp68 과 다른 실험이 같이 돌고 있다. 값을 고정하는 대신 상한만 건다 --
    # 임베딩 결과는 스레드 수와 무관하므로 exp21 과의 비교는 그대로 성립한다.
    _orig = torch.set_num_threads
    torch.set_num_threads = lambda n: _orig(min(n, THREADS))
    t0 = time.time()
    try:
        X = E9.embed_positions(seqs, var, tag)
    finally:
        torch.set_num_threads = _orig
    io.open(side, "w").write(want)
    print(f"    임베딩 완료 {X.shape}  [{(time.time() - t0) / 60:.1f}분]",
          flush=True)
    return X


def run_antigen(ag, parent):
    p = f"data/absolut/matched_{ag}.parquet"
    if not os.path.exists(p):
        return None, "라이브러리 없음"
    d = pd.read_parquet(p)
    d = d[d.cdr3.str.len() == len(parent)].reset_index(drop=True)
    if len(d) < HOLD + max(SIZES):
        return None, f"서열이 {len(d):,}개뿐"
    y = d.energy.to_numpy(np.float32)
    Xs, var = encode(d.cdr3.tolist(), parent)
    Xe = embeddings(d.cdr3.tolist(), var, f"matched_{ag}")
    if len(Xe) != len(y):
        return None, f"임베딩 {len(Xe)} 대 라벨 {len(y)}"

    # 정렬 검사. 특징과 라벨이 어긋나면 상관이 조용히 0 근처가 된다.
    chk = np.random.default_rng(0).permutation(len(y))
    tr, te = chk[:2000], chk[2000:3500]
    s_oh = float(spearmanr(y[te], Xs[te] @ Ridge(alpha=1.0, solver="lsqr")
                           .fit(Xs[tr], y[tr]).coef_).statistic)
    s_es = float(spearmanr(y[te], Ridge(alpha=50.0, solver="lsqr")
                           .fit(Xe[tr], y[tr]).predict(Xe[te])).statistic)
    print(f"    정렬 검사 one-hot {s_oh:.3f} · ESM {s_es:.3f}", flush=True)
    if min(abs(s_oh), abs(s_es)) < 0.05:
        return None, f"정렬 검사 실패 (one-hot {s_oh:.3f}, ESM {s_es:.3f})"

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
            rows.append(dict(antigen=ag, n_train=m, seed=s,
                             onehot=round(a, 4), esm=round(b, 4),
                             onehot_minus_esm=round(a - b, 4)))
    return pd.DataFrame(rows), f"{len(d):,}개 · 변이 위치 {len(var)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    # 모항체는 라이브러리에서 복원한다. exp68 의 결과표는 그 프로세스가 도는
    # 동안 통째로 다시 쓰이므로 값이 뒤바뀔 수 있다.
    import matched_libs
    libs = {ag: par for ag, (par, _) in matched_libs.arms(a.only).items()}
    print(f"정합 라이브러리 {len(libs)}종")

    all_rows, notes = [], []
    for ag, par in libs.items():
        print(f"\n=== {ag}  모항체 {par} ===", flush=True)
        t0 = time.time()
        got, why = run_antigen(ag, par)
        if got is None:
            print(f"    건너뜀 — {why}")
            notes.append({"antigen": ag, "status": "skipped", "why": why})
            continue
        print(f"    {why} · {(time.time() - t0) / 60:.1f}분", flush=True)
        all_rows.append(got)
        pd.concat(all_rows).to_csv(OUT, index=False, encoding="utf-8")

    for ag, path in PRIOR.items():
        if any(ag in df.antigen.values for df in all_rows):
            continue
        if not os.path.exists(path):
            continue
        prior = pd.read_csv(path)
        prior = prior[prior.antigen == ag]
        print(f"[{ag}] exp21에서 이미 잰 {len(prior)}셀을 그대로 씁니다 "
              f"({path})")
        all_rows.append(prior[["antigen", "n_train", "seed", "onehot", "esm",
                               "onehot_minus_esm"]])

    if not all_rows:
        sys.exit("분석할 항원이 없습니다")
    R = pd.concat(all_rows, ignore_index=True)
    R.to_csv(OUT, index=False, encoding="utf-8")

    def summarise(d):
        boot = np.array([np.random.default_rng(11 + i)
                         .choice(d, len(d), True).mean() for i in range(10000)])
        try:
            w = float(wilcoxon(d).pvalue)
        except ValueError:
            w = float("nan")
        return dict(cells=len(d), mean_diff=round(d.mean(), 4),
                    ci_lo=round(np.percentile(boot, 2.5), 4),
                    ci_hi=round(np.percentile(boot, 97.5), 4),
                    esm_wins=int((d < 0).sum()),
                    p_ttest=float(ttest_1samp(d, 0.0).pvalue), p_wilcoxon=w)

    S = pd.DataFrame([dict(antigen=ag, **summarise(g.onehot_minus_esm.to_numpy()))
                      for ag, g in R.groupby("antigen")]).sort_values("antigen")
    S.to_csv(OUT2, index=False, encoding="utf-8")
    P = pd.DataFrame([dict(antigen=ag, n_train=int(m),
                           **summarise(g.onehot_minus_esm.to_numpy()))
                      for (ag, m), g in R.groupby(["antigen", "n_train"])])
    P = P.sort_values(["antigen", "n_train"])
    P.to_csv(OUT3, index=False, encoding="utf-8")
    print()
    print(S.to_string(index=False))

    n_esm = int((S.ci_hi < 0).sum())
    n_oh = int((S.ci_lo > 0).sum())
    L = ["# exp71 표현 비교 — 정합 항원 여러 종", "",
         "정합 라이브러리마다 exp21과 같은 절차를 적용했다. 20회 분할, 같은 "
         "학습 크기, 같은 두 모형, 같은 씨앗. 바뀌는 것은 항원뿐이다. "
         "음수는 언어 모형 우위다.", "",
         "| 항원 | 셀 | 평균 차 | 95% CI | ESM 우세 |", "|---|---|---|---|---|"]
    for _, x in S.iterrows():
        L.append(f"| {x.antigen} | {int(x.cells)} | {x.mean_diff:+.4f} | "
                 f"{x.ci_lo:+.4f} ~ {x.ci_hi:+.4f} | "
                 f"{int(x.esm_wins)}/{int(x.cells)} |")
    L += ["", f"구간이 0보다 작은 항원 {n_esm}종, 0보다 큰 항원 {n_oh}종, "
          f"0을 걸치는 항원 {len(S) - n_esm - n_oh}종.", "",
          "여러 항원에서 같은 방향이면 이 결과는 항원 하나의 성질이 아니다. "
          "방향이 갈리면 '정합하면 뒤집힌다'가 아니라 '항원에 따라 다르다'가 "
          "참이고, 그 경우 본문의 주장을 그렇게 바꿔야 한다.", "",
          "## 학습 크기별", "",
          "통합 평균만 보면 안 된다. 2DD8_S 에서 언어 모형의 우위는 라벨이 "
          "적을 때만 있고 4,000개에서는 0으로 붙는다. 같은 모양인지 항원마다 "
          "따로 본다.", "",
          "| 항원 | 라벨 | 평균 차 | 95% CI | ESM 우세 |", "|---|---|---|---|---|"]
    for _, x in P.iterrows():
        L.append(f"| {x.antigen} | {int(x.n_train):,} | {x.mean_diff:+.4f} | "
                 f"{x.ci_lo:+.4f} ~ {x.ci_hi:+.4f} | "
                 f"{int(x.esm_wins)}/{int(x.cells)} |")
    L.append("")
    if notes:
        L += ["## 제외된 항원", "", "| 항원 | 사유 |", "|---|---|"]
        L += [f"| {n['antigen']} | {n['why']} |" for n in notes]
        L.append("")
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}\nwrote {OUT2}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
