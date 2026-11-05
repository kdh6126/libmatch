"""실험 73 — 정합 후에도 남는 탐색 우위는 무엇을 따라가는가.

exp72가 시드를 20개로 늘리자 본문의 문장이 성립하지 않게 됐다. 예산 1,000에서
정합 격자는 3.71배(2DD8_S)와 5.76배(1ADQ_A)인데 실측 네 표적은 2.11~3.35배다.
"정합하면 배수가 일치한다"는 읽기는 시드 5개짜리 추정에 기댄 것이었다.

그러면 남은 차이는 무엇인가. 두 팔은 크기(6,000)와 "모항체 변이체"라는 구조를
맞췄지만 **설계 공간의 차원**을 맞추지 못했다. 격자 쪽은 변이 자리가 15개이고
실측 쪽은 59~89개다. 자리가 적으면 같은 라벨 수로 대리 모형이 더 잘 맞고,
유도 탐색은 대리 모형이 잘 맞는 만큼 잘 된다.

이 설명이 맞는지 본다. 팔마다 (1) 변이 자리 수, (2) 라벨 1,000개로 적합한
대리 모형의 보류 Spearman, (3) exp72의 예산 1,000 배수를 나란히 놓고 연관을
잰다. 배수가 (2)를 따라가면 남은 우위는 '격자가 합성이라서'가 아니라 '설계
공간이 좁아서'다. 따라가지 않으면 이 설명은 틀렸고 그대로 적어야 한다.

팔이 여섯이므로 상관계수 하나에 기대지는 않는다. 순서가 뒤집히는 팔이 어디인지
함께 싣는다.

    python src/exp73_why_residual.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")

AA = "ACDEFGHIKLMNPQRSTVWY-"
AAIDX = {a: i for i, a in enumerate(AA)}
N_TRAIN = 1000          # exp72의 예산과 같은 라벨 수
HOLD = 1500
SEEDS = 20
BUDGET = 1000
LIBS = "results/exp68_matched_libraries.csv"
PUBLISHED = {"2DD8_S": "CARDRLLWLRLWYFDVW"}
RATIOS = "results/exp72_ratios.csv"
OUT = "results/exp73_why_residual.csv"
REPORT = "results/exp73_why_residual.md"
NL = chr(10)


def encode(seqs, ref):
    L = len(ref)
    proj = np.array([list(s.ljust(L)[:L]) for s in seqs])
    var = [j for j in range(L) if len(set(proj[:, j])) > 1]
    X = np.zeros((len(seqs), len(var) * 21), np.float32)
    for c, j in enumerate(var):
        for i, ch in enumerate(proj[:, j]):
            X[i, c * 21 + AAIDX.get(ch, AAIDX["-"])] = 1.0
    return sparse.csr_matrix(X), var


def learnability(y, X):
    """라벨 N_TRAIN개로 적합한 대리 모형의 보류 Spearman, 시드 20개."""
    out = []
    for s in range(SEEDS):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:HOLD], perm[HOLD:]
        tr = pool[:N_TRAIN]
        r = Ridge(alpha=1.0, solver="lsqr").fit(X[tr], y[tr])
        p = X[te] @ r.coef_ + r.intercept_
        out.append(float(spearmanr(y[te], p).statistic))
    return np.asarray(out)


def matched_arms(only=None):
    """정합 라이브러리 목록. 모항체는 라이브러리 자체에서 복원한다.

    exp68 의 결과표를 읽지 않는다. 그 표는 exp68 이 항원을 끝낼 때마다 전체를
    다시 쓰므로, 그 프로세스가 도는 동안의 값은 믿을 수 없다 -- 실제로 한 항원의
    모항체가 뒤바뀐 채 읽혀 엉뚱한 자리에서 죽은 적이 있다.
    """
    import matched_libs
    return {ag: par for ag, (par, _) in matched_libs.arms(only).items()}


def main():
    if not os.path.exists(RATIOS):
        sys.exit(f"{RATIOS} 가 없습니다. exp72를 먼저 돌리십시오.")
    R = pd.read_csv(RATIOS)
    R = R[R.budget == BUDGET].set_index("arm")

    rows, n_ref = [], None
    for ag, par in matched_arms().items():
        tag = f"matched:{ag}"
        d = pd.read_parquet(f"data/absolut/matched_{ag}.parquet")
        d = d[d.cdr3.str.len() == len(par)].reset_index(drop=True)
        y = d.energy.to_numpy(np.float32)
        n_ref = n_ref or len(y)
        X, var = encode(d.cdr3.tolist(), par)
        rho = learnability(y, X)
        rows.append(dict(arm=tag, domain="matched lattice", n=len(y),
                         n_var=len(var), rho=round(rho.mean(), 4),
                         rho_sd=round(rho.std(ddof=1), 4)))
        print(f"{tag:22s} 자리 {len(var):3d}  대리 Spearman "
              f"{rho.mean():.3f}±{rho.std(ddof=1):.3f}", flush=True)

    from exp8_multi_target import build as build_real, DATASETS
    for name, (ds, ag) in DATASETS.items():
        tag = f"real:{name}"
        y, Xs, tok, vocab, antigen = build_real(name, ds, ag)
        m = min(n_ref or 6000, len(y))
        # exp72와 같은 부분집합이어야 한다. 다른 표본을 쓰면 학습 난이도와
        # 탐색 배수가 서로 다른 라이브러리의 값이 된다.
        sub = np.sort(np.random.default_rng(4242).permutation(len(y))[:m])
        ys, Xss = y[sub], Xs[sub]
        n_var = int(Xss.shape[1] // 21)
        rho = learnability(ys, Xss)
        rows.append(dict(arm=tag, domain="measured", n=m, n_var=n_var,
                         rho=round(rho.mean(), 4),
                         rho_sd=round(rho.std(ddof=1), 4)))
        print(f"{tag:22s} 자리 {n_var:3d}  대리 Spearman "
              f"{rho.mean():.3f}±{rho.std(ddof=1):.3f}", flush=True)

    D = pd.DataFrame(rows)
    D["ratio"] = [float(R.loc[t, "ratio"]) if t in R.index else np.nan
                  for t in D.arm]
    D["ci_lo"] = [float(R.loc[t, "ci_lo"]) if t in R.index else np.nan
                  for t in D.arm]
    D["ci_hi"] = [float(R.loc[t, "ci_hi"]) if t in R.index else np.nan
                  for t in D.arm]
    D = D.sort_values("ratio", ascending=False)
    D.to_csv(OUT, index=False, encoding="utf-8")
    print()
    print(D.to_string(index=False))

    ok = D.dropna(subset=["ratio"])
    s_var = spearmanr(ok.n_var, ok.ratio)
    s_rho = spearmanr(ok.rho, ok.ratio)
    p_rho = pearsonr(ok.rho, ok.ratio)

    # 자기검증. 정합 팔과 실측 팔이 두 축 모두에서 갈라져 있으면 전체 상관은
    # 군집이 둘로 나뉘어 있다는 사실을 다시 말하는 것일 뿐이다. 군집 안에서도
    # 같은 방향인지 따로 본다 -- 아니면 "따라간다"고 쓸 수 없다.
    within = []
    for dom, g in ok.groupby("domain"):
        if len(g) < 3:
            within.append((dom, len(g), float("nan"), float("nan")))
            continue
        r = pearsonr(g.rho, g.ratio)
        within.append((dom, len(g), float(r.statistic), float(r.pvalue)))
    for dom, n, r, pv in within:
        if np.isfinite(r):
            print(f"  군집 안 {dom}: 팔 {n}개 Pearson {r:+.2f} (p = {pv:.3f})")
        else:
            print(f"  군집 안 {dom}: 팔 {n}개 — 상관을 낼 수 없습니다")

    L = ["# exp73 정합 후에도 남는 탐색 우위는 무엇을 따라가는가", "",
         f"exp72는 예산 {BUDGET:,}에서 정합 격자가 실측 네 표적보다 위에 "
         "남는다는 것을 보였다. 정합이 맞춘 것은 크기와 '모항체 변이체'라는 "
         "구조이고, 맞추지 못한 것은 설계 공간의 차원이다.", "",
         "| 팔 | 종류 | 변이 자리 | 대리 Spearman | 배수 | 95% CI |",
         "|---|---|---|---|---|---|"]
    for _, x in ok.iterrows():
        L.append(f"| {x.arm} | {x.domain} | {int(x.n_var)} | "
                 f"{x.rho:.3f} ± {x.rho_sd:.3f} | {x.ratio:.2f} | "
                 f"{x.ci_lo:.2f} ~ {x.ci_hi:.2f} |")
    L += ["",
          f"배수와 변이 자리 수: Spearman {s_var.statistic:+.2f} "
          f"(p = {s_var.pvalue:.3f}, 팔 {len(ok)}개)",
          f"배수와 대리 모형 Spearman: {s_rho.statistic:+.2f} "
          f"(p = {s_rho.pvalue:.3f}), Pearson {p_rho.statistic:+.2f} "
          f"(p = {p_rho.pvalue:.3f})", "",
          "| 군집 | 팔 | 군집 안 Pearson |", "|---|---|---|"]
    for dom, n, r, pv in within:
        L.append(f"| {dom} | {n} | "
                 + (f"{r:+.2f} (p = {pv:.3f})" if np.isfinite(r) else "—")
                 + " |")
    L += ["",
          "군집 안 상관이 약하면 전체 상관은 '정합 팔이 두 축 모두에서 실측 "
          "팔보다 위에 있다'를 다시 말하는 것에 가깝다. 그때는 '배수가 적합도를 "
          "따라간다'가 아니라 '두 군집이 두 축에서 함께 떨어져 있다'가 정확한 "
          "진술이다.", "",
          "팔이 적으므로 이 상관계수 자체는 증거가 약하다. 방향과 순서가 "
          "설명과 맞는지를 보는 용도이고, 어긋나는 팔이 있으면 그것을 적는다.",
          "",
          "결정적인 대조는 자리 수를 격자 안에서 직접 바꾸는 것이다. 같은 항원 "
          "같은 오라클로 k자리만 변이시킨 라이브러리를 여러 개 만들어 배수가 "
          "k와 함께 움직이는지 보면 된다. 그것은 채점이 새로 필요하므로 여기서 "
          "하지 않았고, 지금 말할 수 있는 것은 연관의 방향까지다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
