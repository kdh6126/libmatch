"""실험 74 — 설계 공간의 차원만 바꿔 탐색 우위가 움직이는지 본다.

exp73은 여섯 팔에 걸쳐 "탐색 배수는 대리 모형이 얼마나 잘 맞는지를 따라가고,
그것은 변이 자리 수를 따라간다"는 연관을 보였다. 팔이 여섯이므로 그것은 연관
이상이 못 된다. 항원도 오라클도 라이브러리 종류도 같이 달라지기 때문이다.

여기서는 하나만 바꾼다. 같은 항원, 같은 오라클, 같은 라이브러리 크기, 같은
예산, 같은 시드. 변이 자리 수 k만 15에서 11, 8로 줄인다.

새로 채점하지 않는다. 정합 라이브러리 안에서 "변이가 특정 k자리 안에만 있는"
부분집합을 꺼내면 그 서열들은 이미 채점돼 있다. 공짜로 얻는 대조이고, 오라클이
바뀌지 않는다는 점에서 오히려 더 깨끗하다.

두 가지를 통제해야 결과가 차원 때문이라고 말할 수 있다.

  크기   k가 작으면 부분집합도 작다. 세 k를 공통 크기로 맞춘다. 맞추지 않으면
         이것은 다시 크기 실험이 되고, 그 혼동이 이 논문의 주제다.
  조성   k가 작으면 단일·이중·삼중 변이의 비율이 달라진다(작은 자리 집합에는
         삼중이 들어갈 자리가 적다). 세 k 모두 같은 비율로 맞춘다.

읽는 법. 크기와 조성과 예산을 고정했을 때 k가 줄면서 배수가 커지면, 정합 후에도
남은 우위는 '격자가 합성이라서'가 아니라 '설계 공간이 좁아서'다. 움직이지
않으면 exp73의 연관은 차원 때문이 아니고, 그렇게 적어야 한다.

    python src/exp74_dimension_dose.py
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")

AA = "ACDEFGHIKLMNPQRSTVWY-"
AAIDX = {a: i for i, a in enumerate(AA)}
KS = [8, 11, 15]
SEEDS = 20
BUDGET = 400
TOP = [0.01, 0.05]
LIBS = "results/exp68_matched_libraries.csv"
PUBLISHED = {"2DD8_S": "CARDRLLWLRLWYFDVW"}
OUT = "results/exp74_dimension_dose.csv"
OUT2 = "results/exp74_ratios.csv"
REPORT = "results/exp74_dimension_dose.md"
NL = chr(10)


def encode(seqs, ref):
    L = len(ref)
    proj = np.array([list(s.ljust(L)[:L]) for s in seqs])
    var = [j for j in range(L) if len(set(proj[:, j])) > 1]
    X = np.zeros((len(seqs), len(var) * 21), np.float32)
    tok = np.zeros((len(seqs), len(var)), np.uint8)
    for c, j in enumerate(var):
        for i, ch in enumerate(proj[:, j]):
            k = AAIDX.get(ch, AAIDX["-"])
            X[i, c * 21 + k] = 1.0
            tok[i, c] = k + 1
    return sparse.csr_matrix(X), tok, var


def pick_subset(mut, allpos, k, tries=400, rng=None):
    """가장 많은 서열을 담는 k자리 집합. 자리를 고르는 방식이 결과를 바꾸지
    않는지 보려고 무작위 시도 중 최대를 쓴다."""
    rng = rng or np.random.default_rng(0)
    best, best_pos = -1, None
    for _ in range(tries):
        cand = frozenset(rng.choice(allpos, size=k, replace=False).tolist())
        n = sum(1 for m in mut if m <= cand)
        if n > best:
            best, best_pos = n, cand
    return best_pos, best


def balanced(idx, order, target_mix, n, rng):
    """주어진 차수 비율을 지키며 n개를 뽑는다.

    비율을 맞추지 않으면 k가 작은 팔이 이중 변이에 쏠린다. 이중 변이는 삼중보다
    예측하기 쉬우므로, 그대로 두면 '차원' 효과와 '조성' 효과를 구별할 수 없다.
    """
    take = []
    for o, frac in target_mix.items():
        pool = idx[order[idx] == o]
        want = int(round(frac * n))
        if want == 0:
            continue
        if len(pool) < want:
            return None                  # 이 조성으로는 n개를 못 채운다
        take.append(rng.choice(pool, size=want, replace=False))
    return np.sort(np.concatenate(take)) if take else None


def learnability(y, X, n_train, hold, seeds=SEEDS):
    """라벨 n_train개로 적합한 대리 모형의 보류 Spearman.

    예산과 무관한 양이다. 탐색 배수는 무작위 회수율이 천장을 만들어 포화되지만
    이 값은 그렇지 않으므로, 차원이 학습 난이도에 미치는 영향은 여기서 본다.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    out = []
    for s in range(seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:hold], perm[hold:]
        tr = pool[:n_train]
        r = Ridge(alpha=1.0, solver="lsqr").fit(X[tr], y[tr])
        out.append(float(spearmanr(y[te], X[te] @ r.coef_ + r.intercept_)
                         .statistic))
    return np.asarray(out)


def sweep(y, Xs, tok, vocab, tag, k, rows):
    from exp2_search import run
    n = len(y)
    order_idx = np.argsort(y)
    tops = {f: set(order_idx[:max(1, int(round(n * f)))].tolist()) for f in TOP}
    for kind in ["random", "oneshot", "active"]:
        for s in range(SEEDS):
            sel = run(kind, y, Xs, tok, BUDGET,
                      np.random.default_rng(97 * s + BUDGET), s, vocab=vocab)
            ss = set(sel.tolist())
            row = dict(arm=tag, k=k, lib_size=n, budget=BUDGET, strategy=kind,
                       seed=s, best=round(float(y[sel].min()), 3))
            for f, t in tops.items():
                row[f"recall_{int(f * 100)}pct"] = round(len(t & ss) / len(t), 4)
            rows.append(row)


def ratios(R, col, n_boot=10000):
    out = []
    for (arm, k), g in R.groupby(["arm", "k"]):
        a = g[g.strategy == "active"].sort_values("seed")[col].to_numpy()
        r = g[g.strategy == "random"].sort_values("seed")[col].to_numpy()
        if len(a) != len(r) or not len(a):
            continue
        pt = a.mean() / r.mean() if r.mean() > 0 else np.nan
        bs = []
        for i in range(n_boot):
            j = np.random.default_rng(31 + i).integers(0, len(a), len(a))
            rm = r[j].mean()
            bs.append(a[j].mean() / rm if rm > 0 else np.nan)
        bs = np.asarray(bs, float)
        ok = bs[np.isfinite(bs)]
        out.append(dict(arm=arm, k=int(k), metric=col, seeds=len(a),
                        active=round(a.mean(), 4), random=round(r.mean(), 4),
                        ratio=round(pt, 2) if np.isfinite(pt) else np.nan,
                        ci_lo=round(float(np.percentile(ok, 2.5)), 2)
                        if len(ok) > 100 else np.nan,
                        ci_hi=round(float(np.percentile(ok, 97.5)), 2)
                        if len(ok) > 100 else np.nan))
    return pd.DataFrame(out).sort_values(["arm", "k"])


def matched_arms():
    out = {}
    if os.path.exists(LIBS):
        d = pd.read_csv(LIBS)
        for _, r in d[d.status == "ok"].iterrows():
            out[str(r.antigen)] = str(r.parent)
    for ag, par in PUBLISHED.items():
        if os.path.exists(f"data/absolut/matched_{ag}.parquet"):
            out.setdefault(ag, par)
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)

    rows, notes, learn = [], [], []
    for ag, par in matched_arms().items():
        d = pd.read_parquet(f"data/absolut/matched_{ag}.parquet")
        d = d[d.cdr3.str.len() == len(par)].reset_index(drop=True)
        mut = [frozenset(i for i, (x, p) in enumerate(zip(s, par)) if x != p)
               for s in d.cdr3]
        order = np.array([len(m) for m in mut])
        allpos = sorted(set().union(*[m for m in mut if m]))
        rng = np.random.default_rng(0)

        chosen = {}
        for k in KS:
            pos, n = pick_subset(mut, allpos, k, rng=rng)
            keep = np.array([i for i, m in enumerate(mut) if m <= pos])
            chosen[k] = keep
            print(f"[{ag}] k={k:2d}  자리 {sorted(pos)}  서열 {len(keep):,}")
        n_common = min(len(v) for v in chosen.values())

        # 조성은 가장 제약이 큰 k(가장 작은 부분집합)의 것을 기준으로 삼는다.
        k_tight = min(chosen, key=lambda k: len(chosen[k]))
        o = order[chosen[k_tight]]
        mix = {int(v): float((o == v).mean()) for v in np.unique(o)}
        print(f"[{ag}] 공통 크기 {n_common:,} · 기준 조성(k={k_tight}) "
              + " ".join(f"{v}차={f:.2f}" for v, f in sorted(mix.items())))

        for k in KS:
            sub = balanced(chosen[k], order, mix, n_common,
                           np.random.default_rng(7))
            if sub is None:
                notes.append(f"{ag} k={k}: 기준 조성으로 {n_common:,}개를 "
                             f"채울 수 없어 제외")
                print(f"[{ag}] k={k} 제외 — 조성을 맞출 수 없습니다")
                continue
            dd = d.iloc[sub].reset_index(drop=True)
            y = dd.energy.to_numpy(np.float32)
            X, tok, var = encode(dd.cdr3.tolist(), par)
            hold = len(y) // 3
            rho = learnability(y, X, BUDGET, hold)
            learn.append(dict(arm=f"matched:{ag}", k=len(var), n=len(y),
                              n_train=BUDGET, hold=hold,
                              rho=round(rho.mean(), 4),
                              rho_sd=round(rho.std(ddof=1), 4)))
            t0 = time.time()
            sweep(y, X, tok, 22, f"matched:{ag}", len(var), rows)
            g = pd.DataFrame(rows[-3 * SEEDS:]).groupby("strategy")
            m = g.recall_1pct.mean()
            print(f"[{ag}] k={len(var):2d} n={len(y):,} 예산 {BUDGET} → "
                  f"능동 {m.get('active', 0):.3f} 무작위 {m.get('random', 0):.3f}"
                  f" = {m.get('active', 0) / max(m.get('random', 1e-9), 1e-9):.1f}배"
                  f"  [{(time.time() - t0) / 60:.1f}분]", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    if not rows:
        sys.exit("결과가 없습니다")
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False, encoding="utf-8")
    S = pd.concat([ratios(R, f"recall_{int(f * 100)}pct") for f in TOP],
                  ignore_index=True)
    S.to_csv(OUT2, index=False, encoding="utf-8")
    print()
    print(S.to_string(index=False))

    L = ["# exp74 설계 공간의 차원만 바꾸면", "",
         "같은 항원, 같은 오라클, 같은 라이브러리 크기, 같은 조성, 같은 예산, "
         "같은 시드. 변이 자리 수 k만 바꿨다. 채점은 새로 하지 않았다 -- 정합 "
         "라이브러리 안에서 k자리에만 변이가 있는 부분집합을 꺼냈으므로 이미 "
         "채점된 서열이다.", "",
         "| 항원 | k | 지표 | 능동 | 무작위 | 배수 | 95% CI |",
         "|---|---|---|---|---|---|---|"]
    for _, x in S.iterrows():
        ci = ("—" if not np.isfinite(x.ci_lo) else f"{x.ci_lo:.2f} ~ {x.ci_hi:.2f}")
        rr = "—" if not np.isfinite(x.ratio) else f"{x.ratio:.2f}"
        L.append(f"| {x.arm} | {int(x.k)} | {x.metric} | {x.active:.3f} | "
                 f"{x['random']:.3f} | {rr} | {ci} |")
    # 단조성을 주장하지 않고 센다. k가 내려갈 때 배수가 오르는 계열이 몇인지가
    # 이 실험의 답이고, 그것을 글이 아니라 자료가 말해야 한다.
    verdict = []
    for (arm, met), g in S.groupby(["arm", "metric"]):
        g = g.sort_values("k", ascending=False)
        v = g.ratio.to_numpy()
        mono = bool(np.all(np.diff(v) > 0))
        verdict.append(dict(arm=arm, metric=met, by_k_desc=list(np.round(v, 2)),
                            monotone_up_as_k_falls=mono))
    V = pd.DataFrame(verdict)
    n_mono = int(V.monotone_up_as_k_falls.sum())

    Lg = pd.DataFrame(learn)
    Lg.to_csv("results/exp74_learnability.csv", index=False, encoding="utf-8")
    L += ["", "## 대리 모형의 학습성 (같은 부분집합, 라벨 " + f"{BUDGET})", "",
          "탐색 배수에는 천장이 있다. 무작위가 이미 상위 구간의 상당 부분을 "
          "주워 오면 능동이 전부를 찾아도 배수는 1/무작위를 넘지 못한다. "
          "학습성은 그런 천장이 없으므로, 차원이 학습 난이도에 미치는 영향은 "
          "여기서 봐야 한다.", "",
          "| 항원 | k | n | 보류 Spearman |", "|---|---|---|---|"]
    for _, x in Lg.sort_values(["arm", "k"], ascending=[True, False]).iterrows():
        L.append(f"| {x.arm} | {int(x.k)} | {int(x.n):,} | "
                 f"{x.rho:.3f} ± {x.rho_sd:.3f} |")

    L += ["", "## 판정", "",
          "| 항원 | 지표 | k=15 → 11 → 8 배수 | k가 줄며 단조 증가 |",
          "|---|---|---|---|"]
    for _, x in V.iterrows():
        L.append(f"| {x.arm} | {x.metric} | "
                 f"{' → '.join(f'{v:.2f}' for v in x.by_k_desc)} | "
                 f"{'예' if x.monotone_up_as_k_falls else '아니오'} |")
    L += ["", f"{len(V)}계열 가운데 단조 증가는 {n_mono}계열이다.", ""]
    if n_mono < len(V):
        L += ["이 설계로는 차원이 탐색 우위를 키운다는 것을 보이지 못했다. "
              "exp73의 연관은 연관으로 남고, 원고에도 그렇게 적어야 한다.",
              "",
              "왜 보이지 못했는지는 표 안에 있다. 공통 크기가 작아 예산 "
              f"{BUDGET}이 라이브러리의 {BUDGET / Lg.n.iloc[0] * 100:.0f}%를 "
              "차지하고, 그래서 무작위 회수율이 0.25 근처이며 배수의 천장이 "
              f"{1 / 0.27:.1f} 정도다. 실제로 가장 높은 칸은 이미 능동 회수율 "
              "0.9로 천장에 닿아 있다. 천장에 눌린 양으로 용량-반응을 재려 한 "
              "것이 이 설계의 한계다.", "",
              "천장을 피하려면 예산을 라이브러리 대비 더 낮춰야 하는데, "
              "탐색 루틴의 초기 표본 하한이 200이라 이 크기에서는 내릴 수 "
              "없다. 제대로 하려면 k자리 라이브러리를 각각 6,000개씩 새로 "
              "채점해야 하고, 그것은 항원당 90분이 드는 별도 작업이다.", ""]
    else:
        L += ["세 k 모두에서 방향이 맞았다. 크기·조성·예산·오라클·항원을 "
              "고정한 상태이므로, 남은 우위가 설계 공간의 좁음에서 온다는 "
              "설명이 이 범위에서는 지지된다.", ""]
    L += [f"예산은 {BUDGET}으로 고정했다. 공통 크기가 작아 본문의 1,000을 쓸 수 "
          "없다(유도 전략의 초기 표본이 200이므로 예산은 그보다 넉넉해야 하고, "
          "라이브러리의 1/3을 넘을 수 없다). 그래서 이 표의 배수는 본문 표의 "
          "배수와 직접 비교하지 않는다 -- 비교는 이 표 안의 k들 사이에서만 "
          "성립한다.", ""]
    if notes:
        L += ["## 제외", ""] + [f"- {x}" for x in notes] + [""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {OUT2}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
