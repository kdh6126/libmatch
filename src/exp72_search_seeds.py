"""실험 72 — 탐색 우위를 정합 표현 비교와 같은 자로 다시 잰다.

본문의 탐색 headline("정합하면 수백 배가 3.3배로 내려앉는다")은 exp20에서
나왔는데, exp20은 시드가 5개이고 신뢰구간이 없다. 표현 비교에서 정합 전 팔이
3~5회 분할이었던 것과 같은 문제다(exp70이 그 자리를 메웠다). 5회 평균의 비율
3.3 대 3.0을 "일치한다"고 읽으려면 그 비율이 얼마나 흔들리는지부터 보여야 한다.

바꾸는 것은 시드 수와 보고 방식뿐이다. 라이브러리, 예산, 전략, 상위 비율,
채점은 exp20 그대로다.

덧붙여 항원도 늘린다. exp20의 격자 쪽은 항원 하나(2DD8_S)였다. exp68이 만든
정합 라이브러리를 같은 규약으로 함께 돌리면, 탐색 우위도 표현 비교와 같은
방식으로 "항원 하나의 성질인가"를 물을 수 있다.

읽는 법. 격자 쪽 비율의 신뢰구간이 실측 항원들의 비율 범위 안에 들어가면
"정합 후에는 실측과 같은 크기"가 성립한다. 겹치지 않으면 성립하지 않고,
그때는 본문의 문장을 좁혀야 한다.

    python src/exp72_search_seeds.py --seeds 20
    python src/exp72_search_seeds.py --seeds 3 --arms matched:2DD8_S   (시험용)
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
# 예산 300을 넣은 이유. 정합 격자에서 회수율이 200에서 0.03, 500에서
# 0.8~0.94로 뛴다. 그 사이를 보간해 "절반에 이르는 예산"을 내면 곡선 모양에
# 값이 좌우되고, 보간의 불확실성은 시드 사분위에 담기지 않는다. 한 점을 더
# 재면 보간이 측정이 된다.
BUDGETS = [200, 300, 500, 1000]
LIBS = "results/exp68_matched_libraries.csv"
PUBLISHED = {"2DD8_S": "CARDRLLWLRLWYFDVW"}
OUT = "results/exp72_search_seeds.csv"
OUT2 = "results/exp72_ratios.csv"
REPORT = "results/exp72_search_seeds.md"
NL = chr(10)


def encode(seqs, ref):
    """exp17과 같은 정렬 one-hot. 길이가 같아 위치 i가 위치 i에 대응한다."""
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
    # exp17을 import 하지 않는 이유: 그 모듈은 불러오는 순간 argparse를 돌려
    # 이 스크립트의 인자를 삼킨다.


def matched_arms(only=None):
    """정합 라이브러리 목록. 모항체는 라이브러리 자체에서 복원한다.

    exp68 의 결과표를 읽지 않는다. 그 표는 exp68 이 항원을 끝낼 때마다 전체를
    다시 쓰므로, 그 프로세스가 도는 동안의 값은 믿을 수 없다 -- 실제로 한 항원의
    모항체가 뒤바뀐 채 읽혀 엉뚱한 자리에서 죽은 적이 있다.
    """
    import matched_libs
    return {ag: par for ag, (par, _) in matched_libs.arms(only).items()}


def sweep(y, Xs, tok, vocab, tag, seeds, rows, log):
    from exp2_search import run
    n = len(y)
    n1 = max(1, n // 100)                                        # 상위 1%
    top1 = set(np.argsort(y)[:n1].tolist())
    opt = float(y.min())
    for b in BUDGETS:
        if b > n // 3:
            continue
        t0 = time.time()
        for kind in ["random", "oneshot", "active"]:
            for s in range(seeds):
                # 시드 규칙도 exp20 그대로다. 바꾸면 기존 5개 시드가 새 20개
                # 안에 그대로 들어 있다는 보장이 깨진다.
                sel = run(kind, y, Xs, tok, b,
                          np.random.default_rng(97 * s + b), s, vocab=vocab)
                hits = len(top1 & set(sel.tolist()))
                rows.append(dict(arm=tag, lib_size=n, budget=b, strategy=kind,
                                 seed=s, hits=hits, n_top1pct=n1,
                                 recall=round(hits / n1, 4),
                                 best=round(float(y[sel].min()), 3),
                                 gap=round(float(y[sel].min()) - opt, 3)))
        r = pd.DataFrame(rows[-3 * seeds:])
        m = r.groupby("strategy")["recall"].mean()
        line = (f"  [{tag:22s}] B={b:5,}  능동={m.get('active', 0):.3f} "
                f"일회성={m.get('oneshot', 0):.3f} 무작위={m.get('random', 0):.3f}"
                f"  → {m.get('active', 0) / max(m.get('random', 1e-9), 1e-9):.1f}배"
                f"  [{(time.time() - t0) / 60:.1f}분]")
        print(line, flush=True)
        log.append(line)


def ratios(R, n_boot=10000):
    """능동/무작위 비율과 부트스트랩 구간.

    시드마다 비율을 내면 무작위 회수율이 0인 시드에서 정의되지 않는다(예산
    200에서 흔하다). 그래서 시드를 재표집해 평균의 비율을 다시 계산한다 --
    보고하는 양과 같은 양의 구간이다.
    """
    out = []
    for (arm, b), g in R.groupby(["arm", "budget"]):
        a = g[g.strategy == "active"].sort_values("seed").recall.to_numpy()
        r = g[g.strategy == "random"].sort_values("seed").recall.to_numpy()
        if len(a) != len(r) or not len(a):
            continue
        pt = a.mean() / r.mean() if r.mean() > 0 else np.nan
        bs = []
        for i in range(n_boot):
            k = np.random.default_rng(31 + i).integers(0, len(a), len(a))
            rm = r[k].mean()
            bs.append(a[k].mean() / rm if rm > 0 else np.nan)
        bs = np.asarray(bs, float)
        ok = bs[np.isfinite(bs)]
        out.append(dict(arm=arm, budget=int(b), seeds=len(a),
                        active=round(a.mean(), 4), random=round(r.mean(), 4),
                        ratio=round(pt, 2) if np.isfinite(pt) else np.nan,
                        ci_lo=round(float(np.percentile(ok, 2.5)), 2)
                        if len(ok) > 100 else np.nan,
                        ci_hi=round(float(np.percentile(ok, 97.5)), 2)
                        if len(ok) > 100 else np.nan,
                        undefined=int((~np.isfinite(bs)).sum())))
    return pd.DataFrame(out).sort_values(["budget", "arm"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="matched:AG 또는 real:NAME. 생략하면 전부")
    ap.add_argument("--budgets", nargs="*", type=int,
                    default=None, help="기본 예산을 갈음한다")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)          # exp2_search가 16으로 고정해 둔다
    global BUDGETS
    if a.budgets:
        BUDGETS = sorted(a.budgets)
        print(f"예산을 {BUDGETS} 로 바꿉니다")

    want = set(a.arms) if a.arms else None
    # 이어 달리기는 두지 않는다. 시드 규칙이 예산에만 달려 있어 같은 팔을 두 번
    # 돌리면 같은 행이 다시 쌓이고, 그 중복은 평균과 구간을 조용히 좁힌다.
    rows, log = [], []

    only_ag = ([x.split(":", 1)[1] for x in want if x.startswith("matched:")]
               if want else None)
    libs = matched_arms(only_ag)
    print(f"정합 라이브러리 {len(libs)}종: {', '.join(libs) or '없음'}")

    n_ref = None
    for ag, par in libs.items():
        tag = f"matched:{ag}"
        if want and tag not in want:
            continue
        d = pd.read_parquet(f"data/absolut/matched_{ag}.parquet")
        d = d[d.cdr3.str.len() == len(par)].reset_index(drop=True)
        y = d.energy.to_numpy(np.float32)
        n_ref = n_ref or len(y)
        X, tok, var = encode(d.cdr3.tolist(), par)
        print(f"{NL}{tag} — {len(y):,}개 · 변이 위치 {len(var)}", flush=True)
        sweep(y, X, tok, 22, tag, a.seeds, rows, log)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    from exp8_multi_target import build as build_real, DATASETS
    for name, (ds, ag) in DATASETS.items():
        tag = f"real:{name}"
        if want and tag not in want:
            continue
        y, Xs, tok, vocab, antigen = build_real(name, ds, ag)
        m = min(n_ref or 6000, len(y))
        sub = np.sort(np.random.default_rng(4242).permutation(len(y))[:m])
        print(f"{NL}{tag} — {m:,}개로 맞춤 (전체 {len(y):,})", flush=True)
        sweep(y[sub], Xs[sub], tok[sub], vocab, tag, a.seeds, rows, log)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False, encoding="utf-8")
    S = ratios(R)
    S.to_csv(OUT2, index=False, encoding="utf-8")
    print()
    print(S.to_string(index=False))

    L = ["# exp72 탐색 우위 — 시드 20개와 신뢰구간", "",
         "본문의 탐색 headline 은 시드 5개의 평균 비율이었고 구간이 없었다. "
         "여기서는 라이브러리·예산·전략·상위 1% 규칙·시드 규칙을 exp20 그대로 "
         "두고 시드 수만 늘린 뒤, 보고하는 양(평균의 비율)에 부트스트랩 구간을 "
         "붙였다.", "",
         "| 팔 | 예산 | 능동 | 무작위 | 배수 | 95% CI |",
         "|---|---|---|---|---|---|"]
    for _, x in S.iterrows():
        ci = ("—" if not np.isfinite(x.ci_lo)
              else f"{x.ci_lo:.2f} ~ {x.ci_hi:.2f}")
        rr = "—" if not np.isfinite(x.ratio) else f"{x.ratio:.2f}"
        L.append(f"| {x.arm} | {int(x.budget):,} | {x.active:.3f} | "
                 f"{x['random']:.3f} | {rr} | {ci} |")
    L += ["", "격자 쪽 비율의 구간이 실측 항원들의 비율 범위와 겹치면 "
          "'정합 후에는 실측과 같은 크기'가 성립한다. 겹치지 않으면 본문의 "
          "문장을 그만큼 좁혀야 한다.", "",
          "예산 200에서는 유도 전략이 아직 모형을 적합하지 않는다(초기 표본이 "
          "예산 전부를 쓴다). 그 칸의 배수가 1.00인 것은 결과가 아니라 설계다.",
          ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {OUT2}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
