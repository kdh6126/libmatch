"""실험 56 — 우리가 만든 라이브러리에서 능동 탐색이 결합체를 찾는가.

exp2는 Absolut!이 전수 계산해 둔 무린 691만 서열 위에서 탐색을 검증했다.
라이브러리 전체의 에너지를 알고 있으므로 "진짜 최적을 찾았는가"를 채점할 수
있었지만, 그 라이브러리는 우리가 만든 것이 아니다. exp55는 Virtual OPAL의
서열도 Absolut! 엔진으로 채점된다는 것을 보였다. 이 실험이 둘을 잇는다.

바뀌는 것은 오라클이다. exp2에서 오라클은 배열 조회였고 전부 미리 채워져
있었다. 여기서는 실제로 비싼 호출이다 — 격자 구조 파일을 올려두고 배치마다
바이너리를 부르며, 초당 한 개 남짓이다. 그래서 두 가지가 따라온다.

  전수 채점이 불가능하다   라이브러리의 진짜 최적값을 모르므로 "최적을
                           찾았는가"로 채점할 수 없다. 같은 예산의 무작위
                           대조와 비교하는 것이 유일하게 공정한 판정이다
  라운드당 한 번만 부른다   호출마다 구조 파일(139~226MB)을 다시 올리므로,
                           600개를 한 번에 보내지 개별로 보내지 않는다

무작위 대조를 반드시 같은 예산으로 두는 이유는 exp36에서 배웠다. 대조가
없으면 "찾았다"가 라이브러리 크기의 함수인지 알고리즘의 함수인지 구분되지
않는다.

말할 수 있는 것은 "우리가 만든 라이브러리에서 탐색이 무작위보다 나은 후보를
찾는다"까지다. 격자 벤치마크이지 친화도 측정이 아니므로, 그 후보가 실제로
그 병원체에 결합한다는 뜻은 아니다.

    python src/exp56_vopal_search.py --antigens 2DD8_S 1NCA_N --lib 200000
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
from exp2_search import encode, fit_cnn, cnn_score, RERANK   # noqa: E402
from exp55_vopal_absolut import to_absolut, virtual_opal_h3  # noqa: E402

SRC = "tools/Absolut-main/src"
WSL_SRC = "/mnt/d/ubuntu/BIO/Infectious/tools/Absolut-main/src"
AA = "ACDEFGHIKLMNPQRSTVWY"
AAI = {a: i for i, a in enumerate(AA)}


class AbsolutOracle:
    """배치로만 부르는 지연 오라클. 호출 횟수와 시간을 스스로 센다."""

    def __init__(self, antigen, seqs, threads=16, tag="s56"):
        self.antigen, self.seqs, self.threads, self.tag = (antigen, seqs,
                                                           threads, tag)
        self.energy = np.full(len(seqs), np.nan, dtype=np.float64)
        self.calls, self.batches, self.seconds = 0, 0, 0.0

    def __call__(self, idx):
        idx = np.asarray([i for i in np.unique(idx)
                          if np.isnan(self.energy[i])], dtype=int)
        if not len(idx):
            return
        want = [to_absolut(self.seqs[i]) for i in idx]
        path = f"{SRC}/{self.tag}_input.txt"
        with open(path, "w") as f:
            for j, s in enumerate(want):
                f.write(f"{j}\t{s}\n")
        cmd = (f"cd {WSL_SRC} && rm -f {self.antigen}FinalBindings_Process_*.txt "
               f"&& ./AbsolutNoLib repertoire {self.antigen} "
               f"{self.tag}_input.txt {self.threads}")
        t0 = time.time()
        p = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc", cmd],
                           capture_output=True, text=True)
        el = time.time() - t0
        if p.returncode != 0:
            raise RuntimeError(f"Absolut! 비정상 종료 ({p.returncode})\n"
                               f"{p.stderr[-1500:]}")
        best = {}
        for x in os.listdir(SRC):
            if not x.startswith(f"{self.antigen}FinalBindings"):
                continue
            with open(f"{SRC}/{x}", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    q = line.rstrip("\n").split("\t")
                    if len(q) < 5 or q[2] != "true" or not q[1].isalpha():
                        continue
                    try:
                        best[q[1]] = float(q[4])
                    except ValueError:
                        pass
        got = 0
        for i, s in zip(idx, want):
            if s in best:
                self.energy[i] = best[s]
                got += 1
        self.calls += got
        self.batches += 1
        self.seconds += el
        print(f"    오라클 {got:,}/{len(idx):,}개 · {el:.0f}초 "
              f"(누적 {self.calls:,}회 / {self.seconds / 60:.0f}분)", flush=True)
        if got != len(idx):
            # 부분 결과는 작은 표본이 아니라 편향된 표본이다 (exp15 참조)
            raise RuntimeError(f"채점 실패 {len(idx) - got:,}개")


def features(seqs):
    """exp2와 같은 희소 특징 — 양 말단 11잔기 위치별 원-핫 + 2-mer + 길이."""
    NT = CT = 11
    MAXL = 45
    dim = (NT + CT) * 20 + 400 + MAXL
    rows, cols = [], []
    for i, s in enumerate(seqs):
        L = len(s)
        for j in range(min(NT, L)):
            k = AAI.get(s[j])
            if k is not None:
                rows.append(i), cols.append(j * 20 + k)
        for j in range(min(CT, L)):
            k = AAI.get(s[L - 1 - j])
            if k is not None:
                rows.append(i), cols.append((NT + j) * 20 + k)
        for j in range(L - 1):
            a, b = AAI.get(s[j]), AAI.get(s[j + 1])
            if a is not None and b is not None:
                rows.append(i), cols.append((NT + CT) * 20 + a * 20 + b)
        rows.append(i), cols.append((NT + CT) * 20 + 400 + min(L, MAXL - 1))
    v = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((v, (rows, cols)), shape=(len(seqs), dim))


def active(oracle, Xs, Xenc, budget, rng, seed, rounds=4, explore=0.2):
    N = Xs.shape[0]
    n_init = budget // rounds
    measured = list(rng.choice(N, size=n_init, replace=False))
    print(f"  라운드 0 — 무작위 {n_init:,}개", flush=True)
    oracle(measured)
    seen = set(measured)
    per = (budget - n_init) // (rounds - 1)
    for r in range(rounds - 1):
        take = per if r < rounds - 2 else budget - len(measured)
        n_exp = int(take * (1 - explore))
        idx = np.asarray(measured)
        y = oracle.energy[idx]
        ridge = Ridge(alpha=1.0, solver="lsqr").fit(Xs[idx], y)
        s1 = Xs @ ridge.coef_ + ridge.intercept_
        s1[idx] = np.inf
        head = np.argpartition(s1, min(RERANK, N - 1))[:min(RERANK, N - 1)]
        m, mu, sd = fit_cnn(Xenc[idx], y, seed=seed + r)
        s2 = cnn_score(m, Xenc[head], mu, sd)
        pick = head[np.argsort(s2)[:n_exp]]
        new = [int(i) for i in pick if i not in seen]
        while len(new) < take:                       # 나머지는 explore
            c = int(rng.integers(N))
            if c not in seen and c not in new:
                new.append(c)
        new = new[:take]
        print(f"  라운드 {r + 1} — 모델 {n_exp:,} + 무작위 "
              f"{take - n_exp:,}", flush=True)
        oracle(new)
        seen.update(new)
        measured.extend(new)
    return np.asarray(measured)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--antigens", nargs="*",
                    default=["2DD8_S", "1NCA_N", "1ZTX_E", "3VRL_C",
                             "1OSP_O", "2Q8A_A"])
    ap.add_argument("--lib", type=int, default=200_000)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    print(f"Virtual OPAL 라이브러리 {a.lib:,}개 생성 중 ...", flush=True)
    seqs = virtual_opal_h3(a.lib, seed=a.seed).tolist()
    conv = [to_absolut(s) for s in seqs]
    Xs, Xenc = features(conv), encode(conv)
    print(f"  고유 {len(seqs):,}개 · 특징 {Xs.shape[1]}차원 · "
          f"{time.time() - t0:.0f}초", flush=True)

    rows = []
    for ag in a.antigens:
        print(f"\n=== {ag} ======================================", flush=True)
        rng = np.random.default_rng(a.seed)
        oa = AbsolutOracle(ag, seqs, a.threads, f"s56a_{ag}")
        idx_a = active(oa, Xs, Xenc, a.budget, rng, a.seed)
        print("  무작위 대조 — 같은 예산", flush=True)
        rng2 = np.random.default_rng(a.seed + 500)
        orr = AbsolutOracle(ag, seqs, a.threads, f"s56r_{ag}")
        idx_r = rng2.choice(len(seqs), size=a.budget, replace=False)
        orr(idx_r)

        ea, er = oa.energy[idx_a], orr.energy[idx_r]
        ref = f"data/absolut/oracle_{ag}.parquet"
        pct = np.nan
        if os.path.exists(ref):
            t = pd.read_parquet(ref, columns=["energy"])["energy"].to_numpy()
            pct = float((t < np.nanmin(ea)).mean() * 100)
        print(f"  능동 {np.nanmin(ea):.2f} · 무작위 {np.nanmin(er):.2f} · "
              f"차이 {np.nanmin(ea) - np.nanmin(er):+.2f}"
              + (f" · 무린 분포 상위 {pct:.3f}%" if pct == pct else ""))
        rows.append({"antigen": ag, "lib": len(seqs), "budget": a.budget,
                     "active_best": float(np.nanmin(ea)),
                     "random_best": float(np.nanmin(er)),
                     "active_mean": float(np.nanmean(ea)),
                     "random_mean": float(np.nanmean(er)),
                     "murine_pct_of_active_best": pct,
                     "oracle_minutes": (oa.seconds + orr.seconds) / 60})
        pd.DataFrame(rows).to_csv("results/exp56_vopal_search.csv", index=False)

    d = pd.DataFrame(rows)
    print("\n" + d.to_string(index=False))
    print(f"\nwrote results/exp56_vopal_search.csv  "
          f"(총 {(time.time() - t0) / 60:.0f}분)")


if __name__ == "__main__":
    main()
