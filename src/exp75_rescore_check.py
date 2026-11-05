"""실험 75 — 일관성 검사를 서열 한 개에서 수백 개로 넓힌다.

지금 원고가 말하는 일관성 검사는 사실상 서열 한 개짜리다. 정합 라이브러리는
모항체 주변의 변이체이고 공개 라이브러리는 무관한 CDRH3 691만 개이므로 둘이
겹치는 서열이 거의 없다 -- 2DD8_S는 2개, 1ADQ_A와 1NCA_N은 모항체 하나뿐이다.
"모항체와, 두 표에 공통인 다른 모든 서열이 일치했다"는 문장은 참이지만 그
'모든 다른 서열'이 한 개다.

겹치는 서열이 적은 것은 설계의 결과이므로, 검사를 겹침에 의존하게 두면 안 된다.
방향을 바꾼다. 공개 표에서 서열을 직접 뽑아 같은 바이너리로 다시 채점하고
공개된 값과 맞춰 본다. 이러면 표본 크기를 우리가 정할 수 있다.

표본은 세 갈래다.

  무작위   길이 17에서 200개. 정합 라이브러리와 같은 길이이므로 그 결과에
           직접 대응한다.
  가장자리 가장 낮은 20개와 가장 높은 20개. 분포의 꼬리에서 어긋나는 경우를
           평균이 가려 버리지 않게 한다.
  길이 전체 길이를 가리지 않고 100개. 길이별 처리에 문제가 있으면 여기서 나온다.

바이너리의 출력 파일 이름이 항원 이름으로만 정해지므로, 다른 실험이 같은 항원을
채점하는 중에 이것을 돌리면 서로의 출력을 지운다. exp68이 끝난 뒤에 돌린다.

    python src/exp75_rescore_check.py --antigens 1ADQ_A 1NCA_N 2DD8_S
"""
import argparse
import importlib.util
import io
import os
import sys
import time

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")
import exp15_matched_lattice as M                              # noqa: E402

_s1 = importlib.util.spec_from_file_location(
    "e60", "src/exp60_native_percentile.py")
E60 = importlib.util.module_from_spec(_s1)
_s1.loader.exec_module(E60)
_s2 = importlib.util.spec_from_file_location(
    "e61", "src/exp61_fulltable_check.py")
E61 = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(E61)

RAW = "data/absolut/raw"
N_RANDOM17 = 200
N_EDGE = 20
N_ANYLEN = 100
TOL = 1e-3
OUT = "results/exp75_rescore_check.csv"
OUT2 = "results/exp75_rescore_detail.csv"
REPORT = "results/exp75_rescore_check.md"
NL = chr(10)


def sample(table, rng):
    """세 갈래 표본. 겹치지 않게 모으고 어디서 왔는지 이름을 붙여 둔다."""
    l17 = sorted(s for s in table if len(s) == 17)
    take, why = [], {}

    def add(seqs, tag):
        for s in seqs:
            if s not in why:
                why[s] = tag
                take.append(s)

    by_e = sorted(l17, key=lambda s: table[s])
    add(by_e[:N_EDGE], "lowest")
    add(by_e[-N_EDGE:], "highest")
    add(rng.choice(l17, size=min(N_RANDOM17, len(l17)),
                   replace=False).tolist(), "random17")
    allseq = sorted(table)
    add(rng.choice(allseq, size=min(N_ANYLEN, len(allseq)),
                   replace=False).tolist(), "anylen")
    return take, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--antigens", nargs="*", default=None)
    a = ap.parse_args()

    have = sorted(x[:-4] for x in os.listdir(RAW) if x.endswith(".zip"))
    if a.antigens:
        have = [x for x in have if x in set(a.antigens)]
    print(f"전수표를 가진 항원 {len(have)}종: {', '.join(have)}")

    rows, detail = [], []
    for ag in have:
        t0 = time.time()
        print(f"{NL}[{ag}] 전수표 읽는 중 …", flush=True)
        d, _ = E61.parse_zip(f"{RAW}/{ag}.zip")
        table = dict(zip(d.cdr3, d.energy.astype(float)))
        seqs, why = sample(table, np.random.default_rng(3))
        print(f"[{ag}] 표 {len(table):,}행 · 표본 {len(seqs)}개 "
              + " ".join(f"{k}={sum(1 for v in why.values() if v == k)}"
                         for k in ["lowest", "highest", "random17", "anylen"]),
              flush=True)

        st, _, det = E60.probe(ag)
        if st != "present":
            print(f"[{ag}] 건너뜀 — 탐침 {st}: {det}")
            rows.append({"antigen": ag, "status": f"probe:{st}"})
            continue
        try:
            got = M.score_with_absolut(seqs, ag, f"m75_{ag}")
        except Exception as exc:
            print(f"[{ag}] 채점 실패 — {type(exc).__name__}: {exc}")
            rows.append({"antigen": ag, "status": "score_fail"})
            continue

        miss = [s for s in seqs if s not in got]
        # 채점이 빠진 서열을 조용히 빼면 검사가 통과하는 쪽으로 기운다.
        # 1NCA_N이 원래 어긋났던 이유가 바로 이 누락이었다.
        if miss:
            print(f"[{ag}] 채점 누락 {len(miss)}개 — 통과로 처리하지 않습니다")
        diffs = []
        for s in seqs:
            if s not in got:
                detail.append(dict(antigen=ag, cdr3=s, kind=why[s],
                                   released=table[s], rescored=np.nan,
                                   absdiff=np.nan))
                continue
            dd = abs(float(got[s]) - table[s])
            diffs.append(dd)
            detail.append(dict(antigen=ag, cdr3=s, kind=why[s],
                               released=round(table[s], 4),
                               rescored=round(float(got[s]), 4),
                               absdiff=dd))
        diffs = np.asarray(diffs)
        n_bad = int((diffs >= TOL).sum())
        ok = (not miss) and n_bad == 0
        print(f"[{ag}] 대조 {len(diffs)}개 · 중앙차 {np.median(diffs):.2e} · "
              f"최대차 {diffs.max():.2e} · 허용초과 {n_bad}개 → "
              f"{'통과' if ok else '실패'}  [{(time.time() - t0) / 60:.1f}분]",
              flush=True)
        rows.append({"antigen": ag, "n_sampled": len(seqs),
                     "n_scored": len(diffs), "n_missing": len(miss),
                     "median_diff": float(np.median(diffs)),
                     "max_diff": float(diffs.max()),
                     "n_over_tol": n_bad,
                     "minutes": round((time.time() - t0) / 60, 1),
                     "status": "ok" if ok else "fail"})
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
        pd.DataFrame(detail).to_csv(OUT2, index=False, encoding="utf-8")

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False, encoding="utf-8")
    pd.DataFrame(detail).to_csv(OUT2, index=False, encoding="utf-8")
    print()
    print(R.to_string(index=False))

    L = ["# exp75 되채점 일관성 검사", "",
         "원고가 말하는 일관성 검사는 사실상 서열 한 개짜리였다. 정합 "
         "라이브러리와 공개 라이브러리가 겹치는 서열이 1~2개뿐이기 때문이다. "
         "여기서는 공개 표에서 서열을 직접 뽑아 같은 바이너리로 다시 채점하고 "
         f"공개된 값과 맞췄다. 허용오차는 {TOL:g}로, float32 반올림보다 크고 "
         "실제 불일치(13.15 단위)보다 1만 배 작다.", "",
         "| 항원 | 표본 | 채점 | 누락 | 중앙차 | 최대차 | 초과 | 판정 |",
         "|---|---|---|---|---|---|---|---|"]
    for _, x in R.iterrows():
        if x.get("status") in (None, "") or pd.isna(x.get("n_scored", np.nan)):
            L.append(f"| {x.antigen} | — | — | — | — | — | — | {x.status} |")
            continue
        L.append(f"| {x.antigen} | {int(x.n_sampled)} | {int(x.n_scored)} | "
                 f"{int(x.n_missing)} | {x.median_diff:.1e} | "
                 f"{x.max_diff:.1e} | {int(x.n_over_tol)} | {x.status} |")
    L += ["", "표본은 길이 17 무작위 200개, 에너지 최저·최고 각 20개, 길이를 "
          "가리지 않은 100개다. 가장자리를 따로 넣은 이유는 꼬리에서만 어긋나는 "
          "경우를 중앙값이 가리지 못하게 하기 위해서다.", "",
          "채점이 누락된 서열은 빼고 계산하지 않는다. 누락을 조용히 버리면 "
          "검사가 통과하는 쪽으로 기울고, 1NCA_N이 원래 어긋났던 이유가 바로 "
          "그 누락이었다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write(NL.join(L) + NL)
    print()
    print(NL.join(L))
    print(f"wrote {OUT}{NL}wrote {OUT2}{NL}wrote {REPORT}")


if __name__ == "__main__":
    main()
