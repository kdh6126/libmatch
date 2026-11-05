"""실험 68 — 정합 라이브러리를 항원 여러 종으로 넓힌다.

지금 논문의 정합 라이브러리 결과는 항원 하나(2DD8_S)에 기댄다. 두 번째로 만든
1NCA_N이 일관성 검사에서 떨어져 보류됐기 때문이다. 심사자가 가장 먼저 짚을
자리이고, 논문 스스로도 "오라클이 어느 서열에나 돌아가므로 값싸다"고 적어 두었다.

내려받기는 필요 없다. 전수표 zip이 일곱 종 남아 있어(4N9G_C는 서버 원본이
손상) 여섯 종을 더 만들 수 있다.

절차는 발표된 것과 같게 둔다. 바꾸면 기존 결과와 비교할 수 없다.

  1 그 항원의 전수표에서 길이 17짜리 중 에너지가 가장 낮은 것을 모항체로
  2 비말단 15자리의 단일 변이 285개 + 무작위 이중·삼중으로 6,000개
  3 같은 바이너리로 채점
  4 일관성 검사 — 모항체 에너지가 공개 전수표의 값과 같아야 한다

4가 핵심이다. 같은 바이너리·같은 항원·같은 채점 함수이므로 두 값은 반드시
같아야 하고, 1NCA_N은 여기서 13.15 단위가 어긋났다. 어긋나는 라이브러리는
저장하지 않는다 -- 부분 라이브러리는 작은 라이브러리가 아니라 편향된
라이브러리다.

    python src/exp68_more_matched.py --list      모항체만 뽑아 본다
    python src/exp68_more_matched.py             전부 만든다
"""
import argparse
import importlib.util
import io
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")
import exp15_matched_lattice as M                          # noqa: E402

_s = importlib.util.spec_from_file_location(
    "e60", "src/exp60_native_percentile.py")
E60 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E60)
_s2 = importlib.util.spec_from_file_location(
    "e61", "src/exp61_fulltable_check.py")
E61 = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(E61)

RAW = "data/absolut/raw"
OUT = "results/exp68_matched_libraries.csv"
PARENT_LEN = 17          # 발표된 라이브러리와 같은 길이
N_LIB = 6000
# 서버 원본 아카이브가 잘려 있어 읽을 수 없다(크기·md5는 서버 값과 일치).
SKIP = {"4N9G_C"}

# 이미 만들어 둔 라이브러리. 원고의 정합 결과가 여기에 기댄다.
#
# 2DD8_S 는 발표된 모항체로 만들었는데, 아래의 규칙(길이 17 중 최저 에너지)은
# 다른 서열을 고른다 -- 발표된 모항체는 748,021개 가운데 1위가 아니라 4위이고
# 에너지가 -121.84 대 -125.07 이다. 규칙대로 새로 만들면 파일 이름이 같아
# 기존 라이브러리를 덮어쓰고, 그 순간 원고의 표와 그림이 더는 재현되지 않는
# 파일을 가리키게 된다. 그래서 이 항원은 다시 만들지 않고 기존 것을 쓴다.
PUBLISHED = {"2DD8_S": "CARDRLLWLRLWYFDVW"}


def keep_published(ag, table, n17):
    """기존 라이브러리를 확인하고 그대로 쓴다. 맞지 않으면 손대지 않는다."""
    path = f"data/absolut/matched_{ag}.parquet"
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
    par = PUBLISHED[ag]
    if par not in set(d.cdr3):
        print(f"[{ag}] 기존 라이브러리에 발표 모항체가 없습니다 — 사람이 "
              f"봐야 합니다. 덮어쓰지 않고 건너뜁니다.")
        return {"antigen": ag, "parent": par, "status": "manual_check"}
    e_rel = float(table.get(par, float("nan")))
    e_lib = float(d.loc[d.cdr3 == par, "energy"].iloc[0])
    gap = abs(e_lib - e_rel)
    print(f"[{ag}] 기존 라이브러리 사용 {path} {d.shape} · 모항체 {par} "
          f"{e_lib:.2f} 대 공개 {e_rel:.2f} (차 {gap:.2e})")
    return {"antigen": ag, "parent": par, "n_len17": n17,
            "released_energy": round(e_rel, 4),
            "rescored_energy": round(e_lib, 4), "parent_gap": gap,
            "n_common": 1, "max_diff": gap, "designed": len(d),
            "energy_min": round(float(d.energy.min()), 2),
            "energy_max": round(float(d.energy.max()), 2), "minutes": 0.0,
            "status": "ok" if gap < 1e-3 else "inconsistent"}


def _read_scored(path):
    """채점 출력에서 서열 -> 최적 슬라이스 에너지."""
    best = {}
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            q = line.rstrip("\n").split("\t")
            if len(q) < 5 or q[2] != "true" or not q[1].isalpha():
                continue
            try:
                best[q[1]] = float(q[4])
            except ValueError:
                pass
    return best


def _covers(path, seqs):
    """이 출력이 지금 설계한 서열을 모두 담고 있는가.

    항원 이름만으로 쓰이는 파일이라 다른 실험의 결과일 수 있다. 서열 집합을
    대조해서 확인한 뒤에만 재사용한다 -- 그림 1이 대조 실험의 201개 자세로
    그려졌던 것이 바로 이 확인을 안 해서 생긴 일이다.
    """
    got = set(_read_scored(path))
    return set(seqs) <= got


def released_table(ag):
    """공개 전수표를 cdr3 -> energy 로 읽는다.

    exp61의 파서를 쓴다. 직접 쓴 첫 판은 zip 안의 .txt 중 첫 번째만 읽었는데,
    이 아카이브는 32개로 쪼개져 있어 691만 행 가운데 216,164행만 들어왔다.
    더 나쁜 것은 그게 **모든 항원에서 같은 분할**이라, 서로 다른 항원이 같은
    모항체를 내놓았는데도 그럴듯해 보였다는 점이다. 2DD8_S의 모항체가 발표된
    것과 달라진 것으로 겨우 알아챘다.
    """
    d, _ = E61.parse_zip(f"{RAW}/{ag}.zip")
    return dict(zip(d.cdr3, d.energy.astype(float)))


def pick_parent(table):
    """길이 17 중 에너지가 가장 낮은 서열.

    발표된 라이브러리와 같은 규칙이다. 이것이 전역 최적점이라는 비판은 따로
    있고(실제 캠페인의 모항체는 최적과 거리가 멀다), 그 비판은 규칙을 바꾸는
    대신 별도의 대조로 답하는 편이 맞다 -- 여기서 규칙을 바꾸면 기존 항원과
    비교할 수 없게 된다.
    """
    cand = [(e, s) for s, e in table.items() if len(s) == PARENT_LEN]
    if not cand:
        return None, None
    e, s = min(cand)
    return s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="모항체만 뽑는다")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    have = sorted(x[:-4] for x in os.listdir(RAW) if x.endswith(".zip"))
    have = [x for x in have if x not in SKIP]
    if a.only:
        have = [x for x in have if x in set(a.only)]
    print(f"전수표를 가진 항원 {len(have)}종: {', '.join(have)}")

    rows = []
    if os.path.exists(OUT):
        rows = pd.read_csv(OUT).to_dict("records")
    done = {r["antigen"] for r in rows}

    for ag in have:
        if ag in done and not a.list:
            print(f"[{ag}] 이미 처리됨, 건너뜀")
            continue
        t0 = time.time()
        print(f"\n[{ag}] 전수표 읽는 중 …", flush=True)
        table = released_table(ag)
        par, e_rel = pick_parent(table)
        if par is None:
            print(f"[{ag}] 길이 {PARENT_LEN}짜리가 없어 건너뜁니다")
            continue
        n17 = sum(1 for s in table if len(s) == PARENT_LEN)
        print(f"[{ag}] 표 {len(table):,}행 · 길이{PARENT_LEN} {n17:,}개 · "
              f"모항체 {par} 에너지 {e_rel:.2f}")
        if a.list:
            rows.append({"antigen": ag, "parent": par,
                         "released_energy": round(e_rel, 4),
                         "n_len17": n17, "status": "listed"})
            continue

        if ag in PUBLISHED:
            kept = keep_published(ag, table, n17)
            if kept is not None:
                rows.append(kept)
                pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
                continue

        # 캐시가 없으면 바이너리가 /mnt/d에 대고 만들기 시작해 몇 시간을 먹는다.
        st, _, detail = E60.probe(ag)
        if st != "present":
            print(f"[{ag}] 건너뜀 — 탐침 {st}: {detail}")
            rows.append({"antigen": ag, "parent": par, "status": f"probe:{st}"})
            continue

        rng = np.random.default_rng(0)
        seqs, n1 = M.mutants(par, N_LIB, rng)
        print(f"[{ag}] 설계 {len(seqs):,}개 (단일변이 {n1:,}) · 채점 시작",
              flush=True)
        cached = f"{M.SRC}/{ag}FinalBindings_Process_1_Of_1.txt"
        try:
            if os.path.exists(cached) and _covers(cached, seqs):
                print(f"[{ag}] 기존 채점 재사용 ({cached})", flush=True)
                energies = _read_scored(cached)
            else:
                energies = M.score_with_absolut(seqs, ag, f"m68_{ag}")
        except Exception as exc:
            print(f"[{ag}] 채점 실패 — {type(exc).__name__}")
            rows.append({"antigen": ag, "parent": par, "status": "score_fail"})
            continue

        df = pd.DataFrame({"cdr3": seqs})
        df["energy"] = df["cdr3"].map(energies)
        got = int(df.energy.notna().sum())
        if got != len(df) or par not in energies:
            print(f"[{ag}] 채점 누락 {len(df) - got:,}개 — 저장하지 않습니다")
            rows.append({"antigen": ag, "parent": par, "status": "incomplete",
                         "scored": got, "designed": len(df)})
            continue

        # ---- 일관성 검사. 같은 바이너리·항원·채점 함수이므로 같아야 한다.
        #
        # 허용오차는 1e-3이다. 처음에 1e-6으로 두었더니 1ADQ_A가 3.66e-06으로
        # 떨어졌는데, 그것은 불일치가 아니라 float32 반올림이다 -- 공개표를
        # 읽는 parse_zip이 에너지를 float32로 담고, -107.59의 float32 표현이
        # 정확히 그만큼 어긋난다. 1e-6은 자료형의 정밀도보다 빡빡한 기준이라
        # 멀쩡한 라이브러리를 버린다.
        #
        # 1e-3은 느슨하지 않다. 실제로 잡아야 할 1NCA_N의 어긋남은 13.15
        # 단위로, 이 문턱의 1만 3천 배다.
        TOL = 1e-3
        e_new = float(energies[par])
        gap = abs(e_new - e_rel)
        ok = gap < TOL
        print(f"[{ag}] 일관성 모항체 {e_new:.2f} 대 공개 {e_rel:.2f} "
              f"(차 {gap:.2e}) → {'통과' if ok else '실패'}")
        # 표에 함께 있는 서열 전부로도 본다. 모항체 하나만 맞는 것은 우연일 수
        # 있고, 1NCA_N이 실패한 자리가 정확히 여기다.
        common = [s for s in df.cdr3 if s in table]
        diffs = [abs(float(energies[s]) - table[s]) for s in common]
        maxdiff = max(diffs) if diffs else float("nan")
        print(f"[{ag}] 공통 서열 {len(common):,}개 최대차 {maxdiff:.2e}")
        row = {"antigen": ag, "parent": par, "n_len17": n17,
               "released_energy": round(e_rel, 4),
               "rescored_energy": round(e_new, 4),
               "parent_gap": gap, "n_common": len(common),
               "max_diff": maxdiff, "designed": len(df),
               "energy_min": round(float(df.energy.min()), 2),
               "energy_max": round(float(df.energy.max()), 2),
               "minutes": round((time.time() - t0) / 60, 1),
               "status": "ok" if (ok and maxdiff < TOL) else "inconsistent"}
        rows.append(row)
        if row["status"] == "ok":
            df["parent_margin"] = df.energy - e_new
            p = f"data/absolut/matched_{ag}.parquet"
            df.to_parquet(p, index=False)
            print(f"[{ag}] wrote {p}  {df.shape}  [{row['minutes']}분]")
        else:
            print(f"[{ag}] 일관성 실패 — 저장하지 않습니다")
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}")
    if "status" in d.columns:
        print(d.status.value_counts().to_string())


if __name__ == "__main__":
    main()
