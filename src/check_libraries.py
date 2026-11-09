"""exp68 의 라이브러리 표가 실제 파일과 맞는지 본다.

결과표는 한 번 쓰이면 다시 읽히지 않는다. exp68 의 2DD8_S 행이 그랬다.
그 라이브러리만은 규칙으로 짓지 않고 발표된 모항체로 지어 두었는데, 행이
쓰인 시점에는 규칙판이 돌고 있었다. 나중에 parent 열만 발표 모항체로
고쳐졌고 에너지 다섯 칸은 버려진 후보의 것으로 남았다. 컴파일도 검사도
통과했다 -- 숫자가 그럴듯했기 때문이다. 보충표 S19 는 그 가운데 하나를
"모항체 차이"로 실었고, 그것은 그 라이브러리의 모항체에 대한 수치가
아니었다.

사람이 기억해서 막을 일이 아니므로 기계로 막는다. 표의 각 행을 그 행이
가리키는 parquet 파일과, 모항체 차이는 exp75 의 재채점 기록과 대조한다.

항원별 예외는 두지 않는다. 일곱 행을 같은 규칙으로 본다.

없는 입력은 통과가 아니라 실패다. 대조할 파일이 사라지면 검사는 조용히
사라지는 것이 아니라 소리를 내야 한다 -- 수정 전의 2DD8_S 가 통과한 길이
바로 "대조할 기록이 없어서 넘어감" 이었다.

    python src/check_libraries.py        어긋나면 0 이 아닌 값으로 끝난다
"""
import os
import sys

import pandas as pd

E68 = "results/exp68_matched_libraries.csv"
E75D = "results/exp75_rescore_detail.csv"
TOL = 0.005                      # 표는 소수 두 자리로 반올림해 적는다
GAP_TOL = 1e-9                   # 재채점 차이는 적힌 그대로 맞아야 한다


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    bad = []
    for p in (E68, E75D):
        if not os.path.exists(p):
            bad.append(f"{p} 가 없습니다. 대조할 기록이 없으면 검사가 "
                       f"성립하지 않는다")
    if bad:
        print("어긋난 항목:")
        for b in bad:
            print(f"  {b}")
        sys.exit(1)

    d = pd.read_csv(E68)
    det = pd.read_csv(E75D)
    n_checks = 0

    for _, r in d.iterrows():
        ag = r.antigen
        p = f"data/absolut/matched_{ag}.parquet"
        if not os.path.exists(p):
            bad.append(f"{ag}: {p} 가 없어 이 행은 대조하지 못했습니다")
            continue
        lib = pd.read_parquet(p)

        checks = [("designed", int(r.designed), len(lib), 0),
                  ("energy_min", float(r.energy_min),
                   float(lib.energy.min()), TOL),
                  ("energy_max", float(r.energy_max),
                   float(lib.energy.max()), TOL)]

        par = str(r.parent)
        q = lib.loc[lib.cdr3 == par, "energy"]
        if q.empty:
            bad.append(f"{ag}: 모항체 {par} 가 라이브러리에 없습니다")
        else:
            checks.append(("rescored_energy", float(r.rescored_energy),
                           float(q.iloc[0]), TOL))

        # 모항체 차이는 재채점 기록이 권위다. 기록이 없으면 이 행의 숫자가
        # 어디서 왔는지 알 수 없으므로, 없다는 사실 자체가 실패다.
        m = det[(det.antigen == ag) & (det.cdr3 == par)]
        if len(m) != 1:
            bad.append(f"{ag}: exp75 에 모항체 {par} 의 재채점이 "
                       f"{len(m)}건입니다. 정확히 1건이어야 합니다")
        else:
            checks += [("parent_gap", float(r.parent_gap),
                        float(m.absdiff.iloc[0]), GAP_TOL),
                       ("released_energy", float(r.released_energy),
                        float(m.released.iloc[0]), TOL),
                       ("rescored_energy(exp75)", float(r.rescored_energy),
                        float(m.rescored.iloc[0]), TOL)]

        # 비교한 서열이 하나뿐이면 최대 차이는 그 하나의 차이와 같아야 한다.
        if int(r.n_common) == 1:
            checks.append(("max_diff", float(r.max_diff),
                           float(r.parent_gap), GAP_TOL))

        for name, said, truth, tol in checks:
            n_checks += 1
            if abs(said - truth) > tol:
                bad.append(f"{ag}: {name} 은 {said} 인데 대조값은 {truth}")

        print(f"  {ag:8s} 서열 {len(lib):,} · 모항체 "
              f"{float(q.iloc[0]) if not q.empty else float('nan'):.2f} · "
              f"에너지 {float(lib.energy.min()):.2f}~"
              f"{float(lib.energy.max()):.2f} · 차이 "
              f"{float(r.parent_gap):.3e}")

    if bad:
        print("\n어긋난 항목:")
        for b in bad:
            print(f"  {b}")
        sys.exit(1)
    print(f"\n{len(d)}개 행 · 대조 {n_checks}건이 모두 "
          f"라이브러리 파일과 exp75 재채점 기록에 맞습니다.")


if __name__ == "__main__":
    main()
