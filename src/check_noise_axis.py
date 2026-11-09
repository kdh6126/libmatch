"""꾸러미에 나가는 표가 "재지 못한 잡음" 을 0 으로 적고 있지 않은가.

한 번 일어난 일이다. 두 결과표의 noise 칸이 434개 전부 0.0 으로 적혀 있었다.
그림은 그 표를 쓰지 않았고 본문도 그 열을 인용하지 않았지만, libmatch match
는 카탈로그의 그 칸을 그대로 읽어 "다섯 축이 모두 일치한다" 고 답했다. 재지
않은 축이 합격 판정으로 바뀐 것이다.

고치는 것만으로는 다시 일어난다. 표를 다시 만들거나 손대는 길이 여럿이고,
그 가운데 하나만 옛 코드로 돌아도 0 이 되살아난다. 그래서 검사로 둔다.

세 가지를 본다.

  1. 복제 측정이 없는 행(replicates == 0)이 잡음을 적고 있으면, 그 행은
     라벨이 계산값이라고 선언한 행이어야 한다. 선언 없이 0 은 틀렸다.
  2. 이유 열이 있는 표에서는 적힌 값과 적힌 상태가 서로 맞아야 한다.
  3. 유도 규칙이 libmatch 의 describe() 와 아직 같은지, 표본을 실제로 다시
     재어 대조한다.

    python src/check_noise_axis.py
    python src/check_noise_axis.py --verify 0   표본 재측정을 건너뛴다
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, "libmatch")
sys.path.insert(0, "src")
import libmatch as LM                                           # noqa: E402
from profile_proteingym import (DECLARED_ZERO, MEASURED,        # noqa: E402
                                MIN_REP, UNAVAILABLE)
from repair_proteingym_noise import DECLARED, verify            # noqa: E402

TABLES = ["results/proteingym_five_axes.csv",
          "results/proteingym_catalogue.csv"]


def check_table(path):
    bad = []
    d = pd.read_csv(path)
    if "noise_to_signal" not in d.columns:
        return [f"{path}: noise_to_signal 열이 없습니다"]
    if "replicates" not in d.columns:
        return [f"{path}: replicates 열이 없어 대조할 수 없습니다"]
    nts = pd.to_numeric(d.noise_to_signal, errors="coerce")
    rep = pd.to_numeric(d.replicates, errors="coerce").fillna(-1)

    for i, r in d.iterrows():
        name, v, n = str(r["dataset"]), nts.iloc[i], int(rep.iloc[i])
        declared = bool(DECLARED.get(name))
        if pd.notna(v) and n < MIN_REP and not declared:
            # 바로 이 모양이었다. 복제가 없는데 숫자가 적혀 있다.
            bad.append(f"{path}: {name} 은 복제가 {n}건인데 잡음을 {v} 로 "
                       f"적었습니다. 계산값 선언도 없으므로 이 숫자는 "
                       f"재어 나온 값이 아닙니다")
        if pd.isna(v) and declared:
            bad.append(f"{path}: {name} 은 계산값으로 선언됐는데 잡음이 "
                       f"비어 있습니다. 선언된 표의 복제 퍼짐은 0 입니다")

    if "noise_status" not in d.columns:
        return bad
    for i, r in d.iterrows():
        name, v, st = str(r["dataset"]), nts.iloc[i], str(r["noise_status"])
        if st == UNAVAILABLE and pd.notna(v):
            bad.append(f"{path}: {name} 의 상태는 {st} 인데 값이 {v} 로 "
                       f"적혀 있습니다")
        if st in (MEASURED, DECLARED_ZERO) and pd.isna(v):
            bad.append(f"{path}: {name} 의 상태는 {st} 인데 값이 "
                       f"비어 있습니다")
        if st not in (MEASURED, DECLARED_ZERO, UNAVAILABLE):
            bad.append(f"{path}: {name} 의 상태 '{st}' 를 모르겠습니다")
        if st == DECLARED_ZERO and not DECLARED.get(name):
            bad.append(f"{path}: {name} 이 {DECLARED_ZERO} 로 적혀 있으나 "
                       f"생성 스크립트는 그 표를 계산값으로 선언하지 "
                       f"않았습니다")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, default=4, metavar="N",
                    help="표본 N개를 실제로 다시 재어 유도 규칙과 대조한다")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    bad, n_rows = [], 0
    for p in TABLES:
        if not os.path.exists(p):
            # 없는 입력은 통과가 아니다. 꾸러미에 들어가기로 되어 있는
            # 표가 사라졌다면 그것도 소리를 내야 한다.
            bad.append(f"{p} 가 없습니다. 꾸러미에 들어가는 표입니다")
            continue
        n_rows += len(pd.read_csv(p))
        bad += check_table(p)

    if a.verify:
        n, nbad = verify(a.verify)
        if nbad:
            bad.append(f"유도 규칙이 describe() 와 {nbad}/{n} 어긋났습니다")

    if bad:
        print(f"잡음 축 검사 실패 — {len(bad)}건")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"잡음 축 검사 통과 — {len(TABLES)}개 표 · {n_rows}행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
