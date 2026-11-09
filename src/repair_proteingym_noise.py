"""적혀 있는 noise 0.0 을 "재지 못함" 으로 바로잡는다.

두 결과표의 noise 칸이 전부 0.0 으로 적혀 있었다. 고친 뒤의 libmatch 라면
그렇게 적지 않는다 -- 복제 측정이 없고 라벨이 계산값이라고 선언되지도 않은
표는 잡음을 0 으로 적는 것이 아니라 재지 못했다고 적는다. 0.0 은 "재어 보니
잡음이 없었다" 는 주장이고, 실제로는 아무것도 재지 않았다.

표를 손으로 고치지 않는다. describe() 가 잡음을 정하는 갈래는 기록된 두
값에만 달려 있다 -- 한 번 넘게 나온 서열의 수(replicates)와, 라벨이
계산값이라는 호출자의 선언. 둘 다 이미 표에 있거나 생성 스크립트에 적혀
있으므로, 217개 어세이를 다시 읽지 않고 그 갈래만 다시 밟으면 된다. 다시
읽어 재는 데는 수십 분이 걸리고, 이 칸에 대해서는 그 수고가 같은 값을
낸다 -- 그것을 믿지 말고 확인하라고 --verify 가 있다.

    python src/repair_proteingym_noise.py            고칠 것이 있는지 본다
    python src/repair_proteingym_noise.py --verify 6 표본을 실제로 다시 재어 대조
    python src/repair_proteingym_noise.py --write    고친다

--verify 는 어세이 파일을 실제로 읽어 describe() 를 돌리고, 그 결과가 이
스크립트가 유도한 값과 같은지 본다. 같지 않으면 유도 규칙이 describe() 와
어긋난 것이므로 손대지 않고 멈춘다.
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "libmatch")
sys.path.insert(0, "src")
import libmatch as LM                                           # noqa: E402
from profile_proteingym import (D as ASSAY_DIR, OURS,           # noqa: E402
                                noise_fields)

FIVE = "results/proteingym_five_axes.csv"
CAT = "results/proteingym_catalogue.csv"
REASON = ["noise_status", "noise_note"]

# 라벨이 계산값이라고 선언된 표. 생성 스크립트의 선언을 그대로 쓴다 --
# 여기에 다시 적으면 두 곳이 갈라질 수 있다.
DECLARED = {name: bool(det) for name, _p, _s, _l, det in OURS}


def verdict(dataset, replicates, stored):
    """이 행의 잡음 축을 어떻게 적어야 하는가.

    규칙을 여기 다시 적지 않는다. 생성 스크립트가 쓰는 함수를 그대로
    불러, 고친 표와 다시 만든 표가 갈라질 수 없게 한다.
    """
    return noise_fields(replicates, DECLARED.get(dataset),
                        float(stored) if pd.notna(stored) else None)


def repaired(d, with_reason):
    """고친 사본과 바뀐 행 수를 낸다."""
    out = d.copy()
    noise, status, note = [], [], []
    for _, r in d.iterrows():
        v, st, why = verdict(str(r["dataset"]), r["replicates"],
                             r.get("noise_to_signal"))
        noise.append(v)
        status.append(st)
        note.append(why)
    before = pd.to_numeric(d["noise_to_signal"], errors="coerce")
    after = pd.Series(noise, index=d.index, dtype="float64")
    changed = int((before.notna() != after.notna()).sum()
                  + ((before.notna() & after.notna())
                     & (before.sub(after).abs() > 1e-12)).sum())
    out["noise_to_signal"] = after
    if with_reason:
        out["noise_status"] = status
        out["noise_note"] = note
    return out, changed


def column_order(d, with_reason):
    """이유 열을 생성 스크립트가 두는 자리에 둔다.

    끝에 붙이면 표를 다시 만들 때 열 순서가 달라지고, 그 차이가 "고친 표와
    다시 만든 표가 다르다" 로 보인다. 같은 자리에 둔다.
    """
    cols = [c for c in d.columns if c not in REASON]
    if not with_reason:
        return cols
    i = cols.index("noise_to_signal") + 1
    return cols[:i] + REASON + cols[i:]


def _cmp(label, replicates, got, dataset=None):
    """다시 잰 값과 유도한 값을 견준다."""
    want, st, _why = verdict(dataset or label, replicates, got)
    ok = (want is None and got is None) or (
        want is not None and got is not None and abs(want - got) <= 1e-12)
    print(f"  {'ok ' if ok else 'BAD'} {label[:46]:<46s} "
          f"replicates={int(replicates):<4d} describe={got} 유도={want} ({st})")
    return ok


def verify_ours():
    """선언된 세 표를 다시 재어 나머지 두 갈래를 덮는다.

    어세이 표본은 모두 복제가 없어 'unavailable' 한 갈래만 밟는다. 실제로
    재어 적는 갈래와 계산값으로 선언된 갈래도 확인해야 규칙 전체가 덮인다.
    격자 두 표는 재배포가 허락되지 않아 꾸러미에서 빠졌으므로, 그 자리에는
    함께 커밋된 요약 한 줄을 쓴다 -- make_demo.py --check 가 그 줄이 원본
    parquet 에서 나온 것임을 따로 보장한다.
    """
    n = bad = 0
    for name, path, sc, lc, det in OURS:
        if os.path.exists(path):
            arm = LM.describe(LM.read_table(path, sc, lc), name,
                              deterministic=det)
            got, rep = arm.noise_to_signal, arm.replicates
            label = f"{name} (re-measured)"
        else:
            alt = path.replace(".csv", ".arm.csv")
            if not os.path.exists(alt):
                print(f"  건너뜀 {name}: {path} 도 {alt} 도 없습니다")
                continue
            arm = LM.arm_from_row(pd.read_csv(alt).iloc[0].to_dict())
            got, rep = arm.noise_to_signal, arm.replicates
            label = f"{name} (committed arm row)"
        n += 1
        bad += not _cmp(label, rep, got, dataset=name)
    return n, bad


def verify(n, rng_seed=0):
    """표본 어세이를 실제로 다시 재어 유도 규칙과 대조한다.

    유도가 맞다는 말을 믿지 않는다. 작은 어세이 몇 개를 골라 파일을 읽고
    describe() 를 돌려, 이 스크립트가 적겠다고 한 값과 같은지 본다.
    """
    if not os.path.isdir(ASSAY_DIR):
        print(f"  건너뜀: {ASSAY_DIR} 가 없어 다시 잴 수 없습니다")
        return 0, 0
    paths = sorted(
        (os.path.getsize(p), p) for p in
        (os.path.join(ASSAY_DIR, f) for f in os.listdir(ASSAY_DIR))
        if p.endswith(".csv"))
    # 작은 것부터 고른다. 이 칸의 갈래는 크기와 무관하고, 큰 어세이는
    # difflib 가 O(n^2) 이라 하나에 1분쯤 걸린다.
    pick = [p for _s, p in paths[:n]]
    bad = 0
    for p in pick:
        nm = os.path.basename(p)[:-4]
        d = LM.read_table(p, "mutated_sequence", "DMS_score")
        arm = LM.describe(d, nm)
        bad += not _cmp(nm, arm.replicates, arm.noise_to_signal)
    n_o, bad_o = verify_ours()
    return len(pick) + n_o, bad + bad_o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="표를 고쳐 쓴다")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="표본 N개를 실제로 다시 재어 유도 규칙과 대조한다")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if a.verify:
        print(f"유도 규칙을 실제 측정과 대조합니다 (표본 {a.verify}개)")
        n, bad = verify(a.verify)
        if bad:
            sys.exit(f"\n{bad}/{n} 이 어긋났습니다. 유도 규칙이 describe() 와 "
                     f"맞지 않으므로 표를 고치지 않습니다.")
        if n:
            print(f"  {n}/{n} 일치")
        print()

    total = 0
    for path, with_reason in ((FIVE, True), (CAT, False)):
        if not os.path.exists(path):
            print(f"{path} 가 없습니다 -- 건너뜁니다")
            continue
        d = pd.read_csv(path)
        out, changed = repaired(d, with_reason)
        total += changed
        if with_reason:
            tally = out.noise_status.value_counts().to_dict()
            how = "  ".join(f"{k} {v}" for k, v in sorted(tally.items()))
        else:
            how = (f"measured {int(out.noise_to_signal.notna().sum())}  "
                   f"unavailable {int(out.noise_to_signal.isna().sum())}")
        print(f"{path}")
        print(f"  {len(d)}행 · 고칠 칸 {changed}개 · {how}")
        if not a.write:
            continue
        # 카탈로그는 libmatch match 가 읽는 입력이다. 열을 더하지 않는다 --
        # 빈 칸은 arm_from_row 에서 이미 None 으로 돌아오므로 "재지 못함"
        # 이 그대로 전해진다.
        out[column_order(d, with_reason)].to_csv(
            path, index=False, encoding="utf-8")
        print(f"  wrote {path}")

    if not a.write:
        print(f"\n고칠 칸이 모두 {total}개입니다. --write 로 고칩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
