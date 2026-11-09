"""ProteinGym 치환 벤치마크 217개를 다섯 축으로 잰다.

도구가 우리 자료에서만 돈다면 그것은 이 논문의 부속물이고, 체크리스트도 이
벤치마크 하나에 대한 이야기로 읽힌다. 그래서 항체와 무관한 공개 세트 전부에
같은 도구를 돌려 다섯 축의 분포를 낸다. 그러면 "합성과 실측이 얼마나 떨어져
있는가"가 우리 쌍 하나가 아니라 217개를 배경으로 놓인다.

오라클을 부르지 않는다. ProteinGym assay 파일에는 전장 서열이 이미 들어 있어
야생형을 따로 받을 필요도 없다. 긴 단백질에서 difflib 가 O(n^2) 이라 3,400잔기
assay 하나가 1분쯤 걸리고, 전체는 수십 분이다.

    python src/profile_proteingym.py                 전부
    python src/profile_proteingym.py --limit 20       맛보기
"""
import argparse
import glob
import io
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "libmatch")
import libmatch as LM                                           # noqa: E402

D = "data/proteingym/DMS_ProteinGym_substitutions"
REF = "data/proteingym/DMS_substitutions.csv"
OUT = "results/proteingym_five_axes.csv"

# 비교 대상. 우리 논문의 세 팔을 같은 척도로 함께 놓는다.
# 네 번째 칸은 라벨이 계산값인가이다. Absolut! 격자 에너지는 다시 계산해도
# 같은 값이 나오므로 복제 퍼짐이 0 이고, AlphaSeq 는 실측이라 복제에서
# 재야 한다. ProteinGym 어세이는 선언하지 않는다 -- 한 변이당 한 점수로
# 배포되었을 뿐 그 퍼짐이 0 이라는 뜻이 아니다.
OURS = [("matched lattice", "libmatch/demo/matched_2DD8_S.csv", None, None,
         True),
        ("released lattice", "libmatch/demo/released_2DD8_S.csv", None, None,
         True),
        ("measured antibody", "libmatch/demo/alphaseq_measured.csv",
         "sequence", "label", False)]


MEASURED = "measured"
DECLARED_ZERO = "deterministic"
UNAVAILABLE = "unavailable"

# describe() 가 잡음을 재는 데 요구하는 최소 복제 수.
MIN_REP = 5


def noise_fields(replicates, deterministic, value):
    """잡음 축을 어떻게 적을지와 그 이유.

    빈 칸만 남기면 읽는 사람은 왜 비었는지 알 수 없고, 비어 있다는 것을
    결측으로 읽어 0 으로 채울 수 있다 -- 바로 이 표에서 한 번 일어난 일이다.
    그래서 비었다는 사실과 비게 된 이유를 함께 적는다.

    describe() 의 갈래와 같은 규칙이다. 같다는 것은 믿지 말고
    src/repair_proteingym_noise.py --verify 가 실제 측정과 대조해 확인한다.
    """
    rep = int(replicates)
    if rep >= MIN_REP:
        # 재어 나온 값이다. 0.0 이라도 그것은 측정 결과다.
        return (float(value) if value is not None else None), MEASURED, (
            f"noise-to-signal measured on the {rep:,} sequences that occur "
            f"more than once")
    if rep > 0:
        return None, UNAVAILABLE, (
            f"only {rep} sequences occur more than once, too few to estimate "
            f"a replicate spread")
    if deterministic:
        return 0.0, DECLARED_ZERO, (
            "labels declared deterministic, so one value per sequence means "
            "a replicate spread of zero")
    return None, UNAVAILABLE, (
        "one score per sequence and the labels are not declared "
        "deterministic, so the replicate spread was averaged away before "
        "release and cannot be recovered")


def row_of(name, arm, kind, deterministic=False):
    nts, status, note = noise_fields(arm.replicates, deterministic,
                                     arm.noise_to_signal)
    return dict(dataset=name, kind=kind, n=arm.n,
                len_min=arm.lengths[0], len_max=arm.lengths[1],
                alignable=arm.alignable, structure=arm.structure,
                n_varying=arm.n_varying,
                median_mutations=arm.median_mutations,
                replicates=arm.replicates,
                noise_to_signal=nts,
                noise_status=status, noise_note=note,
                usable_train_max=arm.usable_train_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not os.path.isdir(D):
        sys.exit(f"{D} 가 없습니다. ProteinGym 을 먼저 내려받으십시오.")

    files = sorted(glob.glob(os.path.join(D, "*.csv")))
    if a.limit:
        files = files[:a.limit]
    print(f"ProteinGym assay {len(files)}개", flush=True)

    rows, t0 = [], time.time()
    for name, path, sc, lc, det in OURS:
        if not os.path.exists(path):
            print(f"  건너뜀 {name}: {path} 없음")
            continue
        d = LM.read_table(path, sc, lc)
        rows.append(row_of(name, LM.describe(d, name, deterministic=det),
                           "this study", deterministic=det))
        print(f"  {name}: n={rows[-1]['n']:,} "
              f"구조={rows[-1]['structure']} 차원={rows[-1]['n_varying']}",
              flush=True)

    for i, p in enumerate(files, 1):
        nm = os.path.basename(p)[:-4]
        try:
            d = LM.read_table(p, "mutated_sequence", "DMS_score")
            arm = LM.describe(d, nm)
        except SystemExit as e:
            print(f"[{i}/{len(files)}] {nm[:40]} 건너뜀 — {e}", flush=True)
            continue
        except Exception as e:
            print(f"[{i}/{len(files)}] {nm[:40]} 실패 — "
                  f"{type(e).__name__}: {e}", flush=True)
            continue
        rows.append(row_of(nm, arm, "ProteinGym"))
        if i % 20 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {(time.time()-t0)/60:.1f}분 경과",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, encoding="utf-8")
    pg = d[d.kind == "ProteinGym"]
    print(f"\nwrote {OUT}  ({len(d)}행 · ProteinGym {len(pg)}개 · "
          f"{(time.time()-t0)/60:.1f}분)")
    if len(pg):
        print(f"  구조: " + ", ".join(
            f"{k} {v}" for k, v in pg.structure.value_counts().items()))
        v = pg.n_varying.dropna()
        print(f"  차원(변이 자리): 중앙 {v.median():.0f} · "
              f"{v.min():.0f}~{v.max():.0f}")
        print(f"  크기: 중앙 {pg.n.median():,.0f} · "
              f"{pg.n.min():,}~{pg.n.max():,}")
        print(f"  정렬 가능: {int(pg.alignable.sum())}/{len(pg)}")


if __name__ == "__main__":
    main()
