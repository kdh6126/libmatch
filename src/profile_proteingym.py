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
OURS = [("matched lattice", "libmatch/demo/matched_2DD8_S.csv", None, None),
        ("released lattice", "libmatch/demo/released_2DD8_S.csv", None, None),
        ("measured antibody", "libmatch/demo/alphaseq_measured.csv",
         "sequence", "label")]


def row_of(name, arm, kind):
    return dict(dataset=name, kind=kind, n=arm.n,
                len_min=arm.lengths[0], len_max=arm.lengths[1],
                alignable=arm.alignable, structure=arm.structure,
                n_varying=arm.n_varying,
                median_mutations=arm.median_mutations,
                replicates=arm.replicates,
                noise_to_signal=arm.noise_to_signal,
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
    for name, path, sc, lc in OURS:
        if not os.path.exists(path):
            print(f"  건너뜀 {name}: {path} 없음")
            continue
        d = LM.read_table(path, sc, lc)
        rows.append(row_of(name, LM.describe(d, name), "this study"))
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
