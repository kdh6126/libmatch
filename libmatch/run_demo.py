"""Run libmatch on four dataset pairs and check the verdicts are as expected.

This doubles as the regression test. Each case fixes the verdict libmatch is
supposed to reach, so a change in the axis logic that alters any of them fails
here rather than silently in a paper.

    python run_demo.py            run the four cases
    python run_demo.py --verbose  print the full report for each

The four cases, in increasing distance from the data the tool was written on:

  1  matched vs released   two libraries against the same antigen, one built
                           to a recipe and one taken from a published
                           benchmark. Every axis differs; this is the case the
                           tool exists to catch. The published side is not
                           redistributed here, so it enters as the stored
                           summary row demo/released_2DD8_S.arm.csv --- the
                           five axes read that summary, not the sequences,
                           so the verdict is unchanged.
  2  measured vs matched   a real yeast-display affinity dataset against the
                           matched lattice library. Structure agrees --- both
                           are variant neighbourhoods --- and the dimension
                           axis is what separates them.
  3  ProteinGym RASH       a deep mutational scan of a protein unrelated to
  4  ProteinGym TPK1       antibodies, distributed as mutation notation
                           rather than sequences. Nothing about these is
                           antibody-specific, and both are correctly read as
                           neighbourhoods around a wild type.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import libmatch as LM  # noqa: E402

D = os.path.join(HERE, "demo")

# Each case names the two inputs, an optional wild-type file per side, and
# the verdict expected on each of the five axes.
CASES = [
    dict(name="matched vs released (same antigen, different recipe)",
         a="matched_2DD8_S.arm.csv", a_arm=True,
         b="released_2DD8_S.arm.csv", b_arm=True,
         expect=dict(size=LM.MISMATCH, structure=LM.MISMATCH,
                     dimension=LM.MISMATCH, noise=LM.MATCH,
                     budget=LM.MISMATCH)),
    dict(name="measured yeast display vs matched lattice",
         a="alphaseq_measured.csv", b="matched_2DD8_S.arm.csv", b_arm=True,
         expect=dict(size=LM.MISMATCH, structure=LM.MATCH,
                     dimension=LM.MISMATCH, noise=LM.MISMATCH,
                     budget=LM.MISMATCH)),
    dict(name="ProteinGym RASH_HUMAN_Bandaru_2017 vs matched lattice",
         a="proteingym_RASH_HUMAN_Bandaru_2017.csv", b="matched_2DD8_S.arm.csv", b_arm=True,
         wt_a="proteingym_RASH_HUMAN_Bandaru_2017.wt.txt",
         expect=dict(size=LM.MATCH, structure=LM.MATCH,
                     dimension=LM.MISMATCH, noise=LM.MATCH,
                     budget=LM.MATCH)),
    dict(name="ProteinGym TPK1_HUMAN_Weile_2017 vs matched lattice",
         a="proteingym_TPK1_HUMAN_Weile_2017.csv", b="matched_2DD8_S.arm.csv", b_arm=True,
         wt_a="proteingym_TPK1_HUMAN_Weile_2017.wt.txt",
         expect=dict(size=LM.MATCH, structure=LM.MATCH,
                     dimension=LM.MISMATCH, noise=LM.MATCH,
                     budget=LM.MATCH)),
]


def _side(c, key, wt):
    """사례의 한 쪽을 Arm 으로.

    Absolut! 산출물은 이 꾸러미에 서열이 실리지 않는다. 다섯 축이 쓰는 것은
    서열이 아니라 그로부터 센 요약값이므로, 그런 쪽은 저장해 둔 한 줄을
    되살려 비교한다. 판정은 서열을 다시 읽은 것과 같다.
    """
    name = c[key]
    if c.get(key + "_arm"):
        return LM.arm_from_row(
            pd.read_csv(os.path.join(D, name)).iloc[0].to_dict())
    return LM.describe(LM.read_table(os.path.join(D, name), wt=wt), name)


def run_case(c, verbose=False):
    wt_a = LM._read_wt(os.path.join(D, c["wt_a"])) if c.get("wt_a") else None
    wt_b = LM._read_wt(os.path.join(D, c["wt_b"])) if c.get("wt_b") else None
    A = _side(c, "a", wt_a)
    B = _side(c, "b", wt_b)
    cmp_df = LM.compare(A, B)
    if verbose:
        print(LM.render(A, B, cmp_df))
    got = dict(zip(cmp_df.axis, cmp_df.status))
    wrong = {k: (v, got.get(k)) for k, v in c["expect"].items()
             if got.get(k) != v}
    return got, wrong, cmp_df


def check_catalogue(verbose=False):
    """profile 로 만든 카탈로그를 match 가 다시 읽을 수 있는가.

    두 하위명령은 CSV 한 장으로만 이어져 있어서, 열 이름이 어긋나면 각자는
    멀쩡히 돌면서 함께 쓸 때만 깨진다. 왕복을 한 번 돌려 둔다.
    """
    import tempfile

    # 왕복이 깨지는 자리는 열 이름이고, 그것은 서열 없이도 똑같이 검사된다.
    rows = [LM.arm_to_row(LM.describe(
        LM.read_table(os.path.join(D, "alphaseq_measured.csv")),
        "alphaseq_measured.csv"))]
    for f in ("matched_2DD8_S.arm.csv", "released_2DD8_S.arm.csv"):
        rows.append(LM.arm_to_row(LM.arm_from_row(
            pd.read_csv(os.path.join(D, f)).iloc[0].to_dict())))
    cat = pd.DataFrame(rows)[LM.CAT_COLS]
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "catalogue.csv")
        cat.to_csv(fp, index=False, encoding="utf-8")
        back = pd.read_csv(fp)
        target = LM.arm_from_row(pd.read_csv(
            os.path.join(D, "matched_2DD8_S.arm.csv"))
            .iloc[0].to_dict())
        hits = LM.match(target, back, top=5)
    problems = []
    if len(back) != len(rows):
        problems.append(f"카탈로그가 {len(back)}행, {len(rows)}행이어야 함")
    # 표적 자신은 결과에서 빠져야 한다.
    if "matched_2DD8_S.csv" in set(hits.dataset):
        problems.append("표적이 자기 자신과 대조됐습니다")
    if len(hits) != len(rows) - 1:
        problems.append(f"후보가 {len(hits)}개, {len(rows)-1}개여야 함")
    if verbose:
        print(LM.render_match(target, hits, len(back), 5))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    missing = [f for c in CASES for f in (c["a"], c["b"])
               if not os.path.exists(os.path.join(D, f))]
    if missing:
        sys.exit("missing demo files: " + ", ".join(sorted(set(missing))))

    bad = 0
    for i, c in enumerate(CASES, 1):
        got, wrong, cmp_df = run_case(c, args.verbose)
        n_mis = int((cmp_df.status == LM.MISMATCH).sum())
        mark = "ok  " if not wrong else "FAIL"
        print(f"{mark} {i}. {c['name']}")
        print(f"       {n_mis}/5 mismatched  "
              + "  ".join(f"{k}={'X' if v == LM.MISMATCH else '.'}"
                          for k, v in got.items()))
        for axis, (want, have) in wrong.items():
            print(f"       {axis}: expected {want}, got {have}")
            bad += 1
    probs = check_catalogue(args.verbose)
    print(f"{'ok  ' if not probs else 'FAIL'} 5. profile -> match round trip")
    for p in probs:
        print(f"       {p}")
    bad += len(probs)

    print()
    print("all cases behave as expected" if not bad
          else f"{bad} checks failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
