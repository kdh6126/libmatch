"""정합 라이브러리 목록과 모항체를, 라이브러리 자체에서 얻는다.

실험들이 지금까지 exp68 의 결과표에서 모항체를 읽었다. 그 표는 exp68 이 항원을
하나 끝낼 때마다 메모리의 행 목록으로 **전체를 다시 쓴다**. 그래서 그 프로세스가
도는 동안 표에 가한 수정은 다음 항원에서 사라진다 -- 실제로 2DD8_S 의 모항체를
발표본으로 되돌려 놓았는데 exp68 이 다음 항원을 끝내며 1위 모항체로 되돌렸고,
그 값을 쓰던 실험이 엉뚱한 자리에서 죽었다.

라이브러리는 그렇지 않다. 항원당 한 번만 쓰이고, 모항체는 그 안에서 정확히
복원된다 -- 설계가 단일변이 285개 + 이중·삼중이므로 모든 자리에서 모항체의
문자가 최빈값이다. 표를 믿지 말고 자료를 보면 된다.

    from matched_libs import arms, parent_of
"""
import glob
import os

import numpy as np
import pandas as pd

DIR = "data/absolut"
PREFIX = "matched_"


def parent_of(d):
    """라이브러리에서 모항체를 복원한다.

    자리마다 최빈 문자를 모은다. 단일변이가 285개이고 이중·삼중이 무작위이므로
    어느 자리에서도 모항체 문자가 과반이다. 복원한 서열이 라이브러리에 실제로
    있는지 확인해 돌려준다 -- 없으면 설계가 가정과 다르다는 뜻이므로 조용히
    넘기지 않는다.
    """
    seqs = d.cdr3.tolist()
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("길이가 섞여 있습니다. 정합 라이브러리가 아닙니다.")
    arr = np.array([list(s) for s in seqs])
    par = "".join(pd.Series(arr[:, j]).mode()[0] for j in range(L))
    if par not in set(seqs):
        raise ValueError(f"복원한 모항체 {par} 가 라이브러리에 없습니다.")
    return par


def arms(only=None, extra_ok=False):
    """{항원: (모항체, 경로)}. 파일 이름에서 항원을, 내용에서 모항체를 읽는다."""
    out = {}
    for p in sorted(glob.glob(os.path.join(DIR, PREFIX + "*.parquet"))):
        ag = os.path.basename(p)[len(PREFIX):-len(".parquet")]
        if "_rank1parent" in ag and not extra_ok:
            continue
        try:
            d = pd.read_parquet(p, columns=["cdr3"])
            out[ag] = (parent_of(d), p)
        except (ValueError, KeyError) as e:
            print(f"  [{ag}] 건너뜀 — {e}")
    if only:
        out = {k: v for k, v in out.items() if k in set(only)}
    return out


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    got = arms(extra_ok=True)
    print(f"정합 라이브러리 {len(got)}개")
    for ag, (par, p) in got.items():
        d = pd.read_parquet(p)
        e = float(d.loc[d.cdr3 == par, "energy"].iloc[0])
        print(f"  {ag:20s} {par}  모항체 에너지 {e:8.2f}  "
              f"최저 {float(d.energy.min()):8.2f}")
