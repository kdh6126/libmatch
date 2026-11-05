"""실험 64 — 교차 채점. 격자가 '이 항체는 저 항원에 붙는다'를 아는가.

exp62는 천연 항체 하나를 자기 항원의 무린 전수표에 대고 세운다. 그 값이
우연과 구별되지 않는다는 것까지는 나왔는데, 거기에는 교란이 둘 붙어 있다.

  종     비교 집합은 무린인데 천연 항체에는 사람 것이 섞여 있다
  조성   길이와 잔기 구성이 다르면 백분위가 결합과 무관하게 움직인다

둘 다 '항체 하나를 분포 하나에 댄다'는 설계에서 나온다. 그래서 설계를 바꾼다.
같은 항체를 자기 항원과 남의 항원 **모두**에 채점한다. 그러면 그 항체의 종도
조성도 양쪽에 똑같이 걸리므로 차이에서 상쇄된다. 남는 것은 항원-항체 짝뿐이다.

  귀무가설   격자 에너지는 짝을 모른다. 자기 항원 백분위와 남의 항원 백분위가
             같은 분포에서 온다
  대립가설   자기 항원에서 더 좋다 (백분위가 더 작다)

항체마다 쌍을 이루므로 부호 순위 검정(Wilcoxon)을 쓴다. 항체 수가 분모다.

이 검정은 exp62보다 강하다. exp62가 '못 찾았다'로 끝나는 자리에서, 이쪽은
'격자에 짝 정보가 있는가'를 직접 묻고 양방향으로 답이 난다. 자기 항원이 더
나쁘게 나오면 그것도 결과다 -- 그때는 인공물을 의심해야 한다.

전수표는 다시 받지 않는다. exp62가 남긴 정렬된 에너지 배열(.npy)이면 임의
에너지의 백분위를 낼 수 있다. 새로 드는 비용은 로컬 채점뿐이고, 항원당 한 번
부르므로 항체 90개가 1분 안쪽이다.

    python src/exp64_crossscore.py            exp62가 끝난 뒤에 돌린다
    python src/exp64_crossscore.py --limit 5  항원 몇 종만
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
_s = importlib.util.spec_from_file_location(
    "e60", "src/exp60_native_percentile.py")
E60 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E60)

import exp56_vopal_search as X                              # noqa: E402

X.to_absolut = lambda s: s

SRC_CSV = "results/exp62_native_fulltable.csv"
ENERGY_DIR = "data/absolut/energies"
OUT = "results/exp64_crossscore.csv"
REPORT = "results/exp64_crossscore.md"
MINLEN = 11


PRECHECK = "results/exp67_epitope_precheck.csv"


def broken_premise():
    """그 복합체의 항체가 이 항원의 결합체가 아닌 항원.

    이 실험은 대각선(자기 항원)과 비대각선(남의 항원)의 차이로 답을 낸다.
    그러므로 '자기 항원'이 실제로 그 항체의 항원이어야 한다. 아니면 대각선이
    비대각선과 같은 것이 되어, 있는 신호도 없는 것처럼 보인다.

    실제로 세 종에서 어긋난다. 4K24_A는 항체 이름이 anti-uPAR인데 항원은
    사슬 A(uPA)이고, 항체는 사슬 U에 붙는다. 관문 3은 통과하므로 exp62의
    분모에는 들어와 있다. exp67이 접촉을 계산해서만 가려낸다.
    """
    if not os.path.exists(PRECHECK):
        print("주의: exp67을 아직 안 돌렸다. 전제가 깨진 항원이 섞인 채로 "
              "대각선을 센다.", flush=True)
        return set()
    p = pd.read_csv(PRECHECK)
    return set(p[p.verdict == "전제 실패"].antigen.astype(str))


def natives():
    """채점된 천연 항체를 (유래 항원, 서열)로 모은다.

    같은 서열이 두 항원에서 나올 수 있다. 채점은 한 번만 하되 유래는 다 남긴다 --
    한쪽만 남기면 그 항체의 대각선이 사라진다.
    """
    d = pd.read_csv(SRC_CSV)
    d = d[d.status == "ok"]
    bad = broken_premise()
    if bad:
        n0 = len(d)
        d = d[~d.antigen.astype(str).isin(bad)]
        if n0 != len(d):
            print(f"전제 실패로 제외 {n0 - len(d)}종: "
                  + ", ".join(sorted(bad & set(pd.read_csv(SRC_CSV)
                                               .antigen.astype(str)))),
                  flush=True)
    rows = []
    for _, r in d.iterrows():
        s = str(r.native)
        if len(s) >= MINLEN and s.isalpha():
            rows.append((str(r.antigen), s))
    return rows


def scored_antigens(limit=0):
    ags = sorted(x[:-4] for x in os.listdir(ENERGY_DIR) if x.endswith(".npy"))
    return ags[:limit] if limit else ags


def cross(ag, seqs, threads, trusted):
    """항체 전부를 항원 하나에 채점하고 백분위를 낸다.

    캐시가 없는 채로 오라클을 부르면 바이너리가 /mnt/d에 대고 캐시를 만들기
    시작하고, 그 경로는 분당 1MB라 항원 하나가 몇 시간을 먹는다. 탐침은 같은
    일을 WSL 안에서 하므로 그 앞을 막아 준다.

    다만 탐침도 캐시를 통째로 읽는다. 오라클이 바로 뒤에서 또 읽으므로 항원마다
    수백 MB를 두 번 읽는 셈이고, 95종이면 그 중복만 한 시간이다. 그래서 exp62가
    이미 로컬 채점(관문 4)에 성공한 항원은 건너뛴다 -- 그때 캐시가 있었다는
    뜻이고, 캐시를 지우는 코드는 없다. 그 밖의 항원에서만 탐침을 건다.
    """
    if ag not in trusted:
        st, _, detail = E60.probe(ag)
        if st != "present":
            return None, f"탐침 {st} — {detail}"
    e = np.load(f"{ENERGY_DIR}/{ag}.npy")
    o = X.AbsolutOracle(ag, seqs, threads, f"x64_{ag}")
    o(np.arange(len(seqs)))
    pct = np.full(len(seqs), np.nan)
    ok = ~np.isnan(o.energy)
    if ok.any():
        pct[ok] = np.searchsorted(e, o.energy[ok], "left") / len(e) * 100
    return (o.energy, pct, len(e)), f"{int(ok.sum())}/{len(seqs)}개"


def rank_test(d):
    """항체의 93개 백분위 중에서 자기 항원이 몇 등인가.

    앞의 Wilcoxon은 비대각선을 중앙값 하나로 접는다. 접으면 정보가 버려지고,
    자기 항원이 '두 번째로 좋았다'와 '중간이었다'가 같은 방향의 작은 차이로만
    보인다. 같은 자료로 더 직접 물을 수 있다.

      귀무가설   격자에 짝 정보가 없다. 자기 항원의 순위는 1..N 균등분포다
      대립가설   자기 항원이 앞쪽에 쏠린다

    항체마다 순위를 0~1로 정규화해 모으고, 그 분포가 균등한지를 본다. 해석이
    직접적이라는 것이 장점이다 -- 중앙값 0.5는 "자기 항원이 남의 항원들 사이에서
    딱 가운데"라는 뜻이고, 짝 정보가 있다면 0.5보다 작아야 한다.

    동점은 중간 순위로 처리한다. 격자 에너지는 이산값이라 같은 백분위가 나올 수
    있고, 그때 앞쪽을 주면 신호가 없는데도 있는 것처럼 보인다.
    """
    from scipy import stats

    rr = []
    for (o, s), sub in d.groupby(["origin", "seq"]):
        v = sub.dropna(subset=["pct"])
        if o not in set(v.antigen) or len(v) < 10:
            continue
        own = float(v[v.antigen == o].pct.iloc[0])
        others = v[v.antigen != o].pct.to_numpy()
        # 동점은 절반만 센다(중간 순위)
        r = (others < own).sum() + 0.5 * (others == own).sum()
        rr.append({"origin": o, "seq": s, "n": len(others) + 1,
                   "rank01": r / len(others)})
    if len(rr) < 6:
        return ["", "## 자기 항원 순위", "",
                f"짝을 이룬 항체가 {len(rr)}개뿐이라 검정을 하지 않는다.", ""]
    q = pd.DataFrame(rr)
    x = np.sort(q.rank01.to_numpy())
    ks = stats.kstest(x, "uniform")
    k = int((x < 0.5).sum())
    sg = stats.binomtest(k, len(x), 0.5, alternative="two-sided")
    q.to_csv("results/exp64_ranks.csv", index=False, encoding="utf-8")
    return ["", "## 자기 항원 순위 — 중앙값으로 접지 않은 검정", "",
            "항체마다 '자기 항원 백분위가 남의 항원들 사이에서 몇 등인가'를 "
            "0~1로 재고, 그 분포가 균등한지 본다. 짝 정보가 있다면 0.5보다 "
            "작은 쪽으로 쏠린다.", "",
            "| 통계 | 값 |", "|---|---|",
            f"| 항체 | {len(q)} |",
            f"| 순위 중앙값 | {q.rank01.median():.3f} |",
            f"| 상위 10% 안에 든 항체 | {(q.rank01 < 0.1).sum()} / {len(q)} |",
            f"| KS (균등) | D={ks.statistic:.3f}, p={ks.pvalue:.3g} |",
            f"| 부호 (0.5 기준) | {k}/{len(x)}, p={sg.pvalue:.3g} |", "",
            "순위 중앙값이 0.5 근처이고 p가 크면, 격자는 그 항체를 자기 항원과 "
            "남의 항원에 같은 등수로 놓는다는 뜻이다. 그때 에너지 순위는 "
            "항원-항체 짝이 아니라 서열 자체의 성질을 재고 있다.", ""]


def analyse(d):
    """대각선(자기 항원)과 비대각선(남의 항원)을 항체 안에서 짝지어 본다."""
    from scipy import stats

    L = ["# exp64 교차 채점 — 격자가 항원-항체 짝을 아는가", "",
         "같은 항체를 자기 항원과 남의 항원 모두에 채점했다. 항체의 종과 조성은 "
         "양쪽에 똑같이 걸리므로 차이에서 상쇄된다. 남는 것은 짝뿐이다.", ""]

    # 축소 공개표를 쓴 항원은 백분위 눈금이 다른 항원과 조금 다르다(분모가
    # 2.5%쯤 작다). 대각선과 비대각선을 항원 안에서 비교하므로 영향은 작지만,
    # 그 항원이 '자기 항원'인 항체에서는 대각선만 다른 자로 재는 셈이 된다.
    # 그래서 빼고도 같은지 함께 본다.
    red = set()
    try:
        c = pd.read_csv(SRC_CSV)
        if "reduced_reference" in c.columns:
            red = set(c[c.reduced_reference.fillna(False)].antigen.astype(str))
    except Exception:
        pass

    pairs = []
    for (origin, seq), sub in d.groupby(["origin", "seq"]):
        diag = sub[sub.antigen == origin].pct
        off = sub[sub.antigen != origin].pct
        if len(diag) != 1 or len(off) < 3 or diag.isna().all():
            continue
        pairs.append({"origin": origin, "seq": seq,
                      "diag": float(diag.iloc[0]),
                      "off_median": float(off.median()),
                      "n_off": int(off.notna().sum()),
                      "reduced": origin in red})
    p = pd.DataFrame(pairs)
    if len(p) < 6:
        return L + [f"짝을 이룬 항체가 {len(p)}개뿐이라 검정을 하지 않는다.", ""]

    diff = p.diag - p.off_median
    w = stats.wilcoxon(p.diag, p.off_median, alternative="less")
    better = int((diff < 0).sum())
    sign = stats.binomtest(better, len(p), 0.5, alternative="two-sided")
    L += [f"항체 {len(p)}개, 항체당 남의 항원 중앙 {int(p.n_off.median())}종.", "",
          "| 통계 | 값 |", "|---|---|",
          f"| 자기 항원 백분위 중앙값 | {p.diag.median():.2f}% |",
          f"| 남의 항원 백분위 중앙값 | {p.off_median.median():.2f}% |",
          f"| 차이(자기 − 남) 중앙값 | {diff.median():+.2f}%p |",
          f"| 자기 항원이 더 나은 항체 | {better} / {len(p)} |", "",
          "| 검정 | 통계량 | p | 대립가설 |", "|---|---|---|---|",
          f"| Wilcoxon 부호순위 | W={w.statistic:.0f} | {w.pvalue:.3g} | "
          "자기 항원에서 백분위가 더 작다 |",
          f"| 부호 | {better}/{len(p)} | {sign.pvalue:.3g} | "
          "한쪽으로 쏠렸다(양측) |", "",
          "읽는 법. p가 작고 차이가 음수면 격자에 짝 정보가 있다. p가 크면 "
          "격자 에너지는 어느 항원이든 같은 항체를 같은 등수에 놓는다는 뜻이고, "
          "그때 백분위는 결합이 아니라 서열 자체의 성질을 재고 있다. 차이가 "
          "**양수**이면서 유의하면 자기 항원에서 오히려 나쁘다는 것이므로 "
          "인공물을 먼저 의심해야 한다.", ""]
    p.to_csv("results/exp64_pairs.csv", index=False, encoding="utf-8")
    L += rank_test(d)

    # 중복과 축소 표에 대한 민감도. 같은 H3가 여러 항원에서 유래하면 그 항체가
    # 여러 표를 던지므로, 한 H3당 한 줄만 남긴 결과를 함께 적는다.
    L += ["", "## 민감도", "",
          "| 분석 | n | 차이 중앙값 | Wilcoxon p | 부호 p |",
          "|---|---|---|---|---|"]

    def srow(tag, sub):
        if len(sub) < 6:
            return f"| {tag} | {len(sub)} | — | — | — |"
        dd = sub.diag - sub.off_median
        ww = stats.wilcoxon(sub.diag, sub.off_median, alternative="less")
        bb = int((dd < 0).sum())
        sg = stats.binomtest(bb, len(sub), 0.5, alternative="two-sided")
        return (f"| {tag} | {len(sub)} | {dd.median():+.2f}%p | "
                f"{ww.pvalue:.3f} | {sg.pvalue:.3f} |")

    L.append(srow("주분석", p))
    L.append(srow("고유 H3당 1개", p.sort_values("origin")
                  .drop_duplicates(subset="seq")))
    if p.reduced.any():
        L.append(srow("축소 표 유래 제외", p[~p.reduced]))
    L += ["", "세 줄이 같은 방향이면 중복이나 축소 표가 결론을 만들지 "
          "않았다는 뜻이다.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    nat = natives()
    seqs = sorted({s for _, s in nat})
    ags = scored_antigens(a.limit)
    # exp62가 관문 4를 통과시킨 항원. 그 자리에서 로컬 채점이 됐으므로 캐시가 있다.
    d62 = pd.read_csv(SRC_CSV)
    trusted = set(d62[d62.status == "ok"].antigen.astype(str))
    print(f"항체 {len(seqs)}개(유래 {len(nat)}줄) × 항원 {len(ags)}종 "
          f"· 캐시 확인 생략 {len(set(ags) & trusted)}종", flush=True)

    rows, t0 = [], time.time()
    for i, ag in enumerate(ags, 1):
        t1 = time.time()
        got, msg = cross(ag, seqs, a.threads, trusted)
        if got is None:
            print(f"[{i}/{len(ags)}] {ag}  건너뜀 — {msg}", flush=True)
            continue
        energy, pct, n = got
        for s, en, pc in zip(seqs, energy, pct):
            for origin, s2 in nat:
                if s2 == s:
                    rows.append({"origin": origin, "seq": s, "antigen": ag,
                                 "energy": en, "pct": pc, "table_n": n})
        print(f"[{i}/{len(ags)}] {ag}  {msg} · {time.time() - t1:.0f}초",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    d = pd.DataFrame(rows)
    if not len(d):
        sys.exit("채점된 것이 없습니다.")
    print(f"\n{len(d)}줄 · {time.time() - t0:.0f}초")
    L = analyse(d)
    io.open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {OUT}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
