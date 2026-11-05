"""실험 66 — 격자가 천연 항체를 실제 에피토프에 놓는가.

exp62는 "천연 항체가 격자 에너지에서 몇 등인가"를 물었다. 거기서 등수가 낮게
나와도 두 가지가 갈리지 않는다. 격자가 결합을 못 보는 것인지, 아니면 CDR-H3
한 조각만 경쇄 없이 재는 설정 탓인지다. 이 실험이 그 둘을 가른다.

물음이 에너지가 아니라 **자리**이기 때문이다. 채점 결과의 마지막 열에 자세가
들어 있다.

    ID_slide_Variant  CDR3  Best  Slide  Energy  Structure
    0_00a             CAK.. false  CAK..  -73.72  124837-UDDLLDSLSD

'124837-UDDLLDSLSD'는 격자 시작 위치와 이동열이다. 항원도 같은 형식으로
antigenLib에 있으므로(startPosChains + structChains), 둘을 같은 격자에 펴고
면을 맞댄 자리를 세면 이 항체가 항원의 어느 잔기에 닿았는지 나온다.

정답은 결정 구조에서 온다. Absolut의 hotspots는 쓰지 않는다 -- 그건 격자에
결합한 서열들이 몰리는 자리라 격자를 격자로 검증하는 셈이 된다. 대신 원본 PDB
좌표에서 항체와 4.5A 안에 있는 항원 잔기를 뽑는다.

번호는 항원마다 이어 준다. agSeq는 대개 관측 잔기를 순서대로 이은 것과 같지만
(103종 중 93종) 항상은 아니다 -- 삽입 코드가 붙은 잔기나 Absolut이 버린 끝
잔기 때문에 한 칸씩 밀리는 항원이 있다. 어느 쪽이 빠졌는지 추정해 보정하지
않고 정렬로 잇고, 동일성이 낮으면 그 항원을 뺀다. exp67이 이 검사를 전수로
돌려 101/103이 이어짐을 확인했다.

겹친 잔기 수 하나로는 많다 적다를 말할 수 없다. 그래서 같은 항원에서 무작위
무린 CDR-H3 여러 개를 함께 채점해 그 자세들이 만드는 겹침 분포를 얻고, 천연
항체가 그 분포의 어디에 있는지를 본다. 비교 대상이 같은 항원·같은 격자·같은
오라클이므로 항원의 모양이나 에피토프 크기가 끼어들지 않는다.

  귀무가설   격자는 이 항체를 아무 데나 놓는다. 천연의 겹침은 무작위 서열의
             겹침 분포 안에 있다
  대립가설   천연이 더 많이 겹친다

항원마다 백분위가 하나 나오고, 항원들에 대해 균등분포인지를 본다(exp60_report의
검정과 같은 구조).

    python src/exp66_epitope.py            exp62가 끝난 뒤에 돌린다
    python src/exp66_epitope.py --limit 3  몇 종만
"""
import argparse
import importlib.util
import io
import os
import re
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
import antigen_lib as AL                                    # noqa: E402
from exp60_gates import SRC, auth_chains, pdb_of            # noqa: E402
from lattice_decode import contacts, decode                 # noqa: E402
from pdb_contacts import (antibody_chains, atoms,           # noqa: E402
                          epitope, map_agseq_to_obs,
                          residues, resolve_ag_chains)

_s = importlib.util.spec_from_file_location(
    "e60", "src/exp60_native_percentile.py")
E60 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E60)

import exp56_vopal_search as X                              # noqa: E402

X.to_absolut = lambda s: s

SRC_CSV = "results/exp62_native_fulltable.csv"
OUT = "results/exp66_epitope.csv"
REPORT = "results/exp66_epitope.md"
NREF = 200
MIN_CONTACT_RES = 5             # 항체 사슬이 이 항원에 붙었다고 볼 최소 잔기 수


def ag_path(ag_id, lib):
    v = lib[ag_id]
    out = []
    for start, seg in zip(v["starts"], v["struct"].split("+")):
        out += decode(start, seg)
    return out, v["seq"]


def poses(ag, seqs, threads):
    """서열들을 채점하고 서열 -> (에너지, 자세) 를 돌려준다.

    AbsolutOracle은 에너지만 들고 오므로 출력 파일을 직접 읽는다. Best=true인
    줄이 그 서열의 최적 창이고, 그 줄의 Structure가 우리가 쓸 자세다.
    """
    path = f"{SRC}/x66_{ag}_input.txt"
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        for j, s in enumerate(seqs):
            fh.write(f"{j}\t{s}\n")
    cmd = (f"cd {E60.WSL_SRC} && rm -f {ag}FinalBindings_Process_*.txt && "
           f"./AbsolutNoLib repertoire {ag} x66_{ag}_input.txt {threads}")
    E60.subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc", cmd],
                       capture_output=True, text=True)
    best = {}
    for x in os.listdir(SRC):
        if not x.startswith(f"{ag}FinalBindings"):
            continue
        for line in io.open(f"{SRC}/{x}", encoding="utf-8", errors="replace"):
            q = line.rstrip("\n").split("\t")
            if len(q) < 6 or q[2] != "true" or not q[1].isalpha():
                continue
            m = re.match(r"^(\d+)-([SULDRBF]+)$", q[5].strip())
            if not m:
                continue
            try:
                best[q[1]] = (float(q[4]), int(m.group(1)), m.group(2))
            except ValueError:
                pass
    return best


def site_spread(ag_pts, poses_list, sample=60, seed=0):
    """서열이 달라지면 결합 자리도 달라지는가.

    격자는 에너지가 가장 좋은 자세를 고른다. 그러면 서열이 무엇이든 항원에서
    가장 끈적한 자리로 갈 수 있고, 그때는 천연도 무작위도 같은 곳에 놓인다.
    백분위는 ~50%가 나오고 '신호 없음'으로 읽히지만, 실제 사실은 그보다 강하다 --
    격자에서 결합 자리를 정하는 것이 항체가 아니라 항원이라는 뜻이기 때문이다.

    그래서 추측으로 두지 않고 잰다.

      jaccard   서로 다른 두 서열의 접촉 집합이 얼마나 겹치는가. 1에 가까우면
                자리가 서열과 무관하다
      coverage  전체 서열의 5% 이상이 닿은 잔기가 몇 개인가. 항원 잔기 수에
                비해 작으면 몇 자리에 몰려 있다는 뜻이다

    이 값이 있어야 백분위 결과를 어느 쪽으로도 읽을 수 있다.
    """
    import itertools
    import random
    rng = random.Random(seed)
    sets = []
    for p in poses_list:
        try:
            sets.append(set(contacts(ag_pts, decode(p[1], p[2]))))
        except ValueError:
            pass
    sets = [s for s in sets if s]
    if len(sets) < 5:
        return None
    pick = sets if len(sets) <= sample else rng.sample(sets, sample)
    js = []
    for a, b in itertools.combinations(range(len(pick)), 2):
        u = len(pick[a] | pick[b])
        js.append(len(pick[a] & pick[b]) / u if u else 0.0)
    cnt = {}
    for s in sets:
        for k in s:
            cnt[k] = cnt.get(k, 0) + 1
    thr = max(1, int(0.05 * len(sets)))
    return {"pair_jaccard": float(np.mean(js)) if js else 0.0,
            "covered": sum(1 for v in cnt.values() if v >= thr),
            "ag_residues": len(ag_pts),
            "touch_median": float(np.median([len(s) for s in sets])),
            "_sets": sets}


def native_vs_refs(ag_pts, native_pose, ref_sets, sample=60, seed=0):
    """천연의 접촉 자리가 무작위 서열의 자리와 다른가.

    pair_jaccard가 크면 "자리가 서열과 무관하다"고 말하고 싶어지는데, 그것만으로는
    부족하다. 무작위끼리 서로 닮았다는 것과 천연이 그들과 닮았다는 것은 다른
    진술이고, 천연만 다른 자리로 갈 수도 있기 때문이다.

    그래서 두 값을 같은 자로 잰다.

        ref-ref     무작위 두 개 사이의 Jaccard (기준선)
        nat-ref     천연과 무작위 사이의 Jaccard

    둘이 같으면 천연도 무작위와 구별되지 않는 자리에 놓인다는 뜻이고, 그때
    비로소 "격자에서 자리를 정하는 것은 항체가 아니라 항원"이라고 말할 수 있다.
    nat-ref가 뚜렷이 작으면 천연만 다른 곳에 간다는 뜻이라 해석이 완전히 달라진다.
    """
    import random
    try:
        nat = set(contacts(ag_pts, decode(native_pose[1], native_pose[2])))
    except ValueError:
        return None
    if not nat or len(ref_sets) < 5:
        return None
    rng = random.Random(seed)
    pick = (ref_sets if len(ref_sets) <= sample
            else rng.sample(ref_sets, sample))
    js = []
    for s in pick:
        u = len(nat | s)
        js.append(len(nat & s) / u if u else 0.0)
    return float(np.mean(js))


def overlap(ag_pts, pose, ep):
    """자세가 닿은 항원 잔기와 결정 에피토프의 겹침."""
    ab = decode(pose[1], pose[2])
    touched = set(contacts(ag_pts, ab))
    if not touched:
        return 0, 0, 0.0
    inter = len(touched & ep)
    union = len(touched | ep)
    return inter, len(touched), (inter / union if union else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nref", type=int, default=NREF)
    a = ap.parse_args()

    lib = AL.load()
    d = pd.read_csv(SRC_CSV)
    d = d[d.status == "ok"]
    if a.limit:
        d = d.head(a.limit)
    # 비교용 무작위 서열. 정규 무린 라이브러리에서 뽑되 항원마다 같은 집합을
    # 쓴다 -- 항원 간 백분위가 같은 자로 읽혀야 하기 때문이고, exp60이 비교
    # 집합을 random_state=0으로 고정한 것과 같은 이유다. 씨앗도 그쪽에 맞춘다.
    pool = (pd.read_parquet("data/absolut/library.parquet")["cdr3"]
            .sample(a.nref, random_state=0).tolist())
    # 격자는 11잔기 창으로 재므로 그보다 짧으면 값이 나오지 않는다.
    pool = [s for s in pool if len(s) >= 11]

    # 이미 끝난 항원은 건너뛴다. 항원당 3분이라 88종이면 몇 시간이고, 그동안
    # 한 번 끊기면 처음부터 다시 하는 것은 낭비다.
    rows = []
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = set(prev.antigen.astype(str))
        print(f"이어서 실행 — 이미 끝난 {len(done)}종은 건너뛴다", flush=True)
    for i, (_, r) in enumerate(d.iterrows(), 1):
        ag, nat = str(r.antigen), str(r.native)
        if ag in done:
            continue
        pdb = pdb_of(ag)
        t0 = time.time()
        v = lib.get(ag)
        if not v:
            print(f"[{i}] {ag} 건너뜀 — 정의표에 없음", flush=True)
            continue
        pts, seq = ag_path(ag, lib)
        # 항원이 실제로 어느 사슬인지는 이름을 믿지 않고 서열로 찾는다.
        # 2WUC_I는 이름의 사슬 I에 잔기가 셋뿐이고 항원 238잔기는 사슬 A에 있다.
        # 못 찾으면 그 복합체에 이 항원이 없다는 뜻이라 전제가 서지 않는다.
        found, frac = resolve_ag_chains(pdb, seq, ag.split("_")[1])
        if found is None:
            print(f"[{i}] {ag} 건너뜀 — 항원 서열이 이 PDB에 없음 "
                  f"(최선 동일성 {frac:.3f})", flush=True)
            continue
        ag_ch = "".join(sorted(found))
        obs, _, _ = residues(atoms(pdb), found)
        # agSeq 색인과 관측 잔기 색인은 대개 같지만 항상은 아니다 -- 삽입 코드나
        # Absolut이 버린 끝 잔기 때문에 한 칸씩 밀리는 항원이 있다. 정렬로 잇는다.
        # 번호가 어긋난 채로 계산하면 겹침이 조용히 엉뚱해지고, 그건 결과를
        # 보고는 알 수 없다.
        mp, _ = map_agseq_to_obs(seq, obs)
        if mp is None:
            print(f"[{i}] {ag} 건너뜀 — 잔기 번호를 잇지 못함", flush=True)
            continue
        keep = antibody_chains(pdb, ag_ch, MIN_CONTACT_RES)
        if not keep:
            print(f"[{i}] {ag} 건너뜀 — 접촉하는 항체 사슬 없음", flush=True)
            continue
        _, hit = epitope(pdb, ag_ch, "".join(c for c, _ in keep))
        # epitope()는 관측 색인으로 돌려준다. 격자 접촉은 agSeq 색인이므로
        # 같은 자로 맞춘다.
        back = {o: a for a, o in mp.items()}
        ep = {back[o] for o in hit if o in back}
        want = [nat] + [s for s in pool if s != nat]
        got = poses(ag, want, a.threads)
        if nat not in got:
            print(f"[{i}] {ag} 건너뜀 — 천연 채점 실패", flush=True)
            continue
        n_in, n_touch, jac = overlap(pts, got[nat], ep)
        ref = []
        for s in want[1:]:
            if s in got:
                ref.append(overlap(pts, got[s], ep)[0])
        if len(ref) < 30:
            print(f"[{i}] {ag} 건너뜀 — 비교 자세 {len(ref)}개뿐", flush=True)
            continue
        ref = np.array(ref, float)
        # 백분위: 천연보다 더 많이 겹친 무작위 서열의 비율. 작을수록 천연이 위다.
        #
        # 동점을 절반만 센다. 겹침은 0~10 정도의 작은 정수라 같은 값이 쏟아지는데,
        # '엄격히 큰 것'만 세면 천연이 다수와 동점일 때도 백분위가 낮게 나온다.
        # 1OSP_O가 그랬다 -- 천연 겹침 1, 무작위 평균 0.57인데 엄격 비교로는
        # 1.0%가 나왔다. 실제로는 "1보다 큰 무작위가 드물다"일 뿐이다.
        n_gt = int((ref > n_in).sum())
        n_eq = int((ref == n_in).sum())
        pct = float((n_gt + 0.5 * n_eq) / len(ref) * 100)
        # 자리가 서열에 따라 달라지는지도 함께 잰다. 안 달라지면 백분위가
        # 무엇이 나오든 '격자는 항체가 아니라 항원으로 자리를 정한다'가 된다.
        sp = site_spread(pts, [got[s] for s in want[1:] if s in got]) or {}
        # 천연이 무작위와 '같은 자리'에 놓이는지는 따로 잰다. 무작위끼리 닮았다는
        # 것과 천연이 그들과 닮았다는 것은 다른 진술이다.
        nvr = native_vs_refs(pts, got[nat], sp.get("_sets", []))
        rows.append({"antigen": ag, "pdb": pdb, "native": nat,
                     "pair_jaccard": round(
                         sp.get("pair_jaccard", float("nan")), 4),
                     "nat_ref_jaccard": (round(nvr, 4) if nvr is not None
                                         else float("nan")),
                     "covered": sp.get("covered", 0),
                     "ag_residues": sp.get("ag_residues", len(pts)),
                     "ab_chains": "".join(c for c, _ in keep),
                     "epitope_n": len(ep), "touch_n": n_touch,
                     "overlap": n_in, "jaccard": round(jac, 4),
                     "ref_n": len(ref), "ref_mean": round(float(ref.mean()), 3),
                     "ref_gt": n_gt, "ref_eq": n_eq,
                     "pct_better": round(pct, 4),
                     "minutes": round((time.time() - t0) / 60, 1)})
        print(f"[{i}] {ag}  에피토프 {len(ep)} · 천연 겹침 {n_in} · "
              f"무작위 평균 {ref.mean():.2f} · 상위 {pct:.1f}% "
              f"[{(time.time() - t0) / 60:.1f}분]", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

    if not rows:
        sys.exit("결과 없음")
    f = pd.DataFrame(rows)
    from scipy import stats
    x = np.sort(f.pct_better.to_numpy() / 100)
    ks = stats.kstest(x, "uniform") if len(x) >= 3 else None
    L = ["# exp66 에피토프 대조 — 격자가 천연 항체를 실제 자리에 놓는가", "",
         "천연 항체의 격자 자세가 닿은 항원 잔기를, 원본 결정 구조에서 그 항체와 "
         "4.5A 안에 있던 잔기와 비교했다. 같은 항원에서 무작위 무린 서열 "
         f"{int(f.ref_n.median())}개의 자세가 만드는 겹침을 비교 집합으로 쓴다.",
         "", "| 통계 | 값 |", "|---|---|",
         f"| 항원 | {len(f)} |",
         f"| 천연 겹침 중앙값 | {f.overlap.median():.1f}잔기 |",
         f"| 무작위 겹침 중앙값 | {f.ref_mean.median():.2f}잔기 |",
         f"| 백분위 중앙값 | {f.pct_better.median():.1f}% |",
         f"| 상위 10% 안 | {(f.pct_better < 10).sum()} / {len(f)} |", "",
         "## 자리가 서열에 따라 달라지는가", "",
         "격자는 에너지가 가장 좋은 자세를 고르므로, 서열이 무엇이든 항원에서 "
         "가장 끈적한 자리로 갈 수 있다. 그러면 위의 백분위가 무엇이 나오든 "
         "'격자는 항체가 아니라 항원으로 자리를 정한다'로 읽어야 한다.", "",
         "| 통계 | 값 |", "|---|---|",
         f"| 무작위-무작위 접촉 Jaccard 중앙값 | "
         f"{f.pair_jaccard.median():.3f} |",
         f"| 천연-무작위 접촉 Jaccard 중앙값 | "
         f"{f.nat_ref_jaccard.median():.3f} |",
         f"| 5% 이상이 닿은 잔기 중앙값 | {f.covered.median():.0f} |",
         f"| 항원 잔기 수 중앙값 | {f.ag_residues.median():.0f} |", "",
         "Jaccard가 1에 가깝고 닿은 잔기가 항원 크기에 비해 작으면 자리가 "
         "서열과 무관하다는 뜻이다. 두 Jaccard가 비슷하면 천연도 무작위와 "
         "구별되지 않는 자리에 놓인다는 뜻이고, 천연-무작위 쪽만 뚜렷이 작으면 "
         "천연만 다른 곳으로 간다는 뜻이라 해석이 달라진다.", ""]
    if ks is not None:
        from scipy import stats as _st
        tie = int((f.pct_better == 50.0).sum())
        lo = int((f.pct_better < 50).sum())
        hi = int((f.pct_better > 50).sum())
        sg = (_st.binomtest(lo, lo + hi, 0.5).pvalue if lo + hi else float("nan"))
        L += ["## 검정 — KS의 유의성을 신호로 읽으면 안 된다", "",
              f"| 검정 | 값 |", "|---|---|",
              f"| 균등성 KS | D={ks.statistic:.3f}, p={ks.pvalue:.3g} |",
              f"| 백분위가 정확히 50.0%인 항원 | {tie} / {len(f)} |",
              f"| 50% 미만 / 초과 | {lo} / {hi} |",
              f"| 부호 검정 (동점 제외) | p={sg:.3g} |", "",
              "KS가 균등분포를 기각하지만 방향이 반대다. 기각의 원인은 천연이 "
              "앞쪽으로 쏠린 것이 아니라 백분위가 50% 근처에 **지나치게 몰린** "
              "것이다. 겹침이 작은 정수라 동점이 쏟아지고, 특히 천연도 무작위도 "
              "에피토프에 전혀 닿지 않는 항원에서는 전원이 동점이라 백분위가 "
              "정확히 50.0%로 찍힌다. 균등분포라면 0~100%에 고루 퍼져야 한다.", "",
              "방향을 묻는 것은 부호 검정이고, 그쪽은 유의하지 않다. 즉 천연 "
              "항체의 격자 자세가 실제 에피토프에 더 잘 닿는다는 증거는 없다.", ""]
    io.open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {OUT}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
