"""실험 67 — 에피토프 대조가 몇 종에서 성립하는지 미리 센다.

exp66은 두 전제 위에 선다.

  대응   Absolut의 agSeq가 항원 사슬의 관측 잔기를 순서대로 이은 것과 같다.
         같아야 격자 잔기 번호와 결정 구조 잔기가 정렬 없이 이어진다.
  짝     그 항원 사슬에 실제로 붙어 있는 항체 사슬을 고를 수 있다.

둘 다 표본 몇 개에서는 확인했는데, 그것으로 103종을 장담할 수 없다. 변형 잔기,
여러 벌이 든 결정, Absolut이 손본 항원 같은 것들이 있다. 전제가 깨지는 항원이
많으면 exp66의 분모가 작아지고, 그것을 실행 중에 알면 이미 늦다.

바이너리가 필요 없으므로 확장과 같이 돌려도 된다. 좌표만 내려받으면 된다.

    python src/exp67_epitope_precheck.py
"""
import io
import os
import sys
import time

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")
import antigen_lib as AL                                    # noqa: E402
from exp60_gates import pdb_of                              # noqa: E402
from pdb_contacts import (antibody_chains, atoms,           # noqa: E402
                          epitope, is_antibody_chain,
                          residues, resolve_ag_chains)

IDS = "data/absolut/antigen_ids.txt"
OUT = "results/exp67_epitope_precheck.csv"
MIN_CONTACT_RES = 5


def main():
    lib = AL.load()
    ids = [x.strip() for x in io.open(IDS, encoding="utf-8") if x.strip()]
    rows = []
    for i, ag in enumerate(ids, 1):
        pdb = pdb_of(ag)
        ch = ag.split("_")[1]
        v = lib.get(ag)
        r = {"antigen": ag, "pdb": pdb, "chains": ch, "verdict": "",
             "obs_n": 0, "ag_n": len(v["seq"]) if v else 0,
             "ab_chains": "", "epitope_n": 0, "detail": ""}
        if not v:
            r["verdict"] = "정의표 없음"
            rows.append(r)
            continue
        try:
            al = atoms(pdb)
        except Exception as exc:
            r["verdict"] = "좌표 실패"
            r["detail"] = type(exc).__name__
            rows.append(r)
            continue
        if not al:
            r["verdict"] = "좌표 없음"
            rows.append(r)
            continue
        # 이름이 가리키는 사슬을 먼저 보되, 안 맞으면 서열로 찾는다.
        # 2WUC_I는 이름의 사슬 I에 잔기가 셋뿐이고 실제 항원은 사슬 A다.
        found, frac = resolve_ag_chains(pdb, v["seq"], ch)
        r["identity"] = round(frac, 4)
        if found is None:
            # 이 PDB에 그 항원이 없다는 뜻이고, 그러면 "이 복합체의 항체가 이
            # 항원에 붙는다"는 전제가 서지 않는다. 에피토프 대조뿐 아니라
            # exp62의 백분위도 의미를 잃는다.
            r["verdict"] = "전제 실패"
            r["detail"] = f"항원 서열이 이 PDB에 없음 (최선 동일성 {frac:.3f})"
            rows.append(r)
            continue
        if found != set(ch):
            r["detail"] = f"사슬 {''.join(sorted(found))}에서 찾음 (이름은 {ch})"
        ch = "".join(sorted(found))
        r["chains"] = ch
        obs, _, _ = residues(al, set(ch))
        r["obs_n"] = len(obs)
        keep = [c for c, _ in antibody_chains(pdb, ch, MIN_CONTACT_RES)]
        if not keep:
            # 기술적 실패가 아니라 전제 실패다. 이 복합체에 항체는 있는데 그
            # 항체가 이 항원에 붙어 있지 않다는 뜻이기 때문이다.
            #
            #   4K24_A  항체 이름이 anti-uPAR인데 Absolut은 사슬 A(uPA)를 항원으로
            #           썼다. 항체는 사슬 U(uPAR)에 붙는다
            #   4K3J_A  Onartuzumab은 사슬 B(HGF 베타)에 붙는데 항원은 사슬 A다
            #
            # 관문 3도 서열 대조도 통과한다 -- 항체는 멀쩡히 있고 항원 서열도 그
            # PDB에 있다. 접촉을 봐야만 드러난다. 그래서 exp62의 백분위도 여기서는
            # 관계없는 항체를 재는 것이 된다.
            # 항체가 무엇에 붙어 있는지 찾아 적는다. 후보에서 항체 사슬 자신은
            # 뺀다 -- 넣으면 중쇄가 경쇄에 붙어 있는 것을 집는다.
            ab_all = {c for c in {x[0] for x in al}
                      if is_antibody_chain(al, c)}
            near = ""
            for t in sorted({x[0] for x in al} - set(ch) - ab_all):
                partner = [c for c, _ in antibody_chains(pdb, t,
                                                         MIN_CONTACT_RES)]
                if not partner:
                    continue
                _, h2 = epitope(pdb, {t}, "".join(partner))
                if h2:
                    near = f" · 항체는 사슬 {t}에 붙어 있음({len(h2)}잔기)"
                    break
            r["verdict"] = "전제 실패"
            r["detail"] = "항체가 이 항원에 붙어 있지 않음" + near
            rows.append(r)
            continue
        _, hit = epitope(pdb, set(ch), "".join(keep))
        r.update(verdict="통과", ab_chains="".join(keep), epitope_n=len(hit))
        rows.append(r)
        if i % 10 == 0:
            print(f"  {i}/{len(ids)}", flush=True)
        time.sleep(0.2)          # 공개 서비스다. 줄지어 두드리지 않는다

    d = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    d.to_csv(OUT, index=False, encoding="utf-8")
    print()
    print(d.verdict.value_counts().to_string())
    ok = d[d.verdict == "통과"]
    if len(ok):
        print(f"\n에피토프 크기 중앙 {ok.epitope_n.median():.0f}잔기 "
              f"({ok.epitope_n.min()}~{ok.epitope_n.max()})")
    bad = d[d.verdict != "통과"]
    if len(bad):
        print("\n성립하지 않는 항원")
        for _, r in bad.iterrows():
            print(f"  {r.antigen:10} {r.verdict:16} {r.detail}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
