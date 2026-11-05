"""PDB 좌표에서 결정 구조의 에피토프를 낸다.

에피토프 대조의 '정답' 쪽이다. Absolut의 hotspots는 쓸 수 없다 -- 그건 격자에
결합한 서열들이 공통으로 닿는 자리를 군집화한 값이라, 격자를 격자로 검증하는
셈이 된다. 정답은 결정 구조에서 와야 한다.

번호를 잇는 문제가 남는다. Absolut의 agSeq는 대개 항원 사슬의 **관측된** 잔기를
순서대로 이어 붙인 것과 같다 -- 1FBI_X는 사슬 X의 관측 잔기 129개가 agSeq와
문자 단위로 동일하다. 처음에 표본 다섯을 보고 "항상 같다"로 넘어갈 뻔했는데,
103종으로 넓히니 열 종이 어긋났다.

원인이 한 가지가 아니었다. 관측 끝 잔기를 버린 항원(1OB1_C), 삽입 코드가 붙어
한 칸 밀린 항원(3BN9_A는 잔기 60에 60A~60G가 붙어 있다), 대체 PDB 항목이
variant 서열인 항원(3RVV->5VPL). 그래서 원인별로 보정하지 않고 정렬로 잇는다
(`map_agseq_to_obs`). 잘못 이으면 겹침이 통째로 엉뚱해지므로 동일성으로
스스로 거부한다.

이름도 믿지 않는다. 2WUC_I의 사슬 I에는 잔기가 셋뿐이고 실제 항원 238잔기는
사슬 A에 있다(`resolve_ag_chains`). 어느 사슬에도 없으면 그 복합체에 그 항원이
없다는 뜻이고, 그러면 "이 복합체의 항체가 이 항원에 붙는다"는 전제가 서지
않는다 -- exp62의 백분위도 그때는 다른 것을 재게 된다.

    python src/pdb_contacts.py 1FBI_X
"""
import io
import os
import re
import subprocess
import sys

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
       "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
       "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
CIF_DIR = "data/absolut/pdb"
CUTOFF = 4.5                    # 접촉으로 볼 무거운 원자 사이 거리 (A)


def fetch(pdb):
    p = f"{CIF_DIR}/{pdb}.cif"
    if not os.path.exists(p) or os.path.getsize(p) < 10000:
        os.makedirs(CIF_DIR, exist_ok=True)
        subprocess.run(["curl", "-sL", "--max-time", "180", "-o", p,
                        f"https://files.rcsb.org/download/{pdb}.cif"],
                       check=False)
    return p if os.path.exists(p) and os.getenv("SKIP") is None else p


_ATOM_CACHE = {}


def atoms(pdb):
    """(사슬, 잔기키, 잔기이름, x, y, z)를 순서대로 낸다.

    mmCIF의 _atom_site 열 순서는 파일마다 다를 수 있으므로 머리글에서 읽는다.
    수소는 버린다 -- 결정 구조에 대개 없고, 있으면 접촉 거리를 흐린다.

    같은 PDB를 한 항원에서 열 번 넘게 부른다(사슬을 고를 때 사슬마다 접촉을
    세므로). 파일마다 수천~수만 줄이라 다시 읽을 이유가 없어 기억해 둔다.
    """
    if pdb in _ATOM_CACHE:
        return _ATOM_CACHE[pdb]
    p = fetch(pdb)
    cols, out, inloop, i = [], [], False, {}
    for line in io.open(p, encoding="utf-8", errors="replace"):
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".", 1)[1])
            inloop = True
            continue
        if not inloop:
            continue
        if line.startswith("#"):
            break
        if not i:
            # 열 색인은 한 번만 만든다. 원자 줄마다 다시 만들면 파일 하나에
            # 수십만 번 딕셔너리를 짓는 셈이 된다.
            i = {c: k for k, c in enumerate(cols)}
        r = line.split()
        if len(r) < len(cols):
            continue
        if r[i["group_PDB"]] != "ATOM":
            continue
        if r[i.get("type_symbol", 0)] == "H":
            continue
        # 여러 벌(altloc)이 있으면 첫 벌만 본다
        alt = r[i["label_alt_id"]] if "label_alt_id" in i else "."
        if alt not in (".", "?", "A"):
            continue
        ins = r[i["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in i else "."
        try:
            xyz = (float(r[i["Cartn_x"]]), float(r[i["Cartn_y"]]),
                   float(r[i["Cartn_z"]]))
        except (ValueError, KeyError):
            continue
        out.append((r[i["auth_asym_id"]], (r[i["auth_seq_id"]], ins),
                    r[i["label_comp_id"]], xyz))
    # 구조 하나가 수만 원자라 메모리가 걱정되지만, 항원 목록이 103종이고
    # 한 번에 한 PDB만 반복해서 쓰므로 직전 것 몇 개만 들고 있으면 된다.
    if len(_ATOM_CACHE) > 4:
        _ATOM_CACHE.clear()
    _ATOM_CACHE[pdb] = out
    return out


def residues(atom_list, chains):
    """주어진 사슬들의 관측 잔기를 순서대로 낸다.

    반환: (한 글자 서열, [잔기키...], {잔기키: [좌표...]})
    """
    order, seq, coords = [], [], {}
    for ch, key, name, xyz in atom_list:
        if ch not in chains:
            continue
        k = (ch, key)
        if k not in coords:
            coords[k] = []
            order.append(k)
            seq.append(AA3.get(name, "X"))
        coords[k].append(xyz)
    return "".join(seq), order, coords


def map_agseq_to_obs(agseq, obs, min_identity=0.98):
    """agSeq 색인 -> 관측 잔기 색인. 이을 수 없으면 (None, 비율).

    둘은 대개 문자 단위로 같지만(103종 중 93종) 어긋나는 자리가 있다. 원인이
    한 가지가 아니다.

        1OB1_C, 1QFW_AB   관측 끝 잔기 하나를 Absolut이 버렸다
        3BN9_A 외 4종     삽입 코드. 잔기 60에 60A~60G가 붙어 있고 그중 하나가
                          빠졌다. 어느 것을 빼는지는 규칙으로 적혀 있지 않다

    그래서 원인을 추정해 보정하지 않고 정렬로 잇는다. 어느 쪽이 빠졌든 맞고,
    잘못 이으면 겹침이 통째로 엉뚱해지므로 동일성 비율로 스스로 거부한다.
    대응되지 않는 agSeq 잔기는 지도에서 빠진다 -- 그 자리는 겹침 계산에
    들어가지 않으니 조용히 틀리는 대신 빠지는 쪽이다.
    """
    import difflib
    if not agseq or not obs:
        return None, 0.0
    sm = difflib.SequenceMatcher(a=agseq, b=obs, autojunk=False)
    m, matched = {}, 0
    for a0, b0, n in sm.get_matching_blocks():
        for k in range(n):
            m[a0 + k] = b0 + k
        matched += n
    frac = matched / len(agseq)
    return (m, frac) if frac >= min_identity else (None, frac)


def resolve_ag_chains(pdb, agseq, named, min_identity=0.98):
    """항원 서열이 실제로 어느 사슬에 있는지 찾는다.

    이름이 가리키는 사슬을 먼저 본다. 대개 맞고, 여러 사슬이 이어 붙은 항원
    (3CVH_ABC는 사슬 셋)은 이름으로만 풀린다.

    안 맞으면 사슬을 하나씩 본다. 이름이 틀린 경우가 있다 -- 2WUC_I의 사슬 I에는
    잔기가 셋뿐이고(KQL) agSeq 238잔기는 사슬 A와 동일성 1.000으로 맞는다.
    이름을 믿고 넘어가면 그 항원은 '대응 불일치'로 빠지는데, 실제로는 멀쩡하다.

    어디에도 없으면 (None, 최선 동일성)을 낸다. 그때는 이름이 가리키는 복합체에
    그 항원이 없다는 뜻이고, 그러면 "이 복합체의 항체가 이 항원에 붙는다"는
    전제 자체가 서지 않는다 -- 3L5W_I가 그렇다(어느 사슬과도 0.434 이하).
    """
    al = atoms(pdb)
    obs, _, _ = residues(al, set(named))
    mp, frac = map_agseq_to_obs(agseq, obs, min_identity)
    if mp is not None:
        return set(named), frac
    best = (None, frac)
    for c in sorted({ch for ch, _, _, _ in al}):
        o, _, _ = residues(al, {c})
        m, f = map_agseq_to_obs(agseq, o, min_identity)
        if m is not None and f > best[1]:
            best = ({c}, f)
        elif f > best[1]:
            best = (best[0], f)
    return best


def is_antibody_chain(atom_list, ch):
    """이 사슬이 항체 가변부를 갖는가.

    가변부 끝 자리(85~150)에 C...WGxG(중쇄) 또는 C...FGxG(경쇄)가 있으면 그렇게
    본다. 관문 3이 중쇄를 고를 때 쓰는 기준과 같다.
    """
    seq, _, _ = residues(atom_list, {ch})
    return any(85 <= m.end() - 4 <= 150
               for m in re.finditer(r"C[A-Z]{3,35}?[WF]G.G", seq))


def antibody_chains(pdb, ag_chains, min_contact=5):
    """이 항원에 실제로 붙어 있는 '항체' 사슬을 고른다.

    두 가지를 함께 본다.

      서명   가변부 끝 자리(85~150)에 C...WGxG(중쇄) 또는 C...FGxG(경쇄)가 있다.
             접촉만으로 고르면 항체가 아닌 것이 섞인다 -- 2WUC에는 사슬
             B(ACGRRHK, 7잔기)와 I(KQL, 3잔기)가 항원에 붙어 있는데 펩타이드다.
             그것까지 넣으면 '에피토프'에 항체와 무관한 접촉이 들어가고, 겹침이
             부풀어 결과가 좋아 보이는 쪽으로 틀린다.
      접촉   결정에 복합체가 두 벌 든 일이 흔하다(1FBI는 Fab가 H/L과 Q/P로
             두 벌). 이 항원 사슬에 실제로 붙은 쪽만 그 항원의 짝이다.

    기준은 관문 3이 중쇄를 고를 때 쓰는 것과 같다. 여기서 따로 만들면 두 곳이
    조용히 갈린다.
    """
    al = atoms(pdb)
    ag = set(ag_chains)
    keep = []
    for ch in sorted({c for c, _, _, _ in al} - ag):
        if not is_antibody_chain(al, ch):
            continue
        _, hit = epitope(pdb, ag, ch)
        if len(hit) >= min_contact:
            keep.append((ch, len(hit)))
    return keep


def epitope(pdb, ag_chains, ab_chains, cutoff=CUTOFF):
    """항체 사슬에서 cutoff 안에 있는 항원 잔기의 순번(0부터)을 낸다.

    순번은 관측 순서 기준이므로 agSeq의 색인과 같다 -- 같은지는 부르는 쪽이
    대조한다.
    """
    al = atoms(pdb)
    seq, order, coords = residues(al, set(ag_chains))
    ab = [xyz for ch, key, name, xyz in al if ch in set(ab_chains)]
    if not ab or not order:
        return seq, {}
    c2 = cutoff * cutoff
    hit = {}
    # 격자 해싱. 전부 대 전부로 하면 항원 2천 원자 x 항체 3천 원자가 된다.
    cell = cutoff
    grid = {}
    for x, y, z in ab:
        grid.setdefault((int(x // cell), int(y // cell), int(z // cell)),
                        []).append((x, y, z))
    for idx, k in enumerate(order):
        n = 0
        for x, y, z in coords[k]:
            gx, gy, gz = int(x // cell), int(y // cell), int(z // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for (a, b, c) in grid.get((gx + dx, gy + dy,
                                                   gz + dz), ()):
                            if ((a - x) ** 2 + (b - y) ** 2
                                    + (c - z) ** 2) <= c2:
                                n += 1
        if n:
            hit[idx] = n
    return seq, hit


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    sys.path.insert(0, "src")
    import antigen_lib as A
    ag_id = sys.argv[1] if len(sys.argv) > 1 else "1FBI_X"
    pdb, chains = ag_id.split("_")[0], ag_id.split("_")[1]
    v = A.load().get(ag_id)
    seq, hit = epitope(pdb, chains, "")
    print(f"{ag_id}  관측 {len(seq)}잔기 · agSeq "
          f"{len(v['seq']) if v else '?'} · 동일 "
          f"{bool(v) and seq == v['seq']}")


if __name__ == "__main__":
    main()
