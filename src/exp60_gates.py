"""exp60의 항원별 관문 — 조용히 틀린 결과가 통과하지 못하게 한다.

103종을 무인으로 돌리려면, 실패가 실패로 보여야 한다. 이 프로젝트에서 가장
비쌌던 오류는 예외를 던진 것이 아니라 조용히 반쪽만 성공한 것들이었다.
exp15에서 Absolut! 채점이 부분 실패했는데 알아채지 못해 1NCA_N의 모항체
에너지가 공개값과 13이나 어긋난 채로 라이브러리가 만들어졌고, 한참 뒤에
손으로 발견했다. 그런 것을 항원마다 자동으로 막는다.

다섯 관문이고, 하나라도 걸리면 그 항원은 격리 명부로 간다. 격리는 실패가
아니라 보류다 -- 재시도 후에도 걸리면 이유와 함께 남기고, 통과한 항원만
결과에 들어간다.

  1 전송     Content-Length와 md5가 서버 값과 일치
  2 아카이브  zip 중앙 디렉터리 검사 통과, 푼 파일이 구조 파일 형식
  3 귀속     CDR-H3를 항원 사슬이 아니라 검증된 중쇄에서 뽑았는가
  4 채점     빠진 값 없음, 값이 물리적으로 그럴듯함, 재실행 시 같은 값
  5 격리     실패는 별도 명부에 이유와 함께

3번이 실제 위험이다. 항원도 단백질이므로 C...WGxG 모티프를 우연히 가질 수
있고, 그러면 항원 조각을 '실제 결합 항체'로 착각해 채점하게 된다. 그림도
숫자도 멀쩡해 보이므로 검사 없이는 잡히지 않는다.
"""
import hashlib
import os
import re
import subprocess

import numpy as np

SRC = "tools/Absolut-main/src"
WSL_SRC = "/mnt/d/ubuntu/BIO/Infectious/tools/Absolut-main/src"

# 격자 접촉 에너지가 실제로 놓이는 범위. 밖으로 나가면 잘못 읽은 것이다.
E_LO, E_HI = -260.0, -20.0

# PDB에서 철회된 항목 -> 대체 항목. Absolut의 항원 목록은 2021년 것이라
# 그 뒤에 철회된 ID를 그대로 들고 있다. 옛 ID로 조회하면 FASTA가 빈 채로
# 돌아오고, 그러면 '중쇄 후보 없음'이 되어 항원 하나가 조용히 분모에서 빠진다.
#
#   3RVV  2022-04-27 철회, 5VPL로 대체. "Der f 1 complexed with Fab 4C1"
#         5VPL에서도 항원은 사슬 A라 사슬 제외 규칙이 그대로 맞고, 중쇄는
#         4C1 그대로다(CGRTGVYRYPERAPYW, FR4@110).
#
# 항체 서열만 여기서 가져온다. 격자와 항원 쪽은 Absolut이 옛 좌표로 만든
# 자기 파일을 그대로 쓰므로 바뀌지 않는다.
PDB_REPLACED = {"3RVV": "5VPL"}


def pdb_of(antigen_id):
    """항원 ID에서 조회할 PDB ID를 낸다. 철회된 것은 대체 항목으로 바꾼다."""
    p = str(antigen_id).split("_")[0]
    return PDB_REPLACED.get(p, p)


def md5_of(path, chunk=1 << 20):
    d = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            d.update(blk)
    return d.hexdigest()


def gate_transfer(path, want_size, want_md5):
    """관문 1 — 받은 것이 서버에 있는 것과 같은가."""
    if not os.path.exists(path):
        return False, "파일 없음"
    got = os.path.getsize(path)
    if want_size and got != want_size:
        return False, f"크기 {got:,}/{want_size:,}"
    if want_md5:
        h = md5_of(path)
        if h != want_md5:
            return False, f"md5 {h[:12]}/{want_md5[:12]}"
    return True, f"{got:,}B md5 확인"


def gate_archive(zip_name, txt_name):
    """관문 2 — 검사하고, 풀고, 푼 것을 확인한다. 셋은 따로다.

    처음 판은 `unzip -tq`로 검사만 하고 `-oq`로 푸는 것을 빠뜨렸다. 검사는
    통과하는데 파일이 생기지 않으니 바로 다음 줄에서 "압축 해제 후 파일 없음"이
    되고, 새 항원이 전부 격리됐다. 검사와 추출은 다른 동작이라 둘 다 호출하고
    반환 코드도 각각 본다.
    """
    t = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc",
                        f"cd {WSL_SRC} && unzip -tq '{zip_name}'"],
                       capture_output=True, text=True)
    if t.returncode != 0:
        return False, "zip 중앙 디렉터리 검사 실패"
    x = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc",
                        f"cd {WSL_SRC} && unzip -oq '{zip_name}'"],
                       capture_output=True, text=True)
    if x.returncode != 0:
        return False, f"압축 해제 실패 (코드 {x.returncode}) {x.stderr[-80:]}"
    p = f"{SRC}/{txt_name}"
    if not os.path.exists(p):
        return False, "압축 해제했으나 기대한 파일이 없음"
    n = os.path.getsize(p)
    if n < 1000:
        return False, f"구조 파일이 비어 있다시피 하다 ({n:,}B)"
    # 실제 형식을 보고 맞춘 규칙이다. 처음 판은 "첫 줄이 숫자로 시작한다"고
    # 가정했는데 그건 셋째 줄부터의 모양이고, 머리 두 줄은 Ligand=/Receptors=
    # 로 시작한다. 그 가정 때문에 멀쩡한 파일이 격리됐다.
    #
    #   Ligand=<탭>133152<탭>SURDLRSULDD...     리간드 길이와 이동열
    #   Receptors=<탭>10310432                  구조 개수
    #   120675<탭>SSSLSUSLSU                    시작 위치와 이동열
    #
    # 이동열에는 '+'가 들어갈 수 있고, 그것이 두 번째로 정상 파일을 막았다.
    # 가진 구조 파일 17개의 알파벳을 뽑아 보면 갈래가 뚜렷하다.
    #
    #   이름이 이동열로 시작(SUDDL..., SULSU...)   DLRSU        단일 사슬
    #   이름이 해시로 시작(213f7e..., 1d2e6d...)   DLRSU + B +  다중 분절
    #
    # '+'는 분절 구분자다. 다중 사슬 항원이 해시 이름을 갖는 것과 짝이 맞는다.
    # 그래서 '+'를 뺀 규칙은 다중 사슬 항원만 골라 격리한다 -- 기술적 결함이
    # 항원의 성질과 상관된 채로 분모를 깎으므로, 중앙값이 조용히 편향된다.
    # 추측으로 넓히지 않도록 분절마다 비어 있지 않은지도 같이 본다.
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        head = [fh.readline().rstrip("\n") for _ in range(3)]
    m = re.match(r"^Ligand=\s*\d+\s+([SULDRBF+]+)\s*$", head[0])
    if not m:
        return False, f"첫 줄이 Ligand= 형식이 아님: {head[0][:46]!r}"
    seg = m.group(1).split("+")
    if not all(seg):
        return False, f"리간드 분절이 비어 있음 ({len(seg)}분절): {head[0][:46]!r}"
    if not re.match(r"^Receptors=\s*\d+\s*$", head[1]):
        return False, f"둘째 줄이 Receptors= 형식이 아님: {head[1][:46]!r}"
    if not re.match(r"^\d+\s+[SULDRBF]+\s*$", head[2]):
        return False, f"구조 행 형식이 아님: {head[2][:46]!r}"
    n_struct = int(re.search(r"\d+", head[1]).group())
    # 절단은 크기가 아니라 여기서 잡는다. 처음 판은 "1MB보다 작으면 잘린 것"으로
    # 갈랐는데, 그 임계값은 항원 크기와 아무 관계가 없다. 4N9G_C는 항원이
    # 35잔기뿐이라 구조가 27,491개밖에 없고 파일이 495KB다 -- 멀쩡한데 막혔다.
    #
    # 진짜 검사는 머리글이 선언한 구조 수와 실제 줄 수가 맞는가다. 잘린 파일은
    # 반드시 여기서 어긋나고, 작은 항원은 어긋나지 않는다. 497MB 파일도 0.8초면
    # 센다.
    with open(p, "rb") as fh:
        fh.readline()
        fh.readline()
        rows = sum(1 for _ in fh)
    if rows != n_struct:
        return False, (f"구조 수가 맞지 않는다 — 선언 {n_struct:,} 대 "
                       f"실제 {rows:,} (잘렸을 수 있음)")
    return True, f"{n / 1e6:.0f}MB · 구조 {n_struct:,}개 · 수·형식 확인"


def auth_chains(header):
    """FASTA 머리글에서 실제(auth) 사슬 기호를 뽑는다.

    RCSB는 `Chain C[auth H]`처럼 적고, 여러 사슬이면 `Chains A, B`가 된다.
    Absolut!의 항원 이름(1ADQ_A)이 가리키는 것은 auth 쪽이므로 그것을 쓴다.

    괄호는 사슬마다 따로 붙고, 붙지 않은 사슬은 라벨이 곧 auth 기호다.

        Chains C, F[auth P]     -> {C, P}
        Chains A[auth E], D[auth H] -> {E, H}
        Chains A, B             -> {A, B}

    처음 판은 괄호가 하나라도 있으면 괄호 친 것만 돌려주었다. 그래서 위 첫
    줄에서 C가 사라졌다. 표기만 틀리는 게 아니다 -- 항원 개체가 그런 모양이면
    (`Chains A, B[auth X]`) 항원 사슬 A가 집합에서 빠지고, 항원 제외가 통째로
    헛돈다. 항원도 단백질이라 C...WGxG를 우연히 가질 수 있으므로, 관문 3이
    막으려던 바로 그 일이 일어난다.
    """
    seg = header.split("|")[1] if "|" in header else header
    seg = re.sub(r"^\s*Chains?\s+", "", seg.strip())
    out = set()
    for tok in seg.split(","):
        m = re.search(r"\[auth\s+([A-Za-z0-9]+)\]", tok)
        if m:
            out.add(m.group(1))
            continue
        t = re.sub(r"\[.*?\]", "", tok).strip()
        if t:
            out.add(t)
    return out


def gate_chain(fasta_text, antigen_id):
    """관문 3 — 항원 사슬이 아니라 검증된 중쇄에서 CDR-H3를 뽑는다.

    항원도 단백질이라 C...WGxG를 우연히 가질 수 있다. 그래서 (a) 항원 사슬을
    먼저 제외하고, (b) 경쇄를 배제하고, (c) FR4가 가변부 끝 자리에 있는 것만
    받고, (d) 남은 후보가 하나일 때만 통과시킨다.

    보존 Cys 앞의 모티프는 YYC로 굳어 있지 않다. 1NCA의 중쇄는 FFC이고
    그것 때문에 처음 판은 이 항원을 통째로 놓쳤다. 그래서 앞 두 자리를
    강제하지 않고, 대신 FR4의 위치로 진짜 CDR-H3를 가린다 -- 가변부는
    100잔기 남짓이므로 WGxG가 그 근처에 와야 한다.
    """
    want = set(re.split(r"[_]", antigen_id)[1]) if "_" in antigen_id else set()
    hits = []
    for blk in fasta_text.split(">")[1:]:
        if "\n" not in blk:
            continue
        header, seq = blk.split("\n", 1)
        seq = "".join(seq.split())
        ch = auth_chains(header)
        if want and (ch & want):
            continue                                   # 항원 사슬 제외
        desc = (header.split("|")[2] if header.count("|") >= 2 else "").lower()
        best = None
        for m in re.finditer(r"C([A-Z]{3,35}?)WG.G", seq):
            fr4 = m.end() - 4                          # WGxG가 시작하는 자리
            if not (85 <= fr4 <= 150):
                continue                # 가변부 밖이면 CDR-H3가 아니다
            best = m                    # 조건을 만족하는 마지막 것
        if best is None:
            continue
        hits.append({"chain": "".join(sorted(ch)) or "?",
                     "desc": desc[:44],
                     "cdrh3": "C" + best.group(1) + "W",
                     "fr4_at": best.end() - 4,
                     "says_heavy": "heavy" in desc and "light" not in desc,
                     "says_light": "light" in desc and "heavy" not in desc})
    if not hits:
        # 왜 없는지를 갈라서 적는다. 사유가 뭉개지면 격리 분포를 볼 때 패턴이
        # 가려지고, 편향 신호인지 아닌지를 판단할 수 없다.
        #
        # 1TQB가 그런 경우다. 중쇄도 있고 WGQG@102도 제자리인데 그 앞의 보존
        # Cys가 없다 -- 서열이 YYC/YFC가 아니라 'TATYFFTRGTDYWGQG'다. Absolut의
        # C...W 관례로는 표현할 수 없는 항체이므로 빼는 것은 맞지만, 그것을
        # '중쇄 후보 없음'이라 적으면 중쇄가 아예 없는 항원과 같은 칸에 들어간다.
        for blk in fasta_text.split(">")[1:]:
            if "\n" not in blk:
                continue
            header, seq = blk.split("\n", 1)
            seq = "".join(seq.split())
            if want and (auth_chains(header) & want):
                continue
            for m in re.finditer(r"WG.G", seq):
                if 85 <= m.start() <= 150:
                    return None, (f"중쇄는 있으나 보존 Cys가 없음 "
                                  f"(FR4@{m.start()}, 앞 12잔기 "
                                  f"{seq[max(0, m.start() - 12):m.start()]})")
        return None, "중쇄 후보 없음"
    uniq = {h["cdrh3"] for h in hits}
    if len(uniq) > 1:
        # 후보가 갈릴 때만 주석을 쓴다. 가르는 데는 쓸모가 있지만, 서열이 이미
        # 말한 것을 뒤집게 두지는 않는다.
        pref = [h for h in hits if h["says_heavy"]]
        if len({h["cdrh3"] for h in pref}) == 1:
            hits, uniq = pref, {pref[0]["cdrh3"]}
        else:
            return None, "중쇄 후보가 둘 이상: " + " / ".join(
                f"{h['chain']}:{h['cdrh3']}" for h in hits)
    h = hits[0]
    # PDB 주석이 뒤바뀐 항목이 있다. 1OB1은 'ANTIBODY, LIGHT CHAIN'이라 적힌
    # 개체가 WGxG@109와 CH1 끝(...KVDKKI)을 갖고, 'HEAVY CHAIN'이라 적힌 쪽이
    # FGxG@98과 카파 불변부 끝(...SFNRNEC)을 갖는다. 주석으로 경쇄를 먼저
    # 걷어내던 판은 그래서 진짜 중쇄를 버리고 '중쇄 후보 없음'을 냈다.
    # 이제는 서열로 가른다 -- 중쇄 FR4는 WGxG, 경쇄는 FGxG다.
    note = " · 주석은 경쇄라 하지만 서열은 중쇄" if h["says_light"] else ""
    return h, f"{h['chain']} FR4@{h['fr4_at']} ({h['desc']}){note}"


def gate_scores(e_nat, e_ref, expect_n):
    """관문 4 — 빠진 값 없이, 물리적으로 그럴듯한 값이 나왔는가."""
    miss = int(np.isnan(e_ref).sum()) + int(np.isnan(e_nat))
    if miss:
        return False, f"채점 누락 {miss}/{expect_n + 1}"
    if not (E_LO < e_nat < E_HI):
        return False, f"모항체 에너지가 범위 밖 {e_nat:.2f}"
    if not (E_LO < np.min(e_ref) and np.max(e_ref) < E_HI):
        return False, (f"비교 집합이 범위 밖 "
                       f"{np.min(e_ref):.2f}~{np.max(e_ref):.2f}")
    if np.std(e_ref) < 1.0:
        # 같은 값만 돌아오면 구조 파일이 엉뚱하거나 채점이 실패한 것이다
        return False, f"비교 집합이 축퇴 (표준편차 {np.std(e_ref):.3f})"
    return True, f"누락 0 · {np.min(e_ref):.1f}~{np.max(e_ref):.1f}"


def gate_repeat(oracle_cls, antigen, seqs, energies, threads=4, k=20):
    """관문 4-2 — 다시 돌리면 같은 값이 나오는가.

    항원을 잘못 올렸거나 구조 파일이 섞였을 때, 값 자체는 그럴듯하지만 두 번째
    실행과 어긋난다. 20개만 다시 재서 정확히 일치하는지 본다.
    """
    idx = np.linspace(0, len(seqs) - 1, k, dtype=int)
    o = oracle_cls(antigen, [seqs[i] for i in idx], threads, f"rep_{antigen}")
    o(np.arange(len(idx)))
    d = np.abs(o.energy - energies[idx])
    if np.nanmax(d) > 1e-6:
        return False, f"재실행 불일치 최대 {np.nanmax(d):.4f}"
    return True, f"{k}개 재실행 일치"
