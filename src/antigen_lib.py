"""Absolut!이 소스에 싣고 다니는 항원 정의표를 읽는다.

tools/Absolut-main/src/Absolut/antigenLib.cpp 안에 항원마다 이렇게 들어 있다.

    if(codeID == str2int("1FBI_X")){
        agStruct = string("SULSUURRUULL...");     격자 이동열
        agSeq    = string("KVFGRCELAAAM...");     항원 아미노산 서열
        startPos = 133152;                        격자 시작 위치
        blockV   = {141280, 145313, ...};         막힌 자리
        hotspotsCore  = { {35,43,53,107}, {15,27,92,98} };
        hotspotsLarge = { {35,43,53,107,108}, {15,27,87,88,91,92,98} };
        antibodyChains = "LH";                    원본 PDB의 항체 사슬
    }

이 표가 주는 것은 셋이다.

  좌표      startPos(또는 다중 사슬의 startPosChains)와 구조 문자열. 이것으로
            항원을 격자에 펼 수 있다. 구조 파일의 Ligand= 줄에는 첫 분절의 시작
            위치만 남으므로(insert()가 그렇게 합친다) 다중 사슬 항원을 제자리에
            놓으려면 반드시 여기를 봐야 한다.

  서열      agSeq. 격자 잔기 번호를 아미노산으로 바로 읽는다.

  사슬      antibodyChains는 원본 복합체의 항체 사슬을 벤치마크가 직접 적어 둔
            것이다. 관문 3은 FASTA 주석과 서열로 중쇄를 고르는 추론인데, 이쪽은
            추론이 아니다. 다만 경쇄와 중쇄를 가르지 않고 58종에만 있으므로
            관문 3을 대신하지는 못하고, 대조하는 데 쓴다.

hotspots는 에피토프의 정답이 **아니다**. epitope.cpp를 보면 격자에 결합한
서열들이 공통으로 닿는 항원 잔기를 군집화해서 얻는다 -- 격자에서 나온 값이다.
그래서 "천연 항체의 격자 자세가 hotspots에 오는가"는 격자가 실제 결합을
반영하는지에 대한 답이 될 수 없다. 격자 자세가 격자 자세들이 모이는 곳에
오는지를 묻는 것이라 거의 동어반복이다.

에피토프 대조의 정답은 결정 구조에서 와야 한다 -- 원본 PDB에서 항체와 4~5A
안에 있는 항원 잔기다. 그것은 따로 계산해야 하고, PDB 잔기 번호를 agSeq 번호로
잇는 정렬도 필요하다(결정 구조에서 보이지 않는 잔기는 격자에 올라가지 않아
번호가 어긋난다). hotspots는 그 대조의 기준이 아니라, 천연 항체가 격자에서
'전형적인 결합체처럼' 행동하는지를 보는 보조 지표로만 쓸 수 있다.

    python src/antigen_lib.py        표를 읽어 요약을 찍는다
"""
import io
import os
import re
import sys

LIB = "tools/Absolut-main/src/Absolut/antigenLib.cpp"


def _ints(s):
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _sets(s):
    """'{ {35,43}, {15,27} }' -> [[35,43],[15,27]]"""
    return [_ints(g) for g in re.findall(r"\{([^{}]*)\}", s)]


def load(path=LIB):
    """항원 ID -> 정의. 파일에 있는 것만 담는다."""
    if not os.path.exists(path):
        return {}
    txt = io.open(path, encoding="utf-8", errors="replace").read()
    out = {}
    # 한 항원의 블록은 다음 str2int("...") 전까지다. 중괄호를 세는 대신
    # 그렇게 자른다 -- 블록 안에 중첩 중괄호(hotspots)가 있어서 간단한
    # 짝맞추기로는 오히려 틀리기 쉽다.
    marks = [(m.start(), m.group(1))
             for m in re.finditer(r'str2int\("([A-Za-z0-9_]+)"\)', txt)]
    for i, (pos, ag) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(txt)
        blk = txt[pos:end]
        d = {}
        # 단일 사슬은 agStruct = string("..."), 다중 사슬은 agSeq = "..."
        # 처럼 string() 없이 적힌 자리가 섞여 있다. 둘 다 받는다.
        m = re.search(r'agStruct\s*=\s*(?:string\()?"([^"]*)"', blk)
        if m:
            d["struct"] = m.group(1)
            d["starts"] = None                 # startPos에서 채운다
        else:
            # 다중 사슬 항원은 형식이 다르다. 이쪽이 분절별 시작 위치를 갖는다 --
            # 구조 파일의 Ligand= 줄에는 첫 분절 것만 남으므로(insert()가
            # 첫 시작 위치만 보존한다) 분절을 제자리에 놓으려면 여기를 봐야 한다.
            ms = re.search(r"startPosChains\s*=\s*\{([^}]*)\}", blk)
            mc = re.search(r"structChains\s*=\s*\{(.*?)\}\s*;", blk, re.S)
            if ms and mc:
                parts = re.findall(r'"([^"]*)"', mc.group(1))
                d["struct"] = "+".join(parts)
                d["starts"] = _ints(ms.group(1))
        m = re.search(r'agSeq\s*=\s*(?:string\()?"([^"]*)"', blk)
        if m:
            d["seq"] = m.group(1)
        m = re.search(r"startPos\s*=\s*(-?\d+)", blk)
        if m and d.get("starts") is None:
            d["starts"] = [int(m.group(1))]
        m = re.search(r"blockV\s*=\s*\{([^;]*)\}\s*;", blk)
        if m:
            d["block"] = _ints(m.group(1))
        m = re.search(r"hotspotsCore\s*=\s*\{(.*?)\}\s*;", blk, re.S)
        if m:
            d["core"] = _sets(m.group(1))
        m = re.search(r"hotspotsLarge\s*=\s*\{(.*?)\}\s*;", blk, re.S)
        if m:
            d["large"] = _sets(m.group(1))
        m = re.search(r'antibodyChains\s*=\s*"([^"]*)"', blk)
        if m:
            d["ab_chains"] = m.group(1)
        if d.get("struct") and d.get("seq") and d.get("starts"):
            # 같은 항원이 여러 번 나오면 마지막 것이 유효하다고 보지 않고
            # 처음 것을 지킨다. 뒤쪽은 변이체 정의일 수 있다.
            out.setdefault(ag, d)
    return out


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    d = load()
    print(f"항원 정의 {len(d)}개")
    if not d:
        return
    n_core = sum(1 for v in d.values() if v.get("core"))
    n_ch = sum(1 for v in d.values() if v.get("ab_chains"))
    n_seg = sum(1 for v in d.values() if "+" in v["struct"])
    print(f"  에피토프(hotspotsCore) 있음  {n_core}")
    print(f"  항체 사슬 표기 있음          {n_ch}")
    print(f"  다중 분절 구조               {n_seg}")
    print()
    ids = [x.strip() for x in io.open("data/absolut/antigen_ids.txt",
                                      encoding="utf-8") if x.strip()]
    miss = [x for x in ids if x not in d]
    print(f"우리 목록 {len(ids)}종 중 정의표에 없는 것: {len(miss)}"
          + (" — " + ", ".join(miss) if miss else ""))
    print()
    for ag in ids[:4]:
        v = d.get(ag)
        if not v:
            continue
        print(f"{ag}  잔기 {len(v['seq'])} · 이동 {len(v['struct'])} · "
              f"사슬 {v.get('ab_chains', '?')} · core {v.get('core')}")


if __name__ == "__main__":
    main()
