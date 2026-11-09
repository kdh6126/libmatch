"""보충표의 순서를 못 박는다.

보충표 번호는 문서에 나타나는 순서로 매겨지는데, 생성기를 돌린 순서대로 끼워
넣으면 그 순서가 그때그때 달라진다. 본문이 "Supplementary Table S9"라고 쓰는
동안 S9가 다른 표가 돼 있으면 심사자가 엉뚱한 표를 본다.

그래서 순서를 여기 한 곳에 적어 두고, 생성기를 몇 번을 돌리든 이 순서로
재배열한다. 표를 추가할 때는 ORDER에 한 줄을 더하면 된다.

    python src/si_order.py          순서를 맞추고 번호를 출력
"""
import io
import re
import sys

SI = "manuscript/supplementary_information.tex"

# 문서에 나타날 순서. 앞쪽의 손으로 쓴 표(S1~S5)는 건드리지 않는다.
ORDER = [
    "cognate disposition",      # S6
    "unmatched seeds",          # S7
    "repr many antigens",       # S8
    "search ci",                # S9
    "integrity check",          # S10
    "dimension",                # S11, S12 (표 두 개)
    "cognate power",            # S13
    "parent choice",            # S14
    "cross antigen",            # S15
    "budget half",              # S16
    "rescore check",            # S17 (Methods 에서 인용)
    "pair sensitivity",         # S18, S19 (표 두 개)
    "zoo transfer",             # S20
    "alpha sensitivity",        # S21 (Methods 에서 인용)
]


def spans(s, name):
    """이 이름을 가진 블록이 나타나는 모든 구간. 여러 벌이 있을 수 있다."""
    A, B = f"% ---- BEGIN {name} (generated)", f"% ---- END {name}"
    out, i = [], 0
    while True:
        a = s.find(A, i)
        if a < 0:
            return out
        b = s.find(B, a)
        if b < 0:
            return out
        out.append((a, b + len(B)))
        i = b + len(B)


def blocks(s):
    """이름별 첫 블록의 본문과, 지워야 할 모든 구간.

    문자열이 같은지로 지우면 안 된다. 앞선 실행의 줄바꿈 정규화 때문에 두 번째
    복사본이 첫 번째와 미세하게 달라지면, 같은 이름의 표가 두 벌 남고 그 위에
    또 한 벌이 붙는다. 실제로 그렇게 생성된 표 여덟 개가 열여섯 개가 됐고,
    컴파일은 멀쩡히 됐다 -- 본문이 인용하지 않는 표가 여덟 개 늘었을 뿐이라
    보지 않으면 모른다. 위치로 지운다.
    """
    out, cuts = {}, []
    for name in ORDER:
        sp = spans(s, name)
        if not sp:
            continue
        a, b = sp[0]
        out[name] = s[a:b]
        cuts.extend(sp)
    return out, sorted(cuts)


def main():
    for x in (sys.stdout, sys.stderr):
        try:
            x.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    s = io.open(SI, encoding="utf-8").read()
    found, cuts = blocks(s)
    if not found:
        sys.exit("생성된 표가 없습니다.")
    dup = len(cuts) - len(found)
    if dup:
        print(f"같은 블록이 여러 벌 들어 있었습니다: {dup}벌을 지웁니다")
    for a, b in reversed(cuts):
        s = s[:a] + s[b:]
    s = re.sub(r"\n{3,}", "\n\n", s)
    joined = "\n".join(found[n] for n in ORDER if n in found)
    s = s.replace("\\end{document}", joined + "\n\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)

    # 번호는 문서 전체에서 table 환경이 나오는 순서다.
    n = 0
    pos = {}
    for m in re.finditer(r"\\begin\{table\}", s):
        n += 1
        pos[m.start()] = n
    print(f"보충표 {n}개")
    for name in ORDER:
        if name not in found:
            print(f"  (없음)  {name}")
            continue
        a = s.find(f"% ---- BEGIN {name} (generated)")
        b = s.find(f"% ---- END {name}")
        nums = [v for k, v in pos.items() if a < k < b]
        label = ", ".join(f"S{v}" for v in nums) or "표 없음"
        print(f"  {label:10s} {name}")
    missing = [n for n in ORDER if n not in found]
    if missing:
        print("아직 생성되지 않은 표:", ", ".join(missing))


if __name__ == "__main__":
    main()
