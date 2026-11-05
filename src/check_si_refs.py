"""본문이 인용하는 보충표·보충그림 번호가 실제와 맞는가.

표를 새로 넣거나 순서를 바꾸면 번호가 밀린다. 본문은 "Supplementary Table S9"
라고 쓰는데 S9가 다른 표가 돼 있으면 심사자가 엉뚱한 표를 보고, 그 오류는
컴파일로는 잡히지 않는다 -- 보충자료는 별도 문서라 LaTeX의 상호참조가 걸리지
않기 때문이다.

그래서 여기서 대조한다. 보충자료에서 table 환경이 나오는 순서대로 번호를 매기고
각 표의 캡션 첫 부분을 뽑아, 본문이 인용한 번호 옆에 나란히 찍는다. 맞는지는
사람이 보는 편이 빠르다.

    python src/check_si_refs.py
"""
import io
import re
import sys

MAIN = "manuscript/manuscript.tex"
SI = "manuscript/supplementary_information.tex"


def captions(s, env):
    """문서에 나오는 순서대로 (번호, 캡션 첫 문장)."""
    out = []
    for m in re.finditer(r"\\begin\{" + env + r"\}", s):
        seg = s[m.start():]
        c = re.search(r"\\caption\{", seg)
        if not c:
            out.append("(캡션 없음)")
            continue
        i, depth = c.end(), 1
        while i < len(seg) and depth:
            depth += (seg[i] == "{") - (seg[i] == "}")
            i += 1
        text = re.sub(r"\s+", " ", seg[c.end():i - 1])
        text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
        out.append(text.strip()[:90])
    return out


def main():
    for x in (sys.stdout, sys.stderr):
        try:
            x.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    m = io.open(MAIN, encoding="utf-8").read()
    s = io.open(SI, encoding="utf-8").read()

    tabs = captions(s, "table")
    figs = captions(s, "figure")
    print(f"보충자료: 표 {len(tabs)}개, 그림 {len(figs)}개\n")
    for i, c in enumerate(tabs, 1):
        print(f"  S{i:<3} {c}")
    print()
    for i, c in enumerate(figs, 1):
        print(f"  SFig{i:<2} {c}")

    cited = sorted({int(x) for x in re.findall(r"Tables?~?S(\d+)", m)}
                   | {int(x) for x in re.findall(r"Table~S(\d+)", m)}
                   | {int(x) for x in re.findall(r"S(\d+)", m)
                      if int(x) <= 99})
    print(f"\n본문이 인용한 보충표 번호: "
          + ", ".join(f"S{n}" for n in cited))
    missing = [n for n in cited if n > len(tabs)]
    unused = [i for i in range(1, len(tabs) + 1) if i not in cited]
    if missing:
        print("  없는 표를 인용합니다: " + ", ".join(f"S{n}" for n in missing))
    if unused:
        print("  본문이 인용하지 않는 표: "
              + ", ".join(f"S{n}" for n in unused))
    if not missing and not unused:
        print("  번호가 모두 맞습니다.")

    print("\n인용이 나오는 문맥")
    for mm in re.finditer(r"[^.]*Tables?~?S\d+[^.]*\.", m):
        t = re.sub(r"\s+", " ", mm.group(0)).strip()
        print(f"  - {t[:160]}")


if __name__ == "__main__":
    main()
