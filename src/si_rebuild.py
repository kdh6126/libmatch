"""보충자료의 생성 부분을 지우고 다시 쌓는다.

생성기들이 서로의 구간 안에 끼어들면서 파일이 엉켰다. 표 블록이 Supplementary
Methods 구간 **안**에 들어갔고, 그 상태에서 생성기를 다시 돌리자 짝 없는 END
표시가 생기고 표가 두 벌이 됐다. 컴파일은 그대로 되므로 눈으로 보기 전에는
모른다 -- 본문이 인용하지 않는 표가 여덟 개 늘었을 뿐이다.

한 군데씩 고치면 또 엉킨다. 손으로 쓴 부분(표 S1~S5, 보충 그림, 자료 안내)까지
되감고 생성 부분을 처음부터 다시 쌓는다.

되감기 전에 보충 방법의 네 절을 꺼내 둔다. 그 절들은 본문에서 옮겨 온 것이라
지금 보충자료에만 있고, 그냥 자르면 사라진다.

    python src/si_rebuild.py           무엇을 할지 보여만 준다
    python src/si_rebuild.py --apply   실행한다
"""
import argparse
import io
import os
import re
import subprocess
import sys

SI = "manuscript/supplementary_information.tex"
FIRST_MARK = "% ---- BEGIN"
END_DOC = "\\end{document}"
KEEP_METHODS = ["Noise injection", "Prospective evaluation",
                "Panning re-analysis", "Software and reproducibility"]
GENERATORS = ["si_methods.py", "si_table_cognate.py", "si_table_unmatched.py",
              "si_table_repr_antigens.py", "si_table_search_ci.py",
              "si_table_integrity.py", "si_table_dimension.py",
              "si_table_power.py", "si_table_parent.py",
              "si_table_crossantigen.py", "si_table_half.py",
              "si_table_rescore.py", "si_order.py"]
NL = chr(10)


def cut_subsection(s, title):
    head = "\\subsection*{" + title + "}"
    a = s.find(head)
    if a < 0:
        return None
    nxt = s.find("\\subsection*{", a + len(head))
    end = s.find("\\section*{", a + len(head))
    mark = s.find("% ---- ", a + len(head))
    b = min(x for x in (nxt, end, mark, len(s)) if x > 0)
    return s[a:b].rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    for x in (sys.stdout, sys.stderr):
        try:
            x.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    s = io.open(SI, encoding="utf-8").read()
    n_tab = len(re.findall(r"\\begin\{table\}", s))
    marks = re.findall(r"% ---- (BEGIN|END) ([a-z ]+)", s)
    print(f"지금 {SI}: 표 {n_tab}개, 표시 {len(marks)}개")

    saved = {}
    for t in KEEP_METHODS:
        got = cut_subsection(s, t)
        if got is None:
            print(f"  보충 방법에서 찾지 못함: {t}")
        else:
            saved[t] = got
            print(f"  꺼내 둠: {t} ({len(got)}자)")
    if len(saved) != len(KEEP_METHODS):
        sys.exit("보충 방법의 절을 다 찾지 못했습니다. 되감으면 사라집니다.")

    i = s.find(FIRST_MARK)
    if i < 0:
        sys.exit("생성 표시가 없습니다. 이미 깨끗합니다.")
    head = s[:i].rstrip() + NL
    print(f"  손으로 쓴 부분 {head.count(chr(92) + 'begin{table}')}개 표를 "
          f"남기고 그 뒤를 지웁니다")

    if not a.apply:
        print(NL + "실행하려면 --apply 를 주십시오.")
        return 0

    io.open(SI + ".tangled", "w", encoding="utf-8", newline=NL).write(s)
    print(f"  엉킨 판을 {SI}.tangled 로 남겼습니다")

    # 되감은 뒤, 보충 방법의 네 절을 되돌려 놓는다. si_methods.py 가 이것을
    # 보충자료에서 다시 읽어 제자리에 넣는다.
    block = ("% ---- BEGIN supplementary methods (generated)" + NL
             + "\\clearpage" + NL + NL
             + "\\section*{Supplementary Methods}" + NL + NL
             + (NL + NL).join(saved[t] for t in KEEP_METHODS) + NL
             + "% ---- END supplementary methods" + NL)
    io.open(SI, "w", encoding="utf-8", newline=NL).write(
        head + NL + block + NL + END_DOC + NL)
    print("  되감았습니다")

    for g in GENERATORS:
        p = os.path.join("src", g)
        if not os.path.exists(p):
            print(f"  없음, 건너뜀: {g}")
            continue
        r = subprocess.run([sys.executable, p], capture_output=True, text=True)
        tag = "ok" if r.returncode == 0 else f"실패({r.returncode})"
        last = [x for x in (r.stdout or "").splitlines() if x.strip()]
        print(f"  {g:28s} {tag}  {last[-1][:70] if last else ''}")
        if r.returncode != 0:
            print((r.stderr or "")[-400:])
            sys.exit(f"{g} 에서 멈췄습니다.")

    s2 = io.open(SI, encoding="utf-8").read()
    print(NL + f"다시 쌓은 뒤: 표 "
          f"{len(re.findall(r'\\\\begin{table}', s2))}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
