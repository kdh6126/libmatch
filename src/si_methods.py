"""보충 방법 — 본문 Methods에서 옮겨 온 절들.

두 가지를 한 번에 해결한다.

  1 본문이 "(Supplementary Methods)"를 인용하는데 보충자료에 그런 절이 없다.
    심사자가 곧바로 걸리는 자리다.
  2 본문이 BiB의 5,000 단어 상한을 넘는다. 주장을 읽는 데 필요하지 않고
    재현에만 필요한 절들이 옮기기에 맞다.

무엇을 옮기고 무엇을 남기는지에 기준을 둔다. 본문의 주장을 따라가는 데 필요한
절(자료, 정합 라이브러리, 동족 항체 검사, 모형과 특징, 평가, 통계)은 남기고,
부수 실험의 절차(잡음 주입, 전향 평가, 패닝 재분석, 소프트웨어)는 옮긴다.
옮긴 절은 사라지지 않고 보충자료에서 같은 제목으로 이어진다.

한 절은 새로 쓴다. 본문이 인용하는 '정합 전 팔의 인코더'는 어디에도 적혀 있지
않았다.

    python src/si_methods.py
"""
import io
import re
import sys

MAIN = "manuscript/manuscript.tex"
SI = "manuscript/supplementary_information.tex"
MARK_A = "% ---- BEGIN supplementary methods (generated)"
MARK_B = "% ---- END supplementary methods"

MOVE = ["Noise injection", "Prospective evaluation", "Panning re-analysis",
        "Software and reproducibility"]

# 본문이 인용하지만 어디에도 적혀 있지 않던 절.
EXTRA = r"""\subsection*{Encoders}

The two arms of the representation comparison cannot share an encoder. In a
matched library every sequence is a variant of one parent and has the same
length, so position $i$ of one sequence corresponds to position $i$ of another
and an aligned one-hot encoding is defined: 21 indicator columns per varying
position. The released library is 6{,}917{,}820 unrelated CDRH3 sequences of
lengths 11 to 43, where no such correspondence exists. There a sequence is
encoded by positional one-hot over the 11 residues at each terminus, counts of
all 400 dipeptides, and a one-hot of the length. ESM-2 features are computed
the same way in both arms -- per-residue embeddings from layer 12 of the
35M-parameter checkpoint, reduced to 48 principal components fitted on a sample
of residue vectors -- but at different positions, because the arms have
different notions of position: the positions that vary within a matched
library, and the first 26 positions of the released one. Ridge regularisation is $\alpha = 1$ for one-hot and $\alpha = 50$ for
the embeddings, values fixed before the comparisons reported here and unchanged
between arms.

"""


def cut(s, title):
    """한 하위 절을 그 다음 경계 바로 앞까지 떼어 낸다.

    경계에 `% ---- ` 표시와 문서 끝을 포함해야 한다. 그러지 않으면 마지막
    하위절을 잘라낼 때 파일 끝까지 딸려 오고, 그 조각을 보충자료에 되돌려
    놓을 때 END 표시와 문서 끝이 블록 안에 들어간다. 문서 끝이 둘이 되면
    그 뒤의 생성기들은 모두 count 없는 replace 로 붙이므로 무엇을 붙이든
    두 벌이 된다 -- 실제로 보충표 여덟 개가 열여섯 개가 됐고, 컴파일은
    그대로 돼서 눈으로 보기 전에는 알 수 없었다.
    """
    head = "\\subsection*{" + title + "}"
    a = s.find(head)
    if a < 0:
        return s, None
    bounds = [s.find(x, a + len(head)) for x in
              ("\\subsection*{", "\\section*{", "% ---- ", "\\end{document}")]
    b = min([x for x in bounds if x > 0] + [len(s)])
    return s[:a] + s[b:], s[a:b].rstrip() + "\n\n"


def main():
    for x in (sys.stdout, sys.stderr):
        try:
            x.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    m = io.open(MAIN, encoding="utf-8").read()
    si_now = io.open(SI, encoding="utf-8").read()
    # 이미 보충자료에 있는 절은 거기서 다시 읽어 온다. 이렇게 하지 않으면 두 번째
    # 실행에서 본문에도 없고 보충자료에도 없는 상태가 된다 -- 실제로 한 번
    # 그렇게 네 절이 사라졌고, 컴파일은 멀쩡히 됐다.
    moved = []
    for title in MOVE:
        m, block = cut(m, title)
        if block is None:
            _, kept = cut(si_now, title)
            if kept is None:
                print(f"  어디에도 없습니다: {title}")
            else:
                moved.append(kept)
                print(f"  보충자료에서 유지: {title}")
            continue
        moved.append(block)
        print(f"  옮김: {title}")
    m = re.sub(r"\n{3,}", "\n\n", m)
    io.open(MAIN, "w", encoding="utf-8", newline="\n").write(m)

    body = ["\\clearpage", "\\section*{Supplementary Methods}", "",
            EXTRA.rstrip(), ""] + [b.rstrip() for b in moved]
    tex = MARK_A + "\n" + "\n\n".join(body) + "\n" + MARK_B

    s = io.open(SI, encoding="utf-8").read()
    if MARK_A in s:
        a, b = s.index(MARK_A), s.index(MARK_B) + len(MARK_B)
        s = s[:a] + tex + s[b:]
    else:
        s = s.replace("\\end{document}", tex + "\n\n\\end{document}")
    io.open(SI, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {SI}  (보충 방법 {len(moved) + 1}절)")
    print(f"wrote {MAIN}")


if __name__ == "__main__":
    main()
