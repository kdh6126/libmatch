"""소스에 보이지 않는 손상이 들어가 있는지 훑는다.

이 저장소의 파일들은 셸 heredoc 을 거쳐 여러 번 고쳐졌다. 그 과정에서 백슬래시
한 겹이 먹히면 비원시 문자열의 `\\b` 가 워드 경계가 아니라 백스페이스 문자
(0x08)가 된다. 터미널에서는 보이지 않고, 파이썬은 아무 불평 없이 실행하며,
정규식은 조용히 아무것도 맞추지 못한다. 실제로 그렇게 검사 하나가 0건을
반환하고 있었다.

같은 종류를 전부 찾는다.

  제어문자   줄바꿈·탭을 뺀 C0 문자. 문자열 안에 들어갈 이유가 없다.
  문법       파싱되지 않는 파일. 편집 중간에 멈춘 흔적.
  깨진 정규식 re.compile 에 넘기는 패턴 가운데 컴파일되지 않는 것.

    python src/check_sources.py
"""
import ast
import io
import os
import re
import sys

SRC = "src"
BAD_CTRL = {c for c in range(32)} - {9, 10, 13}
NL = chr(10)


def controls(s):
    return sorted({ord(c) for c in s if ord(c) in BAD_CTRL})


def regexes(path, s):
    """re.compile / re.search 등에 넘기는 리터럴 패턴이 컴파일되는지."""
    out = []
    try:
        tree = ast.parse(s)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = getattr(f, "attr", None)
        mod = getattr(getattr(f, "value", None), "id", None)
        if mod != "re" or name not in {"compile", "search", "match",
                                       "findall", "finditer", "sub", "split"}:
            continue
        if not node.args:
            continue
        a = node.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            try:
                re.compile(a.value)
            except re.error as e:
                out.append((node.lineno, a.value[:50], str(e)))
    return out


def main():
    for x in (sys.stdout, sys.stderr):
        try:
            x.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".py"))
    n_ctrl = n_syn = n_re = 0
    for f in files:
        p = os.path.join(SRC, f)
        try:
            s = io.open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            print(f"  [인코딩] {f} — UTF-8 로 읽히지 않습니다")
            n_syn += 1
            continue
        c = controls(s)
        if c:
            n_ctrl += 1
            print(f"  [제어문자] {f} — {', '.join(f'0x{x:02x}' for x in c)}")
        try:
            ast.parse(s)
        except SyntaxError as e:
            n_syn += 1
            print(f"  [문법] {f}:{e.lineno} — {e.msg}")
            continue
        for ln, pat, msg in regexes(p, s):
            n_re += 1
            print(f"  [정규식] {f}:{ln} — {msg}  ({pat!r})")

    print()
    print(f"파일 {len(files)}개 · 제어문자 {n_ctrl} · 문법 {n_syn} · "
          f"정규식 {n_re}")
    return 1 if (n_ctrl or n_syn or n_re) else 0


if __name__ == "__main__":
    sys.exit(main())
