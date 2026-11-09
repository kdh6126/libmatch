"""본문 Figure 1 -- 이 연구가 무엇을 맞추고 무엇을 맞추지 않았는가.

심사자 두 명이 서로 다른 방식으로 같은 것을 놓쳤다. 한 명은 맞춘
라이브러리가 하나인 줄 읽었고(일곱이다), 다른 한 명은 "자료가 다른 다섯
가지"와 "결과가 다른 네 가지"를 같은 목록으로 읽었다. 둘 다 설계가 한눈에
보이지 않는다는 같은 증상이므로, 설계를 그림으로 만든다.

핵심은 세 팔이다. 맞춘 팔은 라벨 출처를 합성 쪽과 공유하고 크기·구조·
정렬가능성을 실측 쪽과 공유한다. 그래서 두 이웃 모두에 대고 비교하면
라벨 출처가 책임지는 몫이 분리된다. 라벨 출처를 "맞춘 것"처럼 그리면
이 논문의 논증이 거꾸로 보이므로, 같은 값을 가진 두 칸을 한 덩어리로
칠해 어느 쪽과 같은지를 자리로 읽히게 한다 -- 범례로 미루지 않는다.

    python src/fig_overview.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, "src")
from journal_style import (setup, TWO_COL, save_fixed_width, INK, INK2, MUTED,
                           WHITE, BLUE, GREEN, VERM, PURPLE, ORANGE)

setup()
OUT = "manuscript/figs/F11.png"

# 세 팔. 가운데가 이 연구에서 만든 것이다.
SYN, MAT, MEA = VERM, BLUE, GREEN
TINT = {SYN: "#fdf1ea", MAT: "#eef5fb", MEA: "#eaf7f2"}

# (a) 다섯 속성 x 세 팔. 마지막 항목은 맞춘 팔이 어느 쪽과 같은지다.
ROWS = [
    ("Label source",
     "lattice energy", "lattice energy", "$\\log_{10} K_D$", SYN),
    ("Library size",
     "6.9 million", "6,000", "20,000–35,000", MEA),
    ("Sequence structure",
     "unrelated sequences", "one parent's\nmutational neighbourhood",
     "one parent's\nmutational neighbourhood", MEA),
    ("Alignability",
     "lengths 11–43", "aligned, length 17", "aligned", MEA),
    ("Measurement noise",
     "none (deterministic)", "none (deterministic)",
     "$\\sigma = 0.27$ in $\\log_{10} K_D$", SYN),
]

# (b) 겉으로 드러난 네 차이와 맞춘 뒤의 판정.
#
# 판정 낱말은 표 1 의 Verdict 열을 그대로 쓴다. 그림과 표를 나란히 놓고
# 읽는 사람에게 두 벌의 판정이 있는 것처럼 보이면 안 된다. 그 낱말이
# 불일치에 대해 무엇을 뜻하는지는 아래 줄에 따로 적는다.
VERDICTS = [
    ("Search advantage",
     "1,620-fold,\nactive over random",
     "290–500 measurements,\nagainst 799 or more",
     "Narrows, does not close",
     "by three orders of magnitude", PURPLE),
    ("Preferred representation",
     "one-hot ahead\nat every budget",
     "the language model\nahead in all seven",
     "Reverses, then reverts",
     "the discrepancy does not survive", GREEN),
    ("Preferred architecture",
     "crossover seen\nonly on the lattice",
     "crossover at 2,000\nlabels in both",
     "Agrees",
     "the discrepancy does not survive", GREEN),
    ("Learnability decomposition",
     "54% noise /\n46% smoothness",
     "the arms vary at different\nnumbers of positions",
     "Not measurable",
     "the arms are not comparable here", ORANGE),
]

COLX = [32.0, 52.0, 72.0]          # 세 팔의 가운데 x
COLW = 19.0
ANNX = 92.0                        # 가운데 팔이 어느 쪽과 같은지 적는 칸


def card(ax, x, y, w, h, edge, fill, lw=0.9, pad=0.42, round_size=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={round_size}",
        facecolor=fill, edgecolor=edge, linewidth=lw))


def chip(ax, x, y, text, color, size=5.3):
    """칸이 어느 팔과 같은지 적는 작은 표. 범례 없이 읽히게 한다."""
    t = ax.text(x, y, text, ha="center", va="center", fontsize=size,
                color=color, fontweight="bold", zorder=5, linespacing=1.3)
    t.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=TINT[color],
                    edgecolor=color, linewidth=0.5))


fig = plt.figure(figsize=(TWO_COL, 5.5))
ax = fig.add_axes([0.012, 0.01, 0.976, 0.985])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ---------------------------------------------------------------- panel (a)
ax.text(0.5, 99.4, "a", ha="left", va="top", fontsize=9.2,
        fontweight="bold", color=INK)
ax.text(4.2, 99.2, "Five properties differ at once, so three are matched and "
        "two are deliberately left alone",
        ha="left", va="top", fontsize=7.9, fontweight="bold", color=INK)

# 열 머리
HEAD_Y = 88.3
heads = [(SYN, "Synthetic benchmark", "Absolut! lattice, as released",
          "1 library · 6 antigens"),
         (MAT, "Matched libraries", "built here, one per antigen",
          "7 libraries · 7 antigens"),
         (MEA, "Measured affinities", "AlphaSeq, as published",
          "4 libraries · 4 targets")]
for (c, title, sub, n), cx in zip(heads, COLX):
    card(ax, cx - COLW / 2, HEAD_Y, COLW, 7.1, c, TINT[c], lw=1.0)
    ax.text(cx, HEAD_Y + 6.0, title, ha="center", va="top", fontsize=7.0,
            fontweight="bold", color=c)
    ax.text(cx, HEAD_Y + 3.5, sub, ha="center", va="top", fontsize=5.6,
            color=INK2)
    ax.text(cx, HEAD_Y + 1.35, n, ha="center", va="top", fontsize=5.9,
            color=c, fontweight="bold")

# 줄
ROW_TOP = 86.0
ROW_H = 7.1
for i, (name, a, b, c, same) in enumerate(ROWS):
    y = ROW_TOP - (i + 1) * ROW_H
    if i % 2 == 0:
        ax.add_patch(FancyBboxPatch(
            (1.0, y + 0.25), 98.0, ROW_H - 0.5,
            boxstyle="round,pad=0,rounding_size=0.6",
            facecolor="#f7f7f7", edgecolor="none", zorder=0))
    # 같은 값을 가진 두 칸을 한 덩어리로 칠한다. 어느 쪽과 같은지가
    # 글이 아니라 자리로 읽힌다.
    j = 1 if same is MEA else 0
    x0 = COLX[j] - COLW / 2
    ax.add_patch(FancyBboxPatch(
        (x0, y + 0.55), COLX[j + 1] + COLW / 2 - x0, ROW_H - 1.1,
        boxstyle="round,pad=0,rounding_size=0.9",
        facecolor=TINT[same], edgecolor=same, linewidth=0.5, alpha=0.95,
        zorder=1))
    ax.text(2.2, y + ROW_H / 2, name, ha="left", va="center", fontsize=6.5,
            fontweight="bold", color=INK)
    for cx, txt in zip(COLX, (a, b, c)):
        ax.text(cx, y + ROW_H / 2, txt, ha="center", va="center",
                fontsize=6.0, color=INK2, linespacing=1.25, zorder=3)
    chip(ax, ANNX, y + ROW_H / 2,
         "matched to the\nmeasured arm" if same is MEA
         else "not matched: as in\nthe synthetic arm", same)

FOOT_Y = ROW_TOP - 5 * ROW_H - 1.3
ax.text(2.2, FOOT_Y,
        "Each antigen is one independent unit: the seven matched libraries "
        "are built and assessed separately, and twenty splits or seeds\n"
        "quantify uncertainty within an antigen, never across them. The "
        "lattice oracle, the antigen and the scoring function are identical "
        "in the\nleft and middle arms, so no difference between those two "
        "can be attributed to the synthetic label.",
        ha="left", va="top", fontsize=5.7, color=MUTED, linespacing=1.45)

# ---------------------------------------------------------------- panel (b)
PB = 36.0
ax.text(0.5, PB + 6.2, "b", ha="left", va="top", fontsize=9.2,
        fontweight="bold", color=INK)
ax.text(4.2, PB + 6.0, "Four differences appear when the arms are compared "
        "as they stand; matching re-adjudicates each one",
        ha="left", va="top", fontsize=7.9, fontweight="bold", color=INK)

CW, GAP = 23.4, 1.3
for i, (name, before, after, verdict, meaning, c) in enumerate(VERDICTS):
    x = 1.6 + i * (CW + GAP)
    card(ax, x, PB - 28.0, CW, 27.2, "#c9c9c9", WHITE, lw=0.8)
    ax.text(x + CW / 2, PB - 2.0, name, ha="center", va="top", fontsize=6.6,
            fontweight="bold", color=INK)

    ax.text(x + 1.3, PB - 6.4, "as they stand", ha="left", va="top",
            fontsize=5.3, color=SYN, fontweight="bold")
    ax.text(x + 1.3, PB - 8.6, before, ha="left", va="top", fontsize=5.9,
            color=INK2, linespacing=1.34)

    ax.text(x + 1.3, PB - 15.0, "after matching", ha="left", va="top",
            fontsize=5.3, color=MAT, fontweight="bold")
    ax.text(x + 1.3, PB - 17.2, after, ha="left", va="top", fontsize=5.9,
            color=INK2, linespacing=1.34)

    ax.plot([x + 1.3, x + CW - 1.3], [PB - 23.3, PB - 23.3], color="#dcdcdc",
            lw=0.7, solid_capstyle="butt")
    ax.text(x + CW / 2, PB - 24.5, verdict, ha="center", va="top",
            fontsize=6.3, fontweight="bold", color=c)
    ax.text(x + CW / 2, PB - 26.4, meaning, ha="center", va="top",
            fontsize=5.5, color=MUTED)

# ---------------------------------------------------------------- bottom band
card(ax, 1.6, 0.9, 96.8, 4.6, "#b7b7b7", "#f2f2f2", lw=0.7)
ax.text(50, 4.15,
        "Two of the four do not survive matching, one cannot be measured once "
        "the arms are comparable, and one narrows by three orders of magnitude "
        "without closing.",
        ha="center", va="center", fontsize=6.5, color=INK2)
ax.text(50, 1.95,
        "Library size and sequence relatedness, not the synthetic label, "
        "carry most of the apparent benchmark–experiment discrepancy.",
        ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_fixed_width(fig, OUT)
plt.close(fig)
print(f"wrote {OUT}")
