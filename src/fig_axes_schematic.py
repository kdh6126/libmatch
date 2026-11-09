"""Figure: what the five properties are, and what the tool does with them.

Every other figure in this paper is a data plot. A reader meeting the argument
for the first time has to assemble it from prose: that two datasets differ on
five countable properties at once, that this is what makes a comparison between
them uninterpretable, and that the tool reports which of the five differ. That
belongs in a picture.

Three bands, read downwards.

  (a)  the five properties
  (b)  what the tool reads and what it returns
  (c)  the verdict for four dataset pairs, the first of which this paper
       dissects and the last two of which contain no antibody data

Drawn rather than plotted, so the coordinates are arbitrary. They are set out
once at the top instead of being scattered through the drawing calls, because
a schematic is edited by nudging things and that is unreadable when every
number is inline.

    python src/fig_axes_schematic.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, "src")
from journal_style import (setup, save_fixed_width, TWO_COL,    # noqa: E402
                           INK, MUTED, GRID, BLUE, VERM, PANEL, WHITE)

setup()
OUT = "manuscript/figs"

AXES = [("size", "number of\nsequences"),
        ("structure", "one parent, or\nunrelated"),
        ("dimension", "positions\nthat vary"),
        ("noise", "replicate\nspread"),
        ("budget", "labels\navailable")]

# 띠마다 (아래 y, 높이). 위에서부터 (a) (b) (c).
BANDS = {"a": (72.0, 22.0), "b": (50.5, 19.0),
         "c": (28.0, 21.0), "d": (1.0, 25.5)}
YMAX = 95.0
PG = "results/proteingym_five_axes.csv"

# libmatch 의 세 판정. 문자열을 그대로 쓴다 -- 그림이 도구와 다른 어휘를
# 쓰면 둘이 어긋났을 때 알아챌 방법이 없다.
DIFF, SAME, NONE = "mismatched", "matched", "unavailable"

PAIRS_FALLBACK = [
    ("Matched vs released lattice", [DIFF, DIFF, DIFF, SAME, DIFF]),
    ("Measured vs matched lattice", [DIFF, SAME, DIFF, DIFF, DIFF]),
    ("ProteinGym RASH vs matched", [SAME, SAME, DIFF, NONE, SAME]),
    ("ProteinGym TPK1 vs matched", [SAME, SAME, DIFF, NONE, SAME]),
]


def mark(ax, x, y, st, r=0.95):
    """한 칸의 판정을 그린다.

    기호가 둘뿐이면 "재지 못했다" 가 반드시 둘 중 하나로 접히고, 접히는
    쪽은 빈 원 -- 즉 "두 쪽이 일치한다" 다. 아무것도 재지 않았다는 사실이
    일치했다는 주장으로 바뀌므로 세 번째 기호를 따로 둔다.
    """
    if st == DIFF:
        ax.add_patch(plt.Circle((x, y), r, facecolor=VERM, edgecolor=VERM,
                                lw=0.8, zorder=3))
    elif st == SAME:
        ax.add_patch(plt.Circle((x, y), r, facecolor=WHITE, edgecolor=MUTED,
                                lw=0.8, zorder=3))
    else:
        ax.add_patch(plt.Circle((x, y), r, facecolor=PANEL, edgecolor=MUTED,
                                lw=0.8, linestyle=(0, (1.6, 1.3)), zorder=3))
        ax.plot([x - 0.52, x + 0.52], [y, y], color=MUTED, lw=1.0, zorder=4,
                solid_capstyle="round")


def verdicts():
    """가능하면 도구를 실제로 돌려 채운다. 그림이 표와 어긋나면 안 된다."""
    try:
        sys.path.insert(0, "libmatch")
        import libmatch as LM
        import table_libmatch as T
        rows = []
        for name, _sub, a, b in T.PAIRS:
            cmp_df, _A, _B = T.run(a, b)
            d = {r.axis: str(r.status) for _, r in cmp_df.iterrows()}
            rows.append((name, [d[k] for k, _ in AXES]))
        return rows
    except Exception as e:                                   # noqa: BLE001
        print(f"  도구를 돌리지 못해 기록된 값을 씁니다 ({type(e).__name__})",
              file=sys.stderr)
        return PAIRS_FALLBACK


def band(ax, key, title):
    y, h = BANDS[key]
    ax.add_patch(FancyBboxPatch((0.6, y), 98.8, h, boxstyle="round,pad=0.4",
                                facecolor=WHITE, edgecolor=GRID, lw=0.8,
                                zorder=1))
    ax.text(3.2, y + h - 2.2, title, fontsize=7.2, fontweight="bold",
            color=INK, va="center", ha="left", zorder=3)
    ax.text(1.8, y + h - 2.2, f"({key})", fontsize=8, fontweight="bold",
            color=INK, va="center", ha="center", zorder=4)
    return y, h


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = verdicts()

    fig, ax = plt.subplots(figsize=(TWO_COL, TWO_COL * 0.98))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, YMAX)
    ax.axis("off")

    # ---------------------------------------------------------------- (a)
    y, h = band(ax, "a", "Five operational axes measured by libmatch")
    x0, w, gap = 5.0, 17.0, 2.0
    box_y, box_h = y + 8.0, 9.0
    for i, (key, sub) in enumerate(AXES):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, box_y), w, box_h,
                                    boxstyle="round,pad=0.3",
                                    facecolor=PANEL, edgecolor=MUTED, lw=0.7,
                                    zorder=2))
        ax.text(x + w / 2, box_y + box_h - 2.6, key, fontsize=7.0,
                ha="center", va="center", color=INK, zorder=3)
        ax.text(x + w / 2, box_y + 2.8, sub, fontsize=5.9, ha="center",
                va="center", color=MUTED, zorder=3)
    ax.text(50, y + 3.4,
            "a synthetic benchmark and a measured dataset usually differ on "
            "several of these at once, so a\ndifference measured between them "
            "is not attributable to any one of them",
            fontsize=6.2, ha="center", va="center", color=INK, zorder=3)

    # ---------------------------------------------------------------- (b)
    y, h = band(ax, "b", "What the tool reads, and what it returns")
    mid = y + 7.5
    for dy, lab, colr in ((3.8, "dataset A\nsequence + label", BLUE),
                          (-3.8, "dataset B\nsequence + label", VERM)):
        ax.add_patch(FancyBboxPatch((4.5, mid + dy - 3.0), 17, 6.0,
                                    boxstyle="round,pad=0.3",
                                    facecolor=WHITE, edgecolor=colr, lw=1.0,
                                    zorder=2))
        ax.text(13, mid + dy, lab, fontsize=6.2, ha="center", va="center",
                color=INK, zorder=3)
        ax.add_patch(FancyArrowPatch((22.0, mid + dy), (29.4, mid),
                                     arrowstyle="-|>", mutation_scale=7,
                                     color=MUTED, lw=0.8, zorder=3,
                                     shrinkA=0, shrinkB=0))
    ax.add_patch(FancyBboxPatch((30, mid - 4.5), 20, 9.0,
                                boxstyle="round,pad=0.3", facecolor=PANEL,
                                edgecolor=INK, lw=1.0, zorder=2))
    ax.text(40, mid + 1.8, "libmatch", fontsize=7.4, ha="center",
            va="center", color=INK, family="monospace", zorder=3)
    ax.text(40, mid - 2.0, "measures the five\non each side", fontsize=5.9,
            ha="center", va="center", color=MUTED, zorder=3)
    ax.add_patch(FancyArrowPatch((50.6, mid), (58.0, mid), arrowstyle="-|>",
                                 mutation_scale=7, color=MUTED, lw=0.8,
                                 zorder=3))
    ax.text(79, mid + 2.6, "which of the five differ, and by how much",
            fontsize=6.4, ha="center", va="center", color=INK, zorder=3)
    ax.text(79, mid - 1.8,
            "an axis that differs is a rival explanation for any\n"
            "difference the comparison reports",
            fontsize=5.9, ha="center", va="center", color=MUTED, zorder=3)

    # ---------------------------------------------------------------- (c)
    y, h = band(ax, "c", "What it reports for four pairs")
    cx0, cw = 48.0, 9.6
    head_y = y + h - 4.2
    for j, (key, _) in enumerate(AXES):
        ax.text(cx0 + j * cw + cw / 2, head_y, key, fontsize=6.0,
                ha="center", va="center", color=MUTED, zorder=3)
    for i, (name, marks) in enumerate(rows):
        ry = head_y - 3.2 - i * 2.45
        ax.text(4.5, ry, name, fontsize=6.1, ha="left", va="center",
                color=INK, zorder=3)
        for j, st in enumerate(marks):
            mark(ax, cx0 + j * cw + cw / 2, ry, st)
    leg_y = y + 3.9
    for lx, st, lab in ((17.0, DIFF, "the two sides differ"),
                        (38.0, SAME, "the two sides agree"),
                        (59.0, NONE, "the axis was not measured")):
        mark(ax, lx, leg_y, st, r=0.80)
        ax.text(lx + 1.5, leg_y, lab, fontsize=5.9, ha="left", va="center",
                color=MUTED, zorder=3)
    # 왜 재지 못했는가는 캡션이 말한다. 여기서는 한 줄만 둔다 -- 범례와
    # 마지막 행 사이에 들어갈 자리가 그만큼뿐이다.
    note = ("The last two pairs carry one score per variant, so their "
            "replicate spread cannot be recovered; they contain no "
            "antibody data.")
    ax.text(50, y + 1.5, note, fontsize=5.9, ha="center", va="center",
            color=MUTED, zorder=3)

    # ---------------------------------------------------------------- (d)
    y, h = band(ax, "d", "Where the compared libraries sit among 217 "
                         "measured assays")
    import numpy as np
    import pandas as pd
    if os.path.exists(PG):
        pg_all = pd.read_csv(PG)
        pg = pg_all[pg_all.kind == "ProteinGym"]
        ours = pg_all[pg_all.kind == "this study"]
        marks = [("matched lattice", BLUE), ("measured antibody", VERM)]
        cols = [("n", "sequences"), ("n_varying", "varying positions"),
                ("usable_train_max", "training budget")]
        # 바깥 축 위에 작은 축을 얹는다. 좌표가 0-100, 0-YMAX 이므로
        # 그림 비율로 환산해 둔다.
        bb = ax.get_position()
        for k, (col, lab) in enumerate(cols):
            fx = bb.x0 + bb.width * (0.30 + k * 0.235)
            fy = bb.y0 + bb.height * ((y + 7.0) / YMAX)
            sub = fig.add_axes([fx, fy, bb.width * 0.185,
                                bb.height * ((h - 13.0) / YMAX)])
            v = pg[col].dropna().to_numpy()
            v = v[v > 0]
            sub.hist(v, bins=np.logspace(np.log10(v.min()),
                                         np.log10(v.max()), 22),
                     color=GRID, edgecolor=MUTED, lw=0.3)
            sub.set_xscale("log")
            for nm, colr in marks:
                r = ours[ours.dataset == nm]
                if len(r) and pd.notna(r[col].iloc[0]):
                    sub.axvline(float(r[col].iloc[0]), color=colr, lw=1.2)
            sub.set_xlabel(lab, fontsize=5.8)
            sub.tick_params(labelsize=5.0)
            if k == 0:
                sub.set_ylabel("assays", fontsize=5.8)
            else:
                sub.set_yticklabels([])
        for dy, nm, colr in ((3.2, "matched lattice", BLUE),
                             (0.8, "measured antibody", VERM)):
            ax.plot([6.0, 9.0], [y + h / 2 + dy] * 2, color=colr,
                    lw=1.6, zorder=3)
            ax.text(10.0, y + h / 2 + dy, nm, fontsize=5.9, ha="left",
                    va="center", color=INK, zorder=3)
        ax.text(6.0, y + h / 2 - 3.6,
                "only the number of varying\npositions separates them",
                fontsize=5.9, ha="left", va="center", color=MUTED,
                zorder=3)
        ax.text(50, y + 1.0,
                "the noise axis is absent: every one of the 217 distributes a "
                "single score per variant",
                fontsize=5.9, ha="center", va="center", color=MUTED, zorder=3)

    p = f"{OUT}/F0.png"
    save_fixed_width(fig, p)
    plt.close(fig)
    print(f"wrote {p}")
    for name, marks in rows:
        n_d = sum(1 for m in marks if m == DIFF)
        n_n = sum(1 for m in marks if m == NONE)
        print(f"  {name:<34} {n_d}/5 어긋남"
              + (f" · {n_n} 비교불가" if n_n else ""))


if __name__ == "__main__":
    main()
