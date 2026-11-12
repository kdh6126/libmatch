"""Figure: what the lattice oracle does and does not encode.

The matched-library result says a benchmark can rank methods faithfully. It
does not say the labels are a model of the biology. This figure tests the
second question directly, using the benchmark's own provenance: every Absolut!
antigen comes from an antibody-antigen complex, so each antigen carries a
cognate antibody whose CDRH3 the same oracle can score.

Three panels, three ways of asking the same thing.

  (a)  energy percentile of the cognate CDRH3 against the published table
  (b)  its rank on its own antigen among all other antigens
  (c)  why: the contact site is set by the antigen, not by the sequence

All three are plotted against the uniform expectation, which is what "the
oracle carries no pairing information" predicts.

    python src/fig_oracle_scope.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from journal_style import (setup, panel, ygrid,              # noqa: E402
                           save_fixed_width, TWO_COL,
                           INK, MUTED, GRID, BLUE, VERM, ORANGE)

setup()
OUT = "manuscript/figs"
FULL_COVERAGE = 99.0


def pair_sensitivity():
    """exp69/exp70 과 pair_* 열두 팔에서 항원별 강·약 rank01 중앙값."""
    import glob
    import re

    rows = []
    for f in sorted(glob.glob("results/*_ranks.csv")):
        t = os.path.basename(f)[:-len("_ranks.csv")]
        if t == "exp69":
            ag, arm = "2DD8_S", "strong"
        elif t == "exp70":
            ag, arm = "2DD8_S", "weak"
        else:
            m = re.match(r"pair_(.+)_(weak|strong)$", t)
            if not m:
                continue
            ag, arm = m.groups()
        rows.append(dict(antigen=ag, arm=arm,
                         rank01=float(pd.read_csv(f).rank01.median())))
    if not rows:
        return None
    d = pd.DataFrame(rows).pivot(index="antigen", columns="arm",
                                 values="rank01")
    if not {"strong", "weak"} <= set(d.columns):
        return None
    return d.dropna().sort_values("strong").reset_index()


def load():
    d = pd.read_csv("results/exp62_native_fulltable.csv")
    pre = pd.read_csv("results/exp67_epitope_precheck.csv")
    bad = set(pre[pre.verdict == "전제 실패"].antigen.astype(str))
    ok = d[(d.status == "ok") & (~d.antigen.astype(str).isin(bad))]
    ok = ok[ok.coverage_pct.fillna(100) >= FULL_COVERAGE]
    rank = pd.read_csv("results/exp64_ranks.csv")
    epi = pd.read_csv("results/exp66_epitope.csv")
    return ok, rank, epi


def ecdf_vs_uniform(ax, x01, colour, label):
    """Empirical CDF against the uniform line.

    A CDF rather than a histogram: bin width is a choice, and with 90 points a
    histogram's shape changes with it. The diagonal is the null, so the reader
    judges the claim by how far the curve departs from a straight line.
    """
    x = np.sort(np.asarray(x01, float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)),
            label="uniform")
    ax.step(np.concatenate([[0], x, [1]]),
            np.concatenate([[0], y, [1]]), where="post",
            color=colour, lw=1.4, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def legend_inside(ax):
    """범례를 패널 안에 둔다.

    긴 이름을 그대로 두면 범례가 패널보다 넓어져 옆 패널의 축 이름을
    덮는다. 이 그림이 실제로 그랬다 -- (a)의 범례가 (b)의 "Cumulative
    fraction" 을, (b)의 범례가 (c)의 "Contact overlap (Jaccard)" 를 덮고
    있었다. matplotlib 은 범례를 자르지도 경고하지도 않는다.

    이름과 표본 수를 한 줄에 같이 넣으면 어떤 글씨 크기로도 들어가지
    않는다. 네 패널이 가로로 서 있어 패널 하나가 70pt 가 채 안 되기
    때문이다. 그래서 이름만 범례에 두고 수는 따로 적는다(count).
    """
    ax.legend(loc="upper left", frameon=False, fontsize=6.2,
              handlelength=1.6, handletextpad=0.5, borderpad=0.15,
              labelspacing=0.3)


def count(ax, text):
    """표본 수를 오른쪽 아래에 적는다.

    ECDF 는 왼쪽 아래에서 오른쪽 위로 오르므로 오른쪽 아래 구석은 늘
    비어 있다. 범례에서 덜어낸 수를 그리로 보낸다.
    """
    ax.text(0.97, 0.03, text, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.0, color=MUTED)


def check_inside(fig, axes):
    """패널 안에 적은 글이 정말 패널 안에 있는가.

    글이 패널 밖으로 나가도 matplotlib 은 아무 말도 하지 않는다. 옆
    패널의 축 이름 위에 얹히고 나서야 사람 눈에 띄고, 그때는 이미 PDF 가
    나간 뒤다. 가로로 넘치는 것만 본다 -- 패널 글자와 중앙값 꼬리표는
    일부러 위아래로 나가 있다.
    """
    fig.canvas.draw()
    bad = []
    for ax, L in zip(axes, "abcd"):
        box = ax.get_window_extent()
        for a in [ax.get_legend()] + list(ax.texts):
            if a is None:
                continue
            e = a.get_window_extent()
            out = max(box.x0 - e.x0, e.x1 - box.x1)
            if out > 1.0:
                bad.append("(%s) 글이 패널 밖으로 %.0fpx 나갑니다: %r"
                           % (L, out, getattr(a, "get_text",
                                              lambda: "legend")()[:40]))
    return bad


def main():
    ok, rank, epi = load()
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, TWO_COL * 0.30))
    fig.subplots_adjust(wspace=0.62, bottom=0.42, top=0.94)

    # ---- (a) energy percentile of the cognate antibody
    ax = axes[0]
    ecdf_vs_uniform(ax, ok.pct_better / 100, BLUE, "cognate CDRH3")
    ax.set_xlabel("Energy percentile of\nthe cognate CDRH3")
    ax.set_ylabel("Cumulative fraction")
    legend_inside(ax)
    count(ax, f"n = {len(ok)} antigens")
    panel(ax, "a", drop=0.46)

    # ---- (b) rank of the own antigen among the others
    ax = axes[1]
    ecdf_vs_uniform(ax, rank.rank01, VERM, "own antigen")
    ax.set_xlabel("Rank of the cognate\nantigen among all others")
    ax.set_ylabel("Cumulative fraction")
    legend_inside(ax)
    count(ax, f"n = {len(rank)} antibodies")
    panel(ax, "b", drop=0.46)

    # ---- (c) the site is a property of the antigen
    ax = axes[2]
    pairs = [("random\nvs random", epi.pair_jaccard, ORANGE),
             ("cognate\nvs random", epi.nat_ref_jaccard, BLUE)]
    for i, (lab, v, c) in enumerate(pairs):
        v = v.dropna()
        # box without the matplotlib default furniture: the point is the
        # location of the two distributions relative to each other.
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(plt.Rectangle((i - 0.22, q1), 0.44, q3 - q1,
                                   facecolor=c, edgecolor=INK, lw=0.6,
                                   alpha=0.35, zorder=2))
        ax.plot([i - 0.22, i + 0.22], [med, med], color=INK, lw=1.2, zorder=3)
        lo, hi = np.percentile(v, [5, 95])
        ax.plot([i, i], [lo, hi], color=INK, lw=0.6, zorder=1)
        ax.text(i, 1.02, f"{med:.2f}", ha="center", va="bottom", fontsize=7,
                color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([p[0].replace(chr(10), " ") for p in pairs],
                       fontsize=6.0, rotation=90)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel("Contact overlap (Jaccard)")
    ygrid(ax)
    panel(ax, "c", drop=0.46)

    # ---- (d) 검정의 감도는 오라클이 매긴 세기의 함수다
    ax = axes[3]
    pair = pair_sensitivity()
    if pair is not None:
        x = np.arange(len(pair))
        ax.plot(x, pair.strong, "o-", color=BLUE, lw=1.0, ms=4,
                label="strong arm")
        ax.plot(x, pair.weak, "s-", color=VERM, lw=1.0, ms=4,
                label="weak arm")
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        # x 는 패널 비율, y 는 자료값. 자료 좌표로 적으면 항원 수가
        # 달라질 때마다 글이 패널 밖으로 밀려난다 -- 실제로 그랬다.
        ax.text(0.02, 0.5, "chance", transform=ax.get_yaxis_transform(),
                color=MUTED, fontsize=6.0, va="bottom", ha="left")
        ax.set_xticks(x)
        ax.set_xticklabels([a.split("_")[0] for a in pair.antigen],
                           fontsize=5.8, rotation=90)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rank of the home antigen")
        ax.set_xlabel("Antigen")
        ax.legend(loc="upper left", frameon=False, fontsize=6.0,
                  handlelength=1.6, borderpad=0.2)
        ygrid(ax)
    panel(ax, "d", drop=0.46)

    miss = check_inside(fig, axes)
    if miss:
        for m in miss:
            print("  " + m)
        sys.exit("패널 밖으로 나간 글이 있습니다. 줄이거나 자리를 옮긴 뒤 "
                 "다시 그리십시오 -- 네 패널이 가로로 서 있어 넘친 글은 "
                 "옆 패널의 축 이름 위에 얹힙니다.")

    p = f"{OUT}/F7.png"
    save_fixed_width(fig, p)
    plt.close(fig)
    print(f"wrote {p}")
    print(f"  (a) n={len(ok)} median {ok.pct_better.median():.2f}%")
    print(f"  (b) n={len(rank)} median {rank.rank01.median():.3f}")
    print(f"  (c) random-random {epi.pair_jaccard.median():.3f} · "
          f"cognate-random {epi.nat_ref_jaccard.median():.3f}")


if __name__ == "__main__":
    main()
