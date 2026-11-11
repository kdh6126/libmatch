"""그림 2와 3 — 정합이 무엇을 없애고 무엇을 남기는가.

두 그림 모두 예전에는 항원 하나, 시드 3~5개로 그려져 있었다(figs_journal.py).
여기서 다시 그린다. 바뀐 것은 자료의 양이고, 그림이 말하는 바도 그만큼 달라진다.

  F3  표현 선호는 라벨 예산의 함수다. 정합 전에는 one-hot이 앞서고 그 차이가
      라벨이 늘수록 **벌어진다**(exp70). 정합 후에는 언어 모형이 앞서고 그
      차이가 **좁혀진다**(exp71, 항원 셋). 두 곡선이 반대로 기울어 있다는 것이
      요점이고, 그래서 "어느 표현이 낫다"는 질문은 예산을 빼고는 답이 없다.

  F2  (a) 탐색 우위는 한 자리 수로 내려오지만 0으로 가지 않는다. 시드 20개와
          부트스트랩 구간으로 다시 재면 정합 격자가 실측 표적들 위에 남는다.
          예전 판은 시드 5개였고 구간이 없었다.
      (b) 남은 몫은 라이브러리가 얼마나 학습 가능한지를 따라간다. 정합이 맞춘
          것은 크기와 변이체 구조이고, 맞추지 못한 것은 설계 공간의 차원이다.

    python src/fig_matching_limits.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import journal_style as JS                                      # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FIG_REPR = "manuscript/figs/F3.png"
FIG_SEARCH = "manuscript/figs/F2.png"
PER_SZ = "results/exp71_per_size.csv"
UNMATCHED = "results/exp70_per_size_tests.csv"
RATIOS = "results/exp72_ratios.csv"
RESID = "results/exp73_why_residual.csv"
BUDGET = 1000
COLORS = [JS.BLUE, JS.GREEN, JS.PURPLE, JS.ORANGE, JS.SKY]
# (b)의 점 라벨은 서로 붙는다. 겹치는 것만 방향을 바꾼다.
# (b)의 오른쪽 위에 항원 넷이 rho 0.914~0.915 로 겹쳐 선다. 기본값에
# 맡기면 이름끼리 포개져 "1ADQ_A1NCA_N" 처럼 읽을 수 없는 글자가 된다.
# 그래서 점마다 어느 쪽으로 얼마나 밀지, 그리고 어느 끝을 점에 맞출지를
# (dx, dy, 정렬) 로 직접 적는다. 점이 열한 개뿐이므로 자동 배치보다
# 손으로 적는 편이 확실하다.
NUDGE = {
    "2Q8A_A":   (-7, -2, "right"),
    "1ZTX_E":   (0, 12, "center"),   # 위로 띄우고 지시선을 단다(LEADER)
    "1OSP_O":   (-7, -2, "right"),
    "1NCA_N":   (7, -4, "left"),     # 1OSP_O 와 y 가 0.17 밖에 안 떨어진다
    "1ADQ_A":   (-7, -2, "right"),
    "3VRL_C":   (7, -2, "left"),
    "2DD8_S":   (-7, 2, "right"),
    "HER2":     (7, 1, "left"),
    "TIGIT":    (7, 1, "left"),
    "PD-1":     (7, -1, "left"),
    "CoV2-RBD": (2, 7, "left"),      # 왼쪽은 축 밖이다 -- 위로 올린다
}

# 2Q8A_A 의 점이 1ZTX_E 의 이름과 같은 높이에 선다. 가까운 이름을 제 것으로
# 읽는 것이 보통이므로, 그대로 두면 틀리게 읽힌다. 자리를 더 옮기면 다른
# 이름과 부딪히므로, 이 하나에만 가는 선을 그어 임자를 밝힌다.
LEADER = {"1ZTX_E"}


def repr_panel(ax):
    """정합 전 한 선, 정합 후는 항원들의 중앙선과 최소~최대 띠.

    항원마다 선을 긋지 않는다. 일곱 개를 서로 다른 색으로 구별하려면 색이
    돌아가고, 돌아간 색은 개체를 가리키지 못한다 -- 범례가 거짓이 된다.
    일곱이 모두 같은 이야기를 하므로 띠가 더 정직하고 더 잘 읽힌다. 개별
    값은 보충표에 있다.
    """
    u = pd.read_csv(UNMATCHED).sort_values("n_train")
    ax.plot(u.n_train, u.mean_diff, "-o", color=JS.VERM, zorder=5,
            label="before matching")
    ax.fill_between(u.n_train, u.ci_lo, u.ci_hi, color=JS.VERM, alpha=0.15,
                    lw=0)

    P = pd.read_csv(PER_SZ)
    g = P.groupby("n_train").mean_diff
    x = np.array(sorted(P.n_train.unique()), float)
    med = g.median().reindex(x).to_numpy()
    lo = g.min().reindex(x).to_numpy()
    hi = g.max().reindex(x).to_numpy()
    n_ag = P.antigen.nunique()
    ax.fill_between(x, lo, hi, color=JS.BLUE, alpha=0.18, lw=0)
    ax.plot(x, med, "-o", color=JS.BLUE,
            label=f"after matching, {n_ag} antigens")

    ax.axhline(0, color=JS.MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v):,}" for v in x])
    ax.set_xlabel("training labels")
    ax.set_ylabel("Spearman, one-hot $-$ ESM-2")
    ax.legend(loc="upper left")
    JS.ygrid(ax)


def search_panel(ax):
    r = pd.read_csv(RATIOS)
    r = r[r.budget == BUDGET].copy()
    r["kind"] = np.where(r.arm.str.startswith("matched:"), "matched lattice",
                         "measured")
    r["name"] = r.arm.str.split(":").str[1]
    r = r.sort_values(["kind", "ratio"], ascending=[True, True])
    y = np.arange(len(r))
    col = [JS.VERM if k == "matched lattice" else JS.BLUE for k in r.kind]
    for (_, x), yi, c in zip(r.iterrows(), y, col):
        ax.plot([x.ci_lo, x.ci_hi], [yi, yi], "-", color=c, lw=1.0,
                solid_capstyle="butt", zorder=4)
        for e in (x.ci_lo, x.ci_hi):
            ax.plot([e, e], [yi - 0.14, yi + 0.14], "-", color=c, lw=1.0,
                    zorder=4)
        ax.plot([x.ratio], [yi], "o", color=c, zorder=5)
    ax.axvline(1, color=JS.MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(r.name)
    ax.set_xlabel(f"guided / random recovery of the top 1% "
                  f"(budget {BUDGET:,})")
    ax.set_ylim(-0.7, len(r) + 0.5)
    ax.grid(axis="x", color=JS.GRID, lw=0.4)
    ax.set_axisbelow(True)
    for kind, c in [("matched lattice", JS.VERM), ("measured", JS.BLUE)]:
        ax.plot([], [], "o", color=c, label=kind)
    ax.legend(loc="upper left")


def resid_panel(ax):
    d = pd.read_csv(RESID).dropna(subset=["ratio"])
    for kind, c, m in [("matched lattice", JS.VERM, "o"),
                       ("measured", JS.BLUE, "s")]:
        g = d[d.domain == kind]
        ax.plot(g.rho, g.ratio, m, color=c, label=kind, zorder=5, ls="none")
        for _, x in g.iterrows():
            raw = x.arm.split(":")[1]
            dx, dy, ha = NUDGE.get(raw, (5, 3, "left"))
            arrow = (dict(arrowstyle="-", color=JS.MUTED, lw=0.5,
                          shrinkA=1, shrinkB=2.5)
                     if raw in LEADER else None)
            ax.annotate(raw, (x.rho, x.ratio),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=6.2, color=JS.INK2, ha=ha,
                        arrowprops=arrow, zorder=6)
    ax.set_xlabel("surrogate Spearman at 1,000 labels")
    ax.set_ylabel("guided / random")
    ax.set_xlim(0.40, 1.05)
    ax.set_ylim(1.80, 7.25)          # 위아래로 라벨이 설 자리를 비워 둔다
    ax.legend(loc="upper left")
    JS.ygrid(ax)
    return float(np.corrcoef(d.rho, d.ratio)[0, 1])


def main():
    for p in (PER_SZ, UNMATCHED, RATIOS, RESID):
        if not os.path.exists(p):
            sys.exit(f"{p} 가 없습니다.")
    JS.setup()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(JS.TWO_COL, JS.TWO_COL / 2.9))
    repr_panel(ax)
    fig.subplots_adjust(bottom=0.17, top=0.97, left=0.085, right=0.985)
    w, h = JS.save_fixed_width(fig, FIG_REPR, tight=False)
    print(f"wrote {FIG_REPR}  {w}x{h}px")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(JS.TWO_COL, JS.TWO_COL / 2.7))
    search_panel(axes[0])
    r = resid_panel(axes[1])
    for ax, L in zip(axes, "ab"):
        JS.panel(ax, L)
    fig.subplots_adjust(wspace=0.30, bottom=0.26, top=0.97, left=0.085,
                        right=0.985)
    w, h = JS.save_fixed_width(fig, FIG_SEARCH, tight=False)
    print(f"wrote {FIG_SEARCH}  {w}x{h}px   (b)의 Pearson r = {r:+.2f}")


if __name__ == "__main__":
    main()
