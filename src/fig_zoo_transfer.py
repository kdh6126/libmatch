"""Figure: whether the benchmark ranks methods as the measured data do.

A benchmark exists to rank methods, so the question is not whether a model
scores well on it but whether the ordering it produces is the ordering that
holds on real measurements. The benchmark's own validation reported a
correlation of 0.9 for method accuracies, obtained for one binary
classification task with classical classifiers. This figure asks the same
question with seven regression models on both arms, as a function of the label
budget.

  (a)  each model's held-out Spearman against training size, both arms
  (b)  the correlation between the two orderings, with an interval

    python src/fig_zoo_transfer.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from journal_style import (setup, panel, ygrid,                 # noqa: E402
                           save_fixed_width, TWO_COL,
                           INK, MUTED, GRID, BLUE, VERM)

setup()
OUT = "manuscript/figs"
LAT = "results/exp82_zoo_transfer.csv"
MEA = "results/exp16_model_zoo.csv"
SUM = "results/exp82_transfer_summary.csv"

SHORT = {"ridge_onehot": "ridge (one-hot)", "ridge_esm": "ridge (ESM)",
         "ridge_esm+onehot": "ridge (both)", "cnn_maxpool": "CNN",
         "random_forest": "random forest", "boosted_trees": "boosted trees",
         "knn_hamming": "nearest neighbour"}


def main():
    for f in (LAT, MEA, SUM):
        if not os.path.exists(f):
            sys.exit(f"{f} 가 없습니다.")
    os.makedirs(OUT, exist_ok=True)
    L = pd.read_csv(LAT)
    M = pd.read_csv(MEA)
    S = pd.read_csv(SUM)
    sizes = sorted(set(L.n_train) & set(M.n_train))
    models = [m for m in SHORT if m in set(L.model) & set(M.model)]

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, TWO_COL * 0.36))
    fig.subplots_adjust(wspace=0.30, bottom=0.30, top=0.93)

    # ---- (a) 두 팔에서의 모델별 곡선
    ax = axes[0]
    lat = L[L.n_train.isin(sizes)].pivot_table(
        index="n_train", columns="model", values="spearman")
    mea = M[M.n_train.isin(sizes)].pivot_table(
        index="n_train", columns="model", values="spearman")
    for m in models:
        if m in lat:
            ax.plot(lat.index, lat[m], "-", color=BLUE, lw=0.9, alpha=0.75)
        if m in mea:
            ax.plot(mea.index, mea[m], "--", color=VERM, lw=0.9, alpha=0.75)
    # 양쪽에서 최고인 모델만 이름을 단다. 일곱 개를 다 달면 읽히지 않는다.
    for frame, colr, dx in ((lat, BLUE, 1.03), (mea, VERM, 1.03)):
        n = frame.index.max()
        best = frame.loc[n, [m for m in models if m in frame]].idxmax()
        ax.annotate(SHORT[best], xy=(n, frame.loc[n, best]),
                    xytext=(n * dx, frame.loc[n, best]), color=colr,
                    fontsize=6.0, va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("Training labels")
    ax.set_ylabel("Held-out Spearman")
    ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 2.6)
    # 로그 축의 기본 눈금은 2x10^2, 3x10^2 ... 을 찍어 서로 겹친다.
    # 실제로 재어 본 크기만 남긴다.
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.minorticks_off()
    h = [plt.Line2D([], [], color=BLUE, lw=1.1),
         plt.Line2D([], [], color=VERM, lw=1.1, ls="--")]
    ax.legend(h, ["matched lattice, 7 antigens", "measured, 4 targets"],
              loc="lower right", frameon=False, fontsize=6.0,
              handlelength=1.8, borderpad=0.2)
    ygrid(ax)
    panel(ax, "a", drop=0.26)

    # ---- (b) 두 순서의 상관
    ax = axes[1]
    x = np.arange(len(S))
    ax.fill_between(x, S.lo, S.hi, color=GRID, zorder=1)
    ax.plot(x, S.rho, "o-", color=INK, lw=1.2, ms=5, zorder=3)
    for i, r in S.iterrows():
        mark = "same" if r.same_best else "differs"
        ax.annotate(mark, xy=(i, r.rho), xytext=(i, r.rho - 0.085),
                    color=INK if r.same_best else VERM, fontsize=5.8,
                    ha="center", va="top")
    ax.axhline(0.9, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(len(S) - 0.5, 0.905, "0.9 reported for the benchmark",
            color=MUTED, fontsize=5.8, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n):,}" for n in S.n_train])
    ax.set_ylim(0.35, 1.04)
    ax.set_xlabel("Training labels")
    ax.set_ylabel("Correlation between the two orderings")
    ygrid(ax)
    panel(ax, "b", drop=0.26)

    p = f"{OUT}/F10.png"
    save_fixed_width(fig, p)
    plt.close(fig)
    print(f"wrote {p}")
    print(f"  상관 {S.rho.iloc[0]:+.3f} (n={int(S.n_train.iloc[0]):,}) → "
          f"{S.rho.iloc[-1]:+.3f} (n={int(S.n_train.iloc[-1]):,})")
    print(f"  최고 모델 일치 {int(S.same_best.sum())}/{len(S)}")


if __name__ == "__main__":
    main()
