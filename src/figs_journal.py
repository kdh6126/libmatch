"""All submitted figures, in the journal's house style.

Replaces fig_paper_en.py and fig_concept.py. Same data, same analyses; the
presentation changes: white ground, hairline rules, panel letters, no narrative
text inside the image, and page-width sizing. The four-effects grid that used to
be a figure is now a LaTeX table, so this file writes eight figures.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

sys.path.insert(0, "src")
from journal_style import (setup, panel, ygrid, bare, save_fixed_width,
                           TWO_COL, ONE_COL,
                           INK, INK2, MUTED, GRID, PANEL, WHITE, SEQ,
                           BLUE, VERM, GREEN, SKY, ORANGE, PURPLE)

setup()
OUT = "manuscript/figs"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    """폭을 못 박아 저장한다.

    그림은 모두 본문에 width=\\textwidth로 들어가므로, 지면에서의 글자 크기는
    글꼴 pt가 아니라 PNG 폭이 정한다. bbox_inches="tight"의 잘라내기 양이
    그림마다 다르면 같은 7pt도 지면에서는 다른 크기로 인쇄된다 -- 실제로 이
    묶음에서 8.3pt부터 13pt까지 벌어져 있었다.
    """
    p = f"{OUT}/{name}"
    save_fixed_width(fig, p)
    plt.close(fig)
    print(f"wrote {p}")


# The study-design comparison is Table 1 in the manuscript, typeset with
# small graphics in its cells; those are produced by src/fig1_cells.py.
# ==================== Fig 2  matched library construction ==================
# 1NCA_N is withheld: its parent energy disagreed with the oracle and the
# library could not be re-scored (see data/absolut/unreliable/README.txt).
# 정합 라이브러리는 이제 셋이다. 하나였을 때 쓰던 사실 상자를 표로 바꾼다.
LIBS = [("2DD8_S", "CARDRLLWLRLWYFDVW", "SARS-CoV spike RBD"),
        ("1ADQ_A", "CARVGWLLNFSLYFDYW", "human IgG1 Fc"),
        ("1NCA_N", "CAREFDLFGLLLLFDYW", "influenza N9 neuraminidase")]
META = {}
for ag, par, desc in LIBS:
    d = pd.read_parquet(f"data/absolut/matched_{ag}.parquet")
    META[ag] = (par, desc, len(d), d.energy.min(), d.energy.max())

fig = plt.figure(figsize=(TWO_COL, 1.75))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.4], wspace=0.30)
rng = np.random.default_rng(1)

titles = ["Released library", "One parent chosen", "Mutational neighbourhood"]
for k in range(3):
    ax = fig.add_subplot(gs[0, k])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    panel(ax, "abc"[k], drop=0.34)
    ax.set_title(titles[k], fontsize=7.2, pad=8, loc="left", x=0.10)

    if k == 0:
        p = rng.uniform(0.6, 9.4, (150, 2))
        ax.scatter(p[:, 0], p[:, 1], s=3.2, facecolors=VERM, alpha=0.75,
                   edgecolors="none")
        note = "6,917,820 sequences"
    elif k == 1:
        p = rng.uniform(0.6, 9.4, (150, 2))
        ax.scatter(p[:, 0], p[:, 1], s=3.2, facecolors="#cfcfcf",
                   edgecolors="none")
        ax.scatter([5], [5], s=24, facecolors=INK, edgecolors="none", zorder=3)
        ax.annotate("parent", xy=(5.35, 5.35), xytext=(7.3, 8.3), fontsize=6.2,
                    color=INK, ha="left", arrowprops=dict(
                        arrowstyle="-", color=INK, lw=0.5, shrinkA=0, shrinkB=2))
        note = "strongest length-17 binder"
    else:
        ang = rng.uniform(0, 2 * np.pi, 220)
        rad = np.sqrt(rng.uniform(0, 1, 220)) * 3.6
        ax.scatter(5 + rad * np.cos(ang), 5 + rad * np.sin(ang), s=3.2,
                   facecolors=PURPLE, alpha=0.6, edgecolors="none")
        ax.scatter([5], [5], s=24, facecolors=INK, edgecolors="none", zorder=3)
        ax.annotate("parent", xy=(5.35, 5.35), xytext=(7.3, 8.3), fontsize=6.2,
                    color=INK, ha="left", arrowprops=dict(
                        arrowstyle="-", color=INK, lw=0.5, shrinkA=0, shrinkB=2))
        # 285 = 15자리 x 19. 286은 집합에 든 모항체까지 센 값이었다.
        note = "285 singles, plus doubles, triples"
    ax.text(5, -1.4, note, ha="center", va="top", fontsize=6.0,
            color=MUTED)
    if k < 2:
        ax.annotate("", xy=(11.5, 5), xytext=(10.5, 5), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                    mutation_scale=7))

ax = fig.add_subplot(gs[0, 3])
bare(ax)
panel(ax, "d", drop=0.34)
ax.set_title("Scored by the same oracle", fontsize=7.2, pad=8, loc="left",
             x=0.06)

# 세 라이브러리 모두 6,000개이므로 행마다 적지 않고 머리글에 한 번 적는다.
for x, head in [(0, "Antigen"), (36, "Parent CDRH3"), (88, "Energy")]:
    ax.text(x, 92, head, fontsize=6.0, color=MUTED, va="center")
y = 92
for ag, par, desc in LIBS:
    _, _, n, lo, hi = META[ag]
    y -= 22
    ax.text(0, y, ag, fontsize=6.6, color=INK, va="center")
    ax.text(0, y - 8, desc, fontsize=5.6, color=MUTED, va="center")
    ax.text(36, y, par, fontsize=6.2, color=INK, va="center",
            family="monospace")
    ax.text(88, y, f"{lo:.1f} to {hi:.1f}", fontsize=6.2, color=INK,
            va="center")
# the caption already states that the oracle is unchanged; a journal figure
# should not repeat it inside the image
save(fig, "F1.png")


# ======================= Fig 3  search before and after ====================
e2 = pd.read_csv("results/exp2_search.csv")
e2["recall"] = e2["n_top001pct"] / 691.0
e20 = pd.read_csv("results/exp20_matched_search.csv")
e20["domain"] = np.where(e20.arm.str.startswith("real"), "real",
                         "matched_lattice")

fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.5))
fig.subplots_adjust(wspace=0.32)

ax = axes[0]
pts = []
for b in sorted(e2.budget.unique()):
    a = e2[(e2.budget == b) & (e2.strategy == "active")].recall.mean()
    r = e2[(e2.budget == b) & (e2.strategy == "random")].recall.mean()
    pts.append(("Unmatched, 6.9M", b, a / max(r, 1e-9), VERM, "o"))
for dom, c, lab, mk in [("matched_lattice", PURPLE, "Matched lattice, 6,000", "s"),
                        ("real", BLUE, "Measured, 6,000", "^")]:
    s = e20[e20.domain == dom]
    for b in sorted(s.budget.unique()):
        if b < 500:
            continue
        a = s[(s.budget == b) & (s.strategy == "active")].recall.mean()
        r = s[(s.budget == b) & (s.strategy == "random")].recall.mean()
        pts.append((lab, b, a / max(r, 1e-9), c, mk))
P = pd.DataFrame(pts, columns=["arm", "budget", "ratio", "c", "mk"])
for arm, g in P.groupby("arm", sort=False):
    ax.plot(g.budget, g.ratio, color=g.c.iloc[0], marker=g.mk.iloc[0],
            mfc="none", mew=0.9, label=arm)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([500, 1000, 2000, 10000])
ax.set_xticklabels(["500", "1,000", "2,000", "10,000"])
ax.set_yticks([1, 10, 100, 1000])
ax.set_yticklabels(["1", "10", "100", "1,000"])
ax.minorticks_off()
ax.axhline(1, color=MUTED, lw=0.5, ls=(0, (3, 2)))
ax.set_xlabel("Measurement budget")
ax.set_ylabel("Active / random recovery ratio")
# the unmatched curve peaks in the upper right, so the key sits below it
ax.legend(loc="center right", bbox_to_anchor=(1.02, 0.40))
ygrid(ax)
panel(ax, "a")

ax = axes[1]
for dom, c, lab, mk in [("matched_lattice", PURPLE, "Matched lattice", "s"),
                        ("real", BLUE, "Measured", "^")]:
    s = e20[e20.domain == dom]
    for strat, ls in [("active", "-"), ("random", (0, (3, 2)))]:
        g = s[s.strategy == strat].groupby("budget")["recall"]
        ax.plot(g.mean().index, g.mean().values, color=c, ls=ls, marker=mk,
                mfc="none", mew=0.9,
                label=f"{lab}, {'active' if strat == 'active' else 'random'}")
ax.set_xscale("log")
ax.set_xticks([200, 500, 1000])
ax.set_xticklabels(["200", "500", "1,000"])
ax.minorticks_off()
ax.set_xlabel("Measurement budget (library of 6,000)")
ax.set_ylabel("Top 1% recovered")
ax.set_ylim(0, 0.8)
# a longer key line is needed for solid and dashed to read apart
ax.legend(loc="upper left", handlelength=2.6)
ygrid(ax)
panel(ax, "b")
# F2와 F3은 여기서 만들지 않는다. 이 자리의 그림은 항원 하나, 시드 3~5개로
# 그려져 있었고, 시드를 20개로 늘리자 그림이 말하던 것이 바뀌었다(탐색 배수는
# 정합 후에도 실측 범위 위에 남는다). 지금 판은 src/fig_matching_limits.py 가
# 만든다. 여기서 같은 이름으로 저장하면 그 결과를 덮어쓴다.
# save(fig, "F2.png")


# ==================== Fig 4  representation and structure ==================
un = pd.read_csv("results/exp9b_esm_perpos.csv")
un["domain"] = un["arm"].str.split(":").str[0]
u = un[un.domain == "lattice"].pivot_table(index="n_train", columns="features",
                                           values="spearman")
d_un = (u["onehot"] - u["esm_perpos"]).dropna()
E21 = pd.read_csv("results/exp21_seed_expansion.csv")
S21 = pd.read_csv("results/exp21_per_size_tests.csv")

fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.5))
fig.subplots_adjust(wspace=0.30)

ax = axes[0]
ax.axhline(0, color=INK, lw=0.6)
ax.plot(d_un.index, d_un.values, color=VERM, marker="o", mfc="none", mew=0.9,
        label="Unrelated set, 6.9M")
for ag, c, lab, mk in [("2DD8_S", BLUE, "Matched variants, 2DD8_S", "s")]:
    g = S21[S21.antigen == ag].sort_values("n_train")
    ax.fill_between(g.n_train, g.ci_lo, g.ci_hi, color=c, alpha=0.18, lw=0)
    ax.plot(g.n_train, g.mean_diff, color=c, marker=mk, mfc="none", mew=0.9,
            label=lab)
ax.set_xscale("log")
ax.set_xticks([200, 500, 1000, 2000, 5000])
ax.set_xticklabels(["200", "500", "1,000", "2,000", "5,000"])
ax.minorticks_off()
ax.set_xlabel("Training labels")
ax.set_ylabel("Spearman, one-hot minus ESM-2")
ax.legend(loc="upper left")
ygrid(ax)
panel(ax, "a")

ax = axes[1]
lab, val, lo, hi, col = [], [], [], [], []
for ag, c in [("2DD8_S", BLUE)]:
    g = S21[S21.antigen == ag].sort_values("n_train")
    for r in g.itertuples():
        lab.append(f"{r.n_train:,}")
        val.append(r.mean_diff)
        lo.append(r.mean_diff - r.ci_lo)
        hi.append(r.ci_hi - r.mean_diff)
        col.append(c)
pos = np.arange(len(val))[::-1]
ax.axvline(0, color=INK, lw=0.6)
ax.errorbar(val, pos, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.7,
            capsize=1.8)
ax.scatter(val, pos, s=13, facecolors=col, edgecolors="none", zorder=3)
ax.set_yticks(pos)
ax.set_yticklabels(lab, fontsize=6.8)
ax.set_ylabel("Training labels")
ax.set_xlabel("Spearman, one-hot minus ESM-2 (95% CI)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
panel(ax, "b", dx=-0.22)
# F2와 F3은 여기서 만들지 않는다. 이 자리의 그림은 항원 하나, 시드 3~5개로
# 그려져 있었고, 시드를 20개로 늘리자 그림이 말하던 것이 바뀌었다(탐색 배수는
# 정합 후에도 실측 범위 위에 남는다). 지금 판은 src/fig_matching_limits.py 가
# 만든다. 여기서 같은 이름으로 저장하면 그 결과를 덮어쓴다.
# save(fig, "F3.png")


# ======================== Fig 5  prospective split =========================
Z = np.load("results/split_shift.npz")
P5, lab5 = Z["P"], Z["lab"]
d1 = pd.read_csv("results/exp1_alphaseq_baselines.csv")
d1 = d1[d1.model == "rf"]
TASKS = ["T1_random", "T2_prospective"]
TLAB = {"T1_random": "Random", "T2_prospective": "Prospective"}
TCOL = {"T1_random": SKY, "T2_prospective": VERM}

fig = plt.figure(figsize=(TWO_COL, 4.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.55, wspace=0.34)
rng5 = np.random.default_rng(0)


def cloud(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(GRID)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)


ax = fig.add_subplot(gs[0, 0])
held = rng5.random(len(P5)) < 0.2
ax.scatter(P5[~held, 0], P5[~held, 1], s=1.6, c="#cccccc", edgecolors="none",
           label="training")
ax.scatter(P5[held, 0], P5[held, 1], s=1.8, c=VERM, alpha=0.85,
           edgecolors="none", label="held out")
cloud(ax)
ax.legend(loc="lower left", markerscale=3.5)
panel(ax, "a")

ax = fig.add_subplot(gs[0, 1])
ax.scatter(P5[lab5 == 0, 0], P5[lab5 == 0, 1], s=1.6, c=SKY, alpha=0.8,
           edgecolors="none", label="campaign 1, train")
ax.scatter(P5[lab5 == 1, 0], P5[lab5 == 1, 1], s=1.6, c=VERM, alpha=0.6,
           edgecolors="none", label="campaign 2, test")
cloud(ax)
ax.legend(loc="lower left", markerscale=3.5)
panel(ax, "b")

ax = fig.add_subplot(gs[0, 2])
reg = pd.read_parquet("data/alphaseq/panel_cov_reg.parquet")
agn = reg.matalpha_description.value_counts().idxmax()
dd = reg[reg.matalpha_description == agn]
tr0 = dd[dd.iteration == 0]
te1 = dd[dd.iteration == 1]
te1 = te1[~te1.mata_sequence.isin(set(tr0.mata_sequence))]
ref = dd.mata_sequence.value_counts().idxmax()


def muts(g, n=4000):
    s = g.mata_sequence.sample(min(n, len(g)), random_state=0)
    return np.array([sum(a != b for a, b in
                         zip(x.ljust(len(ref))[:len(ref)], ref)) for x in s])


m0, m1 = muts(tr0), muts(te1)
bins = np.arange(0, 41, 2)
ax.hist(np.clip(m0, 0, 40), bins=bins, color=SKY, alpha=0.85,
        label=f"campaign 1 (median {int(np.median(m0))})")
ax.hist(np.clip(m1, 0, 40), bins=bins, color=VERM, alpha=0.6,
        label=f"campaign 2 (median {int(np.median(m1))})")
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xticklabels(["0", "10", "20", "30", "40+"])
ax.set_xlabel("Mutations from parent")
ax.set_ylabel("Sequences")
ax.legend(loc="upper right")
ygrid(ax)
panel(ax, "c")

METRICS = [("spearman", "Spearman", None),
           ("rmse_over_noise", "RMSE / noise", 1.0),
           ("top100@500", "Top 100 recovered", None)]
for k, (m, ylab, refl) in enumerate(METRICS):
    ax = fig.add_subplot(gs[1, k])
    vals = [d1[d1.task == t][m] for t in TASKS]
    ax.bar([0, 1], [v.mean() for v in vals], width=0.55,
           color=[TCOL[t] for t in TASKS], edgecolor="none")
    for i, v in enumerate(vals):
        ax.errorbar(i, v.mean(), yerr=v.std(), fmt="none", ecolor=INK,
                    elinewidth=0.7, capsize=2)
        ax.text(i, v.mean() + v.std() + (0.02 if m != "rmse_over_noise" else 0.1),
                f"{v.mean():.2f}", ha="center", fontsize=7)
    if refl:
        ax.axhline(refl, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.text(1.5, refl + 0.05, "noise floor", ha="right", fontsize=6.4,
                color=MUTED)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([TLAB[t] for t in TASKS])
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.6, 1.6)
    ygrid(ax)
    panel(ax, "def"[k])
save(fig, "F4.png")


# ==================== Supplementary Fig 1  panning =========================
enr = pd.read_csv("results/exp3_enrichment.csv")
rar = pd.read_csv("results/panning_diversity_rarefied.csv")
fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.3))
fig.subplots_adjust(wspace=0.42)

for ax, col, ylab, ltr in [(axes[0], "shannon", "Shannon diversity", "a"),
                           (axes[1], "top10_share", "Top 10 clonotypes (%)", "b")]:
    for chain, c, mk in [("VH", BLUE, "o"), ("VL", VERM, "s")]:
        g = rar[rar.chain == chain].sort_values("round")
        ax.plot(g["round"], g[col], color=c, marker=mk, mfc="none", mew=0.9,
                label=chain)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("Panning round")
    ax.set_ylabel(ylab)
    ax.legend(loc="best")
    ygrid(ax)
    panel(ax, ltr)

ax = axes[2]
rows = [("A_early_rounds", "read counts + ML", SEQ[4]),
        ("A0_baseline_last_early_round", "read counts only", MUTED),
        ("B_sequence_only", "sequence only", BLUE),
        ("B0_permutation_null", "permutation null", "#cccccc")]
vh = enr[enr.chain == "VH"].set_index("task")["auroc"]
vals = [float(vh.get(k, np.nan)) for k, _, _ in rows]
ypos = np.arange(len(rows))[::-1]
ax.barh(ypos, vals, height=0.6, color=[c for _, _, c in rows], edgecolor="none")
for y, v in zip(ypos, vals):
    ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=6.8)
ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)))
ax.set_yticks(ypos)
ax.set_yticklabels([l for _, l, _ in rows], fontsize=6.6)
ax.set_xlim(0, 1.18)
ax.set_xlabel("AUROC, top 1% clonotypes (VH)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
panel(ax, "c", dx=-0.42)
save(fig, "SFig1.png")


# =========================== Fig 6  model zoo ==============================
R = pd.read_csv("results/exp16_model_zoo.csv")
# the target set thins out above 2,000 labels (four targets to 2,000,
# three at 5,000, two at 20,000), so averaging over the full range would
# mix a real trend with a change of composition.
NT = R.groupby("n_train").target.nunique()
FULL = sorted(NT[NT == NT.max()].index)
ZOO_N = max(FULL)
R = R[R.n_train.isin(FULL)]
ORDER = ["knn_hamming", "random_forest", "boosted_trees", "cnn_maxpool",
         "ridge_onehot", "ridge_esm", "ridge_esm+onehot"]
LABEL = {"knn_hamming": "kNN (Hamming)", "random_forest": "Random forest",
         "boosted_trees": "Boosted trees", "cnn_maxpool": "CNN",
         "ridge_onehot": "Ridge, one-hot", "ridge_esm": "Ridge, ESM-2",
         "ridge_esm+onehot": "Ridge, ESM-2 + one-hot"}
COLS = [MUTED, PURPLE, SKY, GREEN, VERM, "#004c7a", BLUE]
MKS = ["x", "v", "D", "^", "s", "P", "o"]

fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.4),
                         gridspec_kw=dict(width_ratios=[1.08, 1]))
fig.subplots_adjust(wspace=0.50)

ax = axes[0]
for m, c, mk in zip(ORDER, COLS, MKS):
    g = R[R.model == m].groupby("n_train")["spearman"]
    ax.plot(g.mean().index, g.mean().values, color=c, marker=mk, mfc="none",
            mew=0.8, ms=2.8, label=LABEL[m])
ax.set_xscale("log")
xs = sorted(R.n_train.unique())
ax.set_xticks(xs)
ax.set_xticklabels([f"{s:,}" for s in xs], fontsize=6.2)
ax.minorticks_off()
ax.set_xlabel("Training labels")
ax.set_ylabel("Spearman (four measured targets)")
# panel b already names every model against the same colour, so a key here
# would only repeat it and there is no room for one that does not sit on a curve
ygrid(ax)
panel(ax, "a")

ax = axes[1]
sub = R[R.n_train == ZOO_N]
tg = sorted(sub.target.unique())
rng7 = np.random.default_rng(0)
ys, los, his, names, cs = [], [], [], [], []
for m, c in zip(ORDER, COLS):
    v = np.array([sub[(sub.target == t) & (sub.model == m)].spearman.mean()
                  for t in tg])
    v = v[~np.isnan(v)]
    if len(v) < 2:
        continue
    b = np.array([rng7.choice(v, len(v), True).mean() for _ in range(10000)])
    ys.append(v.mean())
    los.append(np.percentile(b, 2.5))
    his.append(np.percentile(b, 97.5))
    names.append(LABEL[m])
    cs.append(c)
pos = np.arange(len(ys))[::-1]
ax.errorbar(ys, pos, xerr=[np.array(ys) - los, np.array(his) - np.array(ys)],
            fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.8)
ax.scatter(ys, pos, s=15, facecolors=cs, edgecolors="none", zorder=3)
for v, p in zip(ys, pos):
    ax.text(v, p + 0.26, f"{v:.3f}", ha="center", fontsize=6.4)
ax.set_yticks(pos)
ax.set_yticklabels(names, fontsize=6.6)
ax.set_xlim(0.24, 0.78)
ax.set_xlabel(f"Spearman at {ZOO_N:,} labels (95% CI)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
panel(ax, "b", dx=-0.42)
save(fig, "F5.png")


# ====================== Fig 7  third-party campaign ========================
A = pd.read_csv("results/exp12_featurisation.csv")
NAMES = {"aab-pp489": "PP489\n(anti-TIGIT)",
         "pembrolizumab-scfv": "Pembrolizumab\n(anti-PD-1)",
         "vhh72": "VHH72\n(anti-CoV2 RBD)"}
sub = A[A.comparison.isin(["esm_warm vs ohe_warm", "esm_cold vs ohe_cold"])]
tg = list(NAMES)

fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.4))
fig.subplots_adjust(wspace=0.30)

ax = axes[0]
x = np.arange(len(tg))
w = 0.34
esm = [sub[sub.target == t]["frac_a_improved"].mean() * 100 for t in tg]
ohe = [sub[sub.target == t]["frac_b_improved"].mean() * 100 for t in tg]
ax.bar(x - w / 2, esm, w, color=BLUE, edgecolor="none", label="ESM-2nv")
ax.bar(x + w / 2, ohe, w, color=GREEN, edgecolor="none", label="one-hot")
for xi, (a, b) in enumerate(zip(esm, ohe)):
    ax.text(xi - w / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=6.6)
    ax.text(xi + w / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=6.6)
ax.set_xticks(x)
ax.set_xticklabels([NAMES[t] for t in tg], fontsize=6.6)
ax.set_ylabel("Candidates improving on parent (%)")
ax.set_ylim(0, 108)
ax.legend(loc="upper right")
ygrid(ax)
panel(ax, "a")

ax = axes[1]
ax.axvline(0, color=INK, lw=0.6)
y = np.arange(len(tg))[::-1]
me = [sub[sub.target == t]["median_a"].mean() for t in tg]
mo = [sub[sub.target == t]["median_b"].mean() for t in tg]
ax.barh(y + 0.18, me, 0.32, color=BLUE, edgecolor="none", label="ESM-2nv")
ax.barh(y - 0.18, mo, 0.32, color=GREEN, edgecolor="none", label="one-hot")
for yi, a, b in zip(y, me, mo):
    ax.text(a - 0.05, yi + 0.18, f"{a:+.2f}", va="center", ha="right",
            fontsize=6.4)
    ax.text(b + 0.05 if b > 0 else b - 0.05, yi - 0.18, f"{b:+.2f}",
            va="center", ha="left" if b > 0 else "right", fontsize=6.4)
ax.set_yticks(y)
ax.set_yticklabels([NAMES[t].split("\n")[0] for t in tg], fontsize=6.8)
ax.set_xlim(-2.1, 0.9)
ax.set_xlabel(r"Median change vs parent ($\log_{10}K_D$)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.legend(loc="lower left")
panel(ax, "b", dx=-0.30)
save(fig, "F6.png")

print("\nsix figures plus two supplementary figures")
