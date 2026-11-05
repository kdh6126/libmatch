"""Figure: what the lattice oracle computes, and how the two libraries sit on it.

Every coordinate here is decoded from Absolut!'s own structure records -- the
antigen from its definition in antigenLib.cpp, the CDRH3 poses from the binding
files, ours and the released ones. The decoder is checked by verify_lattice.py
before anything is drawn.

The point of the figure is the paper's point made structurally: an unrelated
CDRH3 set and a mutational library around one parent do not merely differ in
size, they occupy the antigen surface differently.
"""
import os
import re
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, "src")
from lattice_geom import decode, contacts
# 이 그림만 viz_style(회색 바탕, 그림 안 제목)을 쓰고 있었다. 나머지 그림은
# journal_style이라 같은 지면에서 바탕색과 글자 크기가 어긋나 보였다. 여기서
# 집 스타일로 맞춘다 -- 흰 바탕, 패널 글자는 그림 아래 (a) 형태, 설명은
# 그림 안이 아니라 캡션이 진다.
from journal_style import (setup, panel, save_fixed_width,   # noqa
                           TWO_COL, INK, INK2, MUTED,
                           GRID, WHITE, BLUE, ORANGE, GREEN)

AG = "2DD8_S"
SRC = "tools/Absolut-main/src/Absolut/antigenLib.cpp"
# 이 그림 전용 채점 결과. 항원 이름만 쓰는 공용 파일은 다른 실험이
# 덮어쓰므로 접두어를 붙여 따로 둔다.
POSES = ("tools/Absolut-main/src/figbind_2DD8_SFinalBindings_Process_1_Of_1.txt")
LIB = "data/absolut/matched_2DD8_S.parquet"
RAW = "data/absolut/raw/2DD8_S.zip"
OUT = "manuscript/figs/F9.png"
NSAMPLE = 20000

setup()
SURFACE = WHITE
AQUA = GREEN                      # Okabe-Ito 쪽으로 맞춘 강조색


def panel3d(ax, letter, drop=0.06):
    """3D 축의 패널 글자. Axes3D는 text()가 (x, y, z, s)라 text2D를 쓴다."""
    ax.text2D(0.5, drop, f"({letter})", transform=ax.transAxes, fontsize=8,
              fontweight="bold", va="top", ha="center", color=INK)


def read_antigen(code):
    txt = open(SRC, encoding="utf-8", errors="replace").read()
    b = txt[txt.index(f'str2int("{code}")'):][:4000]
    return (re.search(r'agStruct = string\("([SBRLUD]+)"\)', b).group(1),
            re.search(r'agSeq = string\("([A-Z]+)"\)', b).group(1),
            int(re.search(r"startPos = (\d+)", b).group(1)),
            sorted({int(v) for v in re.findall(
                r"\d+", re.search(r"hotspotsCore = \{(.+?)\};", b, re.S).group(1))}))


def best_poses(lines, limit):
    """Take the winning slice of each CDR3 and decode its pose."""
    out, header = [], None
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if header is None:
            header = p
            bi = next(i for i, c in enumerate(header) if c.startswith("Best"))
            si, ei = header.index("Structure"), header.index("Energy")
            continue
        if len(p) != len(header) or p[bi] != "true":
            continue
        try:
            sp, mv = p[si].split("-")
            out.append((decode(int(sp), mv), float(p[ei])))
        except (ValueError, KeyError):
            continue
        if len(out) >= limit:
            break
    return out


def matched_poses():
    """정합 라이브러리 6,000개의 자세. 없거나 수가 안 맞으면 다시 채점한다.

    이 그림은 원래 tools/Absolut-main/src의 FinalBindings 파일을 그대로 읽었다.
    그런데 그 이름은 항원마다 하나뿐이라 같은 항원을 쓰는 다른 실험이 덮어쓴다 --
    실제로 대조 실험(천연 1 + 무작위 200)이 지나간 뒤 이 그림이 6,000개가 아니라
    201개의 자세로, 최저 에너지 -126.5가 아니라 -112.6으로 그려졌다. 그림은
    조용히 만들어졌고 숫자만 달라졌다.

    그래서 남의 임시 파일에 기대지 않는다. 권위 있는 것은 라이브러리 자체
    (matched_2DD8_S.parquet)이므로 거기서 채점을 다시 만들고, 돌아온 에너지가
    저장된 값과 같은지 확인한 뒤에 쓴다.
    """
    lib = pd.read_parquet(LIB)
    if os.path.exists(POSES):
        got = best_poses(open(POSES, encoding="utf-8", errors="replace"),
                         NSAMPLE)
        if len(got) >= len(lib) * 0.95:
            print(f"matched poses reused from {POSES}")
            return got
        print(f"{POSES}의 자세가 {len(got)}개뿐이라 다시 채점합니다")

    tag = "figbind"
    inp = f"tools/Absolut-main/src/{tag}_input.txt"
    with open(inp, "w", newline="\n") as fh:
        for i, s in enumerate(lib.cdr3):
            fh.write(f"{i}\t{s}\n")
    wsl = "/mnt/d/ubuntu/BIO/Infectious/tools/Absolut-main/src"
    out = f"{AG}FinalBindings_Process_1_Of_1.txt"
    cmd = (f"cd {wsl} && rm -f {AG}FinalBindings_Process_*.txt && "
           f"./AbsolutNoLib repertoire {AG} {tag}_input.txt 8 > /dev/null 2>&1 "
           f"&& cp {out} {tag}_{out}")
    subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc", cmd],
                   check=False)
    got = best_poses(open(POSES, encoding="utf-8", errors="replace"), NSAMPLE)
    # 저장된 에너지와 대조한다. 다르면 다른 라이브러리를 그린 것이다.
    e_new = min(e for _, e in got)
    e_old = float(lib.energy.min())
    if abs(e_new - e_old) > 1e-6:
        raise SystemExit(f"에너지가 라이브러리와 다릅니다: 채점 {e_new} 대 "
                         f"저장 {e_old}. 같은 라이브러리가 아닙니다.")
    print(f"matched poses rescored: {len(got):,}, best {e_new:.2f} "
          f"(library {e_old:.2f}) — 일치")
    return got


struct, seq, start, hotspots = read_antigen(AG)
ag = decode(start, struct)
print(f"antigen {AG}: {len(ag)} residues, hotspots {hotspots}")

matched = matched_poses()
with zipfile.ZipFile(RAW) as z:
    name = [n for n in z.namelist() if n.endswith(".txt")][0]
    with z.open(name) as f:
        unrelated = best_poses(f, NSAMPLE)
print(f"matched library poses   {len(matched):,}")
print(f"unrelated library poses {len(unrelated):,}")


def footprint(poses):
    """How often each antigen residue is contacted, as a fraction of poses."""
    f = np.zeros(len(ag))
    for c, _ in poses:
        for j in {j for _, j in contacts(c, ag)}:
            f[j] += 1
    return f / max(len(poses), 1)


fp_m, fp_u = footprint(matched), footprint(unrelated)


def chain(ax, c, color, lw=2.0, alpha=1.0):
    seg = np.stack([c[:-1], c[1:]], axis=1)
    ax.add_collection3d(Line3DCollection(seg, colors=color, linewidths=lw,
                                         alpha=alpha))


def frame(ax, c):
    # 여백을 최소로 둔다. 3D 축은 자기 상자 안에서 이미 작게 그려지는데
    # 여기에 여유까지 주면 지면에서 구조가 알아보기 어려울 만큼 줄어든다.
    mid = c.mean(0)
    r = (c.max(0) - c.min(0)).max() / 2 + 0.5
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-62)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


fig = plt.figure(figsize=(TWO_COL, TWO_COL * 0.30))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.18], wspace=0.02,
                      left=0.005, right=0.995, bottom=0.13, top=1.0)

# ---- (a) the antigen with the strongest binder docked -----------------------
# 범례를 넣지 않는다. 3D 축에서는 범례가 축 바깥으로 밀려 나가 bbox를 늘리고,
# 그만큼 그림이 잘려 지면에서 글자가 커진다. 색이 뜻하는 바는 캡션이 진다.
ax = fig.add_subplot(gs[0, 0], projection="3d")
ax.set_facecolor(SURFACE)
chain(ax, ag, "#c9c8c3", lw=1.1)
ax.scatter(*ag.T, s=4, c="#c9c8c3", depthshade=False, edgecolors="none")
hs = ag[hotspots]
ax.scatter(*hs.T, s=20, c=AQUA, depthshade=False, edgecolors=SURFACE,
           linewidths=0.5)

bestc, beste = min(matched, key=lambda t: t[1])
chain(ax, bestc, ORANGE, lw=1.8)
ax.scatter(*bestc.T, s=17, c=ORANGE, depthshade=False, edgecolors=SURFACE,
           linewidths=0.5)
frame(ax, ag)
panel3d(ax, "a")

# ---- (b) where each library puts its poses ----------------------------------
ax = fig.add_subplot(gs[0, 1], projection="3d")
ax.set_facecolor(SURFACE)
chain(ax, ag, "#e2e1dd", lw=1.0)
rng = np.random.default_rng(0)


def centroids(poses, n=700):
    idx = rng.choice(len(poses), min(n, len(poses)), replace=False)
    return np.array([poses[i][0].mean(0) for i in idx])


cu, cm = centroids(unrelated), centroids(matched)
ax.scatter(*cu.T, s=16, c=ORANGE, alpha=0.40, depthshade=False,
           edgecolors="none")
ax.scatter(*cm.T, s=16, c=BLUE, alpha=0.65, depthshade=False,
           edgecolors="none")
frame(ax, ag)
sp_u = np.linalg.norm(cu - cu.mean(0), axis=1).mean()
sp_m = np.linalg.norm(cm - cm.mean(0), axis=1).mean()
panel3d(ax, "b")

# ---- (c) contact footprint along the antigen --------------------------------
ax = fig.add_subplot(gs[0, 2])
x = np.arange(len(ag))
ax.fill_between(x, fp_u * 100, color=ORANGE, alpha=0.75, lw=0,
                label="unrelated CDRH3 set")
ax.fill_between(x, fp_m * 100, color=BLUE, alpha=0.75, lw=0,
                label="variants of one parent")
for h in hotspots:
    ax.axvline(h, color=AQUA, lw=0.7, ls=(0, (3, 2)), zorder=0)
ax.plot([], [], color=AQUA, lw=0.7, ls=(0, (3, 2)), label="annotated hotspot")
ax.set_xlabel("Antigen residue")
ax.set_ylabel("Poses contacting this residue (%)")
ax.set_xlim(0, len(ag) - 1)
ax.legend(frameon=False, fontsize=6.2, labelcolor=INK, loc="upper right",
          handlelength=1.2, handletextpad=0.4, borderpad=0.2,
          borderaxespad=0.2)
ax.grid(axis="y", color=GRID, lw=0.4)
ax.set_axisbelow(True)
panel(ax, "c", drop=0.17)

save_fixed_width(fig, OUT)
print(f"\nwrote {OUT}")
# 캡션이 져야 할 수치를 찍어 둔다. 그림 안에 적지 않으므로 여기서 읽어
# 캡션에 옮긴다 -- 값이 바뀌면 캡션도 바뀌어야 한다는 것을 잊지 않도록.
print("caption numbers")
print(f"  antigen {AG}, {len(ag)} residues, best energy {beste:.1f}")
print(f"  mean spread of pose centres: unrelated {sp_u:.1f}, "
      f"matched {sp_m:.1f} lattice units")
print(f"  residues contacted by >5% of poses: unrelated {(fp_u > .05).sum()}, "
      f"matched {(fp_m > .05).sum()}")
