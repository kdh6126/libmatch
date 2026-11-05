"""House style for the submitted figures.

Nature Portfolio conventions: white ground, hairline spines, outward ticks,
sans-serif at 7-8 pt, lower-case bold panel letters set outside the axes, and
no explanatory text inside the image -- the caption carries all of it.

Widths are the journal's: 89 mm for one column, 183 mm for two. Colours are the
Okabe-Ito set, which is colour-blind safe and survives greyscale printing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
ONE_COL = 89 * MM
TWO_COL = 183 * MM

# Okabe-Ito
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERM = "#D55E00"
PURPLE = "#CC79A7"

INK = "#000000"
INK2 = "#3f3f3f"
MUTED = "#767676"
GRID = "#d9d9d9"
PANEL = "#f2f2f2"          # light fill for schematic boxes
WHITE = "#ffffff"

SEQ = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]


def setup():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.unicode_minus": False,

        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": "none",

        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.3,

        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,

        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "patch.linewidth": 0.6,

        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


TARGET_PX = 3600           # 지면에서 183mm로 실었을 때 다른 그림과 같은 눈금


def save_fixed_width(fig, path, target_px=TARGET_PX, tight=True):
    """폭을 못 박아 저장한다. 그림마다 글자 크기가 달라지는 것을 막는다.

    모든 그림은 본문에 `width=\\textwidth`로 들어가므로, 지면에서의 글자 크기는
    글꼴 pt가 아니라 **PNG 폭**이 정한다. 같은 7pt로 그려도 2,400px짜리는
    3,600px짜리보다 1.5배 크게 인쇄된다.

    실제로 그랬다. 기존 그림은 3,600px 안팎인데 나중에 추가한 둘은
    bbox_inches="tight"에 잘려 2,249px와 2,420px가 됐고, 지면에서 글자가
    50~60% 컸다. 눈으로는 "이 그림만 글씨가 크다"로 보인다.

    잘라내기를 포기하지 않고 폭만 맞춘다. 한 번 저장해 실제 폭을 재고, 거기서
    필요한 dpi를 역산해 다시 저장한다. 내용에 따라 여백이 달라져도 결과 폭은
    항상 같다.
    """
    import os

    from PIL import Image

    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(path, dpi=300, **kw)
    w = Image.open(path).size[0]
    fig.savefig(path, dpi=max(72, round(300 * target_px / w)), **kw)
    w2, h2 = Image.open(path).size
    if abs(w2 - target_px) / target_px > 0.02:
        print(f"  [주의] {os.path.basename(path)} 폭 {w2}px "
              f"(목표 {target_px}). 다른 그림과 글자 크기가 다를 수 있다.")
    return w2, h2


def panel(ax, letter, dx=None, dy=None, drop=0.22):
    """Panel label as (a), centred below the axes.

    dx and dy are accepted so that older call sites keep working; they placed
    the letter above and to the left, which is no longer where it goes.
    """
    ax.text(0.5, -drop, f"({letter})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="center", color=INK)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def bare(ax):
    """A blank canvas for a schematic, in figure-relative 0-100 coordinates."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
