"""Shared plotting style: a colour-blind-safe categorical palette, recessive grid and axes,
text in ink colours, plus a small helper that draws pipeline diagrams."""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET, MAGENTA, GRAY = (
    "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#a9a8a2")


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    })


def save(fig, path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def pipeline(path, columns, title, accent=BLUE, width=12.5, height=3.6) -> None:
    """Left-to-right pipeline: ``columns`` is a list of (heading, [items]) drawn as boxes."""
    setup()
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()
    ax.set_xlim(0, len(columns) * 3.2)
    ax.set_ylim(0, 4.4)
    ax.set_title(title, loc="left", pad=8)
    for i, (head, items) in enumerate(columns):
        x = i * 3.2 + 0.1
        ax.add_patch(FancyBboxPatch((x, 0.2), 2.6, 3.4, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=GRID, lw=1.2))
        ax.add_patch(FancyBboxPatch((x, 3.0), 2.6, 0.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=accent, ec=accent))
        ax.text(x + 1.3, 3.3, head, ha="center", va="center", color="white", fontweight="bold",
                fontsize=9.5)
        for k, item in enumerate(items):
            ax.text(x + 0.15, 2.65 - k * 0.5, "• " + item, ha="left", va="center", fontsize=8.6,
                    color=INK2)
        if i < len(columns) - 1:
            ax.annotate("", xy=(x + 3.15, 1.9), xytext=(x + 2.65, 1.9),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.6})
    save(fig, path)
