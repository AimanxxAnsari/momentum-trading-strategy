"""Chart style shared by the notebooks: one colour-blind-checked palette and consistent axes.

    from mts import plotting as plot
    plot.apply_style()
    fig, ax = plt.subplots(); ...; plot.save(fig, "name")
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from . import config as C

SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#e34948"
LIGHT_BLUE = "#86b6ef"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
pct_axis = PercentFormatter(1.0, decimals=0)


def apply_style() -> None:
    from cycler import cycler
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
        "axes.prop_cycle": cycler(color=SERIES),
    })


def save(fig, name: str) -> None:
    """Save to images/<name>.png at 150 dpi and show it."""
    C.IMAGE_DIR.mkdir(exist_ok=True)
    fig.savefig(C.IMAGE_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="semibold")
