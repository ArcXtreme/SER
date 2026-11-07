"""
Shared matplotlib style so every figure in the report/slides looks consistent.

Colour rules followed (from a validated colour-blind-safe palette):
  * Categorical colours are assigned in a FIXED order and always follow the entity
    (e.g. "angry" is the same colour in every figure), never its rank.
  * Single-series charts use one colour - no rainbow bars.
  * Text (labels, values, legends) uses ink colours, never the series colour.
  * Sequential data (confusion matrices) uses one hue, light -> dark.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical palette, light mode, fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Fixed entity -> colour maps (identity never changes between figures)
from src.preprocessing.dataset_index import CLASSES  # noqa: E402

EMOTION_COLOR = {c: SERIES[i] for i, c in enumerate(CLASSES)}
EMOTION_COLOR.update({"calm": "#c3c2b7", "surprised": "#898781"})  # excluded classes: neutral greys
DATASET_COLOR = {"RAVDESS": SERIES[0], "CREMA-D": SERIES[1]}
# Model identity for Stage 5 comparison figures (fixed for E1-E4 so a model keeps its colour)
MODEL_COLOR = {"E1-SVM": SERIES[6], "E1-RF": SERIES[3], "E2": SERIES[2], "E3": SERIES[4], "E4": SERIES[0],
               "E3-SA": SERIES[7]}

# One-hue sequential ramp (blue 100 -> 700) for heat-maps / confusion matrices
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply() -> None:
    """Apply the project-wide rcParams. Call once at the top of any plotting script."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig: plt.Figure, path) -> None:
    """Save as PNG (slides/report) and close to free memory."""
    fig.savefig(path)
    plt.close(fig)
