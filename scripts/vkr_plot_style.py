"""
vkr_plot_style.py — единый matplotlib стиль для всех рисунков ВКР.

Usage:
    from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure
    apply_vkr_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color=VKR_PALETTE[0])
    save_figure(fig, "vkr/v2/artifacts/figures/figX_Y.pdf")
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt


# Палитра: 4 основных + 1 акцент. Спокойные холодные тона + красный для рисков.
VKR_PALETTE: Sequence[str] = (
    "#2c5d8a",  # глубокий синий (основной — F1)
    "#a83232",  # тёмно-красный (контраст — F2 / риск)
    "#3a8a64",  # зелёный (DEVELOP / положительные сценарии)
    "#c08a30",  # охра (нейтральный / MONITOR)
    "#555555",  # серый (baseline / annotations)
)


def apply_vkr_style() -> None:
    """Выставляет matplotlib rcParams для единого стиля ВКР."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.prop_cycle": matplotlib.cycler(color=list(VKR_PALETTE)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "figure.figsize": (6.0, 4.0),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })


def save_figure(fig: matplotlib.figure.Figure, out_path: Path | str) -> Path:
    """Сохраняет рисунок в PDF (vector) с применёнными настройками."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    return out_path
