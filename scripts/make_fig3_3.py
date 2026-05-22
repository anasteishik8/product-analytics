"""
make_fig3_3.py — recursive forecast с лаг-признаками.

Концептуальная горизонтальная цепочка из 5 блоков:
    [Состояние T] → [Шаг h=1] → [Шаг h=2] → [ ... ] → [Шаг h=14]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, save_figure  # noqa: E402


OUT_PATH = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig3_3.pdf"


def _box(ax, x, y, w, h, title, body, face, edge="#333333", lw=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.22, title,
            ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.55, body,
            ha="center", va="top", fontsize=9.0)


def _arrow(ax, x_left, y, x_right):
    ax.add_patch(FancyArrowPatch(
        (x_left, y), (x_right, y),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.1,
        color="#444444",
    ))


def build_figure() -> plt.Figure:
    apply_vkr_style()
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6)
    ax.set_axis_off()

    bw, bh = 2.4, 2.5
    y0 = 2.2
    gap = 0.35

    blocks = [
        ("Состояние T",
         "target_lag1,\ntarget_lag7,\nпрочие признаки",
         "#e3edf5"),
        ("Шаг h = 1",
         r"предсказать $\hat y_{T+1}$" + "\n" + r"обновить lag$_1 \leftarrow \hat y_{T+1}$",
         "#fbf2e3"),
        ("Шаг h = 2",
         r"предсказать $\hat y_{T+2}$" + "\n" + r"обновить lag$_1 \leftarrow \hat y_{T+2}$",
         "#fbf2e3"),
        ("…",
         "продолжение\nрекурсии",
         "#f4f4f4"),
        ("Шаг h = 14",
         r"предсказать $\hat y_{T+14}$" + "\n" + "финальный прогноз",
         "#cde9d5"),
    ]

    xs = []
    x = 0.4
    for title, body, face in blocks:
        edge = "#3a8a64" if title == "Шаг h = 14" else "#333333"
        lw = 1.6 if title == "Шаг h = 14" else 1.0
        _box(ax, x, y0, bw, bh, title, body, face, edge=edge, lw=lw)
        xs.append((x, x + bw))
        x += bw + gap

    # Стрелки между блоками
    arr_y = y0 + bh / 2
    for i in range(len(blocks) - 1):
        _arrow(ax, xs[i][1] + 0.02, arr_y, xs[i + 1][0] - 0.02)

    # Подпись внутри figure
    ax.text(7.25, 0.5,
            "На каждом шаге предсказанное значение становится новым "
            r"target_lag$_1$ для следующего шага.",
            ha="center", va="center", fontsize=10, style="italic",
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"))

    fig.tight_layout()
    return fig


def main() -> None:
    fig = build_figure()
    save_figure(fig, OUT_PATH)
    print(f"OK fig3_3 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
