"""
make_fig3_5.py — Monte Carlo сценарный анализ.

Поток:
    [Опорное состояние]
        ↓
    [Изменение управляемых признаков]
        ↓
    [N симуляций с шумом]
        ↓
    [Распределение прогнозов]  →  [Медиана + CI95 + p > опорной]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, save_figure  # noqa: E402


OUT_PATH = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig3_5.pdf"


def _box(ax, x, y, w, h, title, body, face, edge="#333333", lw=1.0, title_size=10.5, body_size=9.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.18, title,
            ha="center", va="top", fontsize=title_size, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.50, body,
                ha="center", va="top", fontsize=body_size)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.1,
        color="#444444",
    ))


def build_figure() -> plt.Figure:
    apply_vkr_style()
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_axis_off()

    # ─── 1. Опорное состояние
    _box(ax, 3.0, 10.0, 6.0, 1.4,
         "Опорное состояние",
         "значения признаков в текущей точке",
         "#eeeeee", edge="#555555")

    # ─── 2. Изменение управляемых признаков
    _box(ax, 3.0, 7.8, 6.0, 1.6,
         "Изменение управляемых признаков",
         "crash_rate, onboarding_completion_rate,\nmedian_session_duration",
         "#e3edf5", edge="#2c5d8a")

    # ─── 3. N симуляций
    _box(ax, 3.0, 5.6, 6.0, 1.4,
         "N симуляций с гауссовским шумом",
         "seed = 42, N = 1000",
         "#cde9d5", edge="#3a8a64")

    # ─── 4. Распределение прогнозов (слева)
    _box(ax, 0.6, 3.0, 5.0, 1.6,
         "Распределение прогнозов",
         "по результатам N симуляций",
         "#fbf2e3", edge="#c08a30")

    # ─── 5. Итог
    _box(ax, 6.4, 3.0, 5.0, 1.6,
         "Медиана, CI95,\nвероятность p > опорной траектории",
         "",
         "#f4d9d4", edge="#a83232", lw=1.4)

    # Стрелки (вертикальные)
    _arrow(ax, 6.0, 10.0, 6.0, 9.42)   # 1→2
    _arrow(ax, 6.0, 7.8, 6.0, 7.02)    # 2→3
    # 3 → распределение (диагональ вниз-влево к центру блока распределения)
    _arrow(ax, 5.5, 5.6, 3.6, 4.65)
    # 3 → итог  (диагональ вниз-вправо к центру блока итог)
    _arrow(ax, 6.5, 5.6, 8.4, 4.65)
    # горизонтальная: распределение → итог
    _arrow(ax, 5.62, 3.8, 6.38, 3.8)

    # Подпись внизу
    fig.text(0.5, 0.02,
             "Сценарный анализ Monte Carlo: изменение управляемых признаков → "
             "распределение прогнозов → агрегированная оценка эффекта.",
             ha="center", va="bottom", fontsize=10, style="italic")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main() -> None:
    fig = build_figure()
    save_figure(fig, OUT_PATH)
    print(f"OK fig3_5 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
