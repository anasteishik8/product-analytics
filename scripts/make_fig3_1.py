"""
make_fig3_1.py — Общая методическая схема системы (главный рисунок главы 3).

Вертикальная цепочка из 8 шагов:
    1. Данные и признаки
    2. Модели-кандидаты (горизонтальная полоса из 6 мини-блоков)
    3. Feature selection + нормализация
    4. Кросс-валидация / начальное обучение
    5. Recursive forecast validation (14 дней)
    6. Рабочая модель per (продукт, метрика)  — акцентный блок
    7. Сценарный анализ + SHAP
    8. Viability verdict
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, save_figure  # noqa: E402


OUT_PATH = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig3_1.pdf"


COLOR_NEUTRAL = "#e3edf5"
COLOR_CANDIDATE = "#fbf2e3"
COLOR_PIVOT = "#d6e4f0"
COLOR_WORKING = "#cde9d5"
COLOR_DOWNSTREAM = "#e8f0fa"
COLOR_VERDICT = "#f4d9d4"


def _box(ax, x, y, w, h, text, face, edge="#333333", lw=1.0, fontsize=11, bold=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True)


def _arrow(ax, x, y_top, y_bot):
    ax.add_patch(FancyArrowPatch(
        (x, y_top), (x, y_bot),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.1,
        color="#444444",
    ))


def build_figure() -> plt.Figure:
    apply_vkr_style()
    fig, ax = plt.subplots(figsize=(10.0, 12.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_axis_off()

    # Геометрия: центр x = 5.0, ширина обычного блока 6.0
    cx = 5.0
    w = 6.4
    x = cx - w / 2
    h = 0.85

    # координаты y (сверху вниз)
    y1 = 12.7
    y2 = 11.2  # полоса моделей-кандидатов (выше из-за высоты)
    y3 = 9.6
    y4 = 8.3
    y5 = 7.0
    y6 = 5.5  # рабочая модель — крупнее
    y7 = 3.9
    y8 = 2.4

    # ─── шаг 1
    _box(ax, x, y1, w, h, "Данные и признаки", COLOR_NEUTRAL, fontsize=12, bold=True)

    # ─── шаг 2: горизонтальная полоса из 6 мини-блоков
    # Общая «обёрточная» рамка
    cand_h = 1.10
    cand_w = 7.6
    cand_x = cx - cand_w / 2
    cand_y = y2 - 0.05
    outer = FancyBboxPatch(
        (cand_x, cand_y), cand_w, cand_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.0,
        edgecolor="#777777",
        facecolor="#fdf7ec",
    )
    ax.add_patch(outer)
    ax.text(cx, cand_y + cand_h - 0.18, "Модели-кандидаты",
            ha="center", va="center", fontsize=11, weight="bold")

    candidates = ["Ridge", "Nadaraya–\nWatson", "LocalLinear",
                  "Random\nForest", "XGBoost", "kNN"]
    n = len(candidates)
    inner_pad = 0.18
    inner_w = (cand_w - 2 * inner_pad) / n - 0.05
    inner_h = 0.55
    inner_y = cand_y + 0.12
    for i, name in enumerate(candidates):
        ix = cand_x + inner_pad + i * (inner_w + 0.05)
        mini = FancyBboxPatch(
            (ix, inner_y), inner_w, inner_h,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=0.7,
            edgecolor="#8a6a2a",
            facecolor=COLOR_CANDIDATE,
        )
        ax.add_patch(mini)
        ax.text(ix + inner_w / 2, inner_y + inner_h / 2, name,
                ha="center", va="center", fontsize=9)

    # ─── шаг 3
    _box(ax, x, y3, w, h, "Отбор признаков и нормализация",
         COLOR_NEUTRAL, fontsize=11)

    # ─── шаг 4
    _box(ax, x, y4, w, h, "Кросс-валидация и начальное обучение",
         COLOR_NEUTRAL, fontsize=11)

    # ─── шаг 5: pivot
    _box(ax, x, y5, w, h, "Recursive forecast validation (горизонт 14 дней)",
         COLOR_PIVOT, edge="#2c5d8a", lw=1.4, fontsize=11, bold=True)

    # ─── шаг 6: акцент
    big_w = 7.0
    big_h = 1.05
    bx = cx - big_w / 2
    _box(ax, bx, y6, big_w, big_h,
         "Рабочая модель для каждой пары\nпродукт-метрика",
         COLOR_WORKING, edge="#3a8a64", lw=2.0, fontsize=12, bold=True)

    # ─── шаг 7
    _box(ax, x, y7, w, h, "Сценарный анализ Monte Carlo + SHAP",
         COLOR_DOWNSTREAM, fontsize=11)

    # ─── шаг 8: итог
    _box(ax, x, y8, w, h, "Вердикт продукта: DEVELOP / MONITOR / DISCONTINUE",
         COLOR_VERDICT, edge="#a83232", lw=1.3, fontsize=11, bold=True)

    # ─── Стрелки между блоками
    arrow_xs = cx
    _arrow(ax, arrow_xs, y1, y2 + cand_h + 0.05)
    _arrow(ax, arrow_xs, y2, y3 + h + 0.04)  # из полосы кандидатов вниз
    _arrow(ax, arrow_xs, y3, y4 + h + 0.04)
    _arrow(ax, arrow_xs, y4, y5 + h + 0.04)
    _arrow(ax, arrow_xs, y5, y6 + big_h + 0.04)
    _arrow(ax, arrow_xs, y6, y7 + h + 0.04)
    _arrow(ax, arrow_xs, y7, y8 + h + 0.04)

    fig.tight_layout()
    return fig


def main() -> None:
    fig = build_figure()
    save_figure(fig, OUT_PATH)
    print(f"OK fig3_1 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
