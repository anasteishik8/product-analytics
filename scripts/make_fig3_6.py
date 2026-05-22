"""
make_fig3_6.py — SHAP как мост от модели к сценариям с шагом operational whitelist.

Вертикальный flow из 4 блоков, 3 стрелки:
    [Обученная модель]
       ↓ SHAP
    [Важные признаки]
       ↓ operational whitelist
    [Управляемые признаки]
       ↓
    [Управляемые сценарии Monte Carlo]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, save_figure  # noqa: E402


OUT_PATH = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig3_6.pdf"


def _box(ax, x, y, w, h, title, body, face, edge="#333333", lw=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.20, title,
            ha="center", va="top", fontsize=11.0, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.55, body,
                ha="center", va="top", fontsize=9.2)


def _arrow_with_label(ax, x, y_top, y_bot, label):
    ax.add_patch(FancyArrowPatch(
        (x, y_top), (x, y_bot),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#444444",
    ))
    if label:
        ax.text(x + 0.18, (y_top + y_bot) / 2, label,
                ha="left", va="center", fontsize=9.5, style="italic",
                color="#444444")


def build_figure() -> plt.Figure:
    apply_vkr_style()
    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_axis_off()

    cx = 6.0
    w = 7.6
    x = cx - w / 2

    # координаты сверху вниз
    y1, h1 = 10.0, 1.2
    y2, h2 = 7.6, 1.6
    y3, h3 = 5.0, 1.6
    y4, h4 = 2.4, 1.4

    # ─── 1. Обученная модель
    _box(ax, x, y1, w, h1,
         "Обученная рабочая модель",
         "прогноз для пары (продукт, метрика)",
         "#e3edf5", edge="#2c5d8a")

    # ─── 2. Важные признаки
    _box(ax, x, y2, w, h2,
         "Значимые признаки (по SHAP)",
         "category_rank, crash_rate, retention_d7,\n"
         "onboarding_completion_rate, …",
         "#fbf2e3", edge="#c08a30")

    # ─── 3. Управляемые признаки
    _box(ax, x, y3, w, h3,
         "Управляемые признаки",
         "crash_rate, onboarding_completion_rate,\n"
         "median_session_duration",
         "#cde9d5", edge="#3a8a64", lw=1.4)

    # ─── 4. Сценарии MC
    _box(ax, x, y4, w, h4,
         "Управляемые сценарии Monte Carlo",
         "оценка эффекта операционных изменений",
         "#f4d9d4", edge="#a83232", lw=1.3)

    # Стрелки + подписи
    _arrow_with_label(ax, cx, y1, y2 + h2, "SHAP")
    _arrow_with_label(ax, cx, y2, y3 + h3, "operational whitelist")
    _arrow_with_label(ax, cx, y3, y4 + h4, "")

    # Подпись про whitelist
    ax.text(cx + 1.6, (y2 + y3 + h3) / 2 - 0.05,
            "отделяет операционно\nуправляемые метрики",
            ha="left", va="center", fontsize=9.0, style="italic", color="#555555")

    fig.tight_layout()
    return fig


def main() -> None:
    fig = build_figure()
    save_figure(fig, OUT_PATH)
    print(f"OK fig3_6 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
