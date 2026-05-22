"""
make_fig1_1.py — Контур принятия продуктового решения (swimlane).

Swimlane-диаграмма: слева источники данных (внешние по отношению к аналитику),
справа — 5 шагов работы аналитика с decision diamond и двумя исходами:
вердикт или итерация (уточнить данные / расширить наблюдения / пересобрать сценарии).

Стиль главы 1 (без «артефакт», без «what-if», без англицизмов в подписях):
  «сценарный анализ» вместо «what-if-анализ»
  «развивать / мониторить / закрывать» вместо «DEVELOP / MONITOR / DISCONTINUE»
  «результат шага» вместо «артефакт»
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PRIMARY_BLUE = VKR_PALETTE[0]
RED = VKR_PALETTE[1]
GREEN = VKR_PALETTE[2]
OCHRE = VKR_PALETTE[3]
GRAY = VKR_PALETTE[4]

BLUE_BG = "#e3edf5"
GREEN_BG = "#e6f1ec"
OCHRE_BG = "#fbf2e3"
GRAY_BG = "#f0f0f0"
NOTE_BG = "#fff8d6"


def _box(ax, x, y, w, h, label, fill=BLUE_BG, edge=PRIMARY_BLUE,
         lw=1.5, fontsize=10, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, fontweight=weight,
            color="black")


def _note(ax, x, y, w, h, label, fontsize=9):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.0, edgecolor=OCHRE, facecolor=NOTE_BG,
    )
    ax.add_patch(patch)
    ax.text(x + 0.12, y + h / 2, label,
            ha="left", va="center", fontsize=fontsize, color="black")


def _arrow(ax, x_from, y_from, x_to, y_to, color=GRAY, lw=1.6,
           connectionstyle=None):
    kwargs = dict(arrowstyle="->,head_length=8,head_width=6",
                  linewidth=lw, color=color, zorder=3)
    if connectionstyle:
        kwargs["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch((x_from, y_from), (x_to, y_to), **kwargs))


def _diamond(ax, cx, cy, w, h, label, fill=OCHRE_BG, edge=OCHRE,
             fontsize=10, weight="bold"):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill,
                          edgecolor=edge, linewidth=1.8))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="black")


def main() -> Path:
    apply_vkr_style()
    fig, ax = plt.subplots(figsize=(13.5, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # --- Заголовки полос ---
    ax.text(2.25, 13.4, "Источники данных", ha="center", va="center",
            fontsize=12, fontweight="bold", color=GRAY)
    ax.text(8, 13.4, "Аналитик", ha="center", va="center",
            fontsize=12, fontweight="bold", color=GRAY)
    # Разделитель полос
    ax.plot([4.5, 4.5], [0.5, 13], color=GRAY, lw=0.8, linestyle="-",
            alpha=0.5, zorder=1)
    # Подложки полос (тонкая полупрозрачная заливка)
    ax.add_patch(FancyBboxPatch(
        (0.3, 0.5), 4.2, 12.5,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=0, facecolor="#fafafa", zorder=0))
    ax.add_patch(FancyBboxPatch(
        (4.7, 0.5), 11.0, 12.5,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=0, facecolor="#ffffff", zorder=0))

    # --- Левая полоса: источники данных (4 блока) ---
    src_x = 0.6
    src_w = 3.6
    src_labels = [
        "Firebase BigQuery",
        "Google Play / store metadata",
        "Kaggle market snapshot",
        "Внешние сигналы:\nGoogle Trends, Wikipedia",
    ]
    src_y = [11.7, 10.4, 9.1, 7.4]
    src_heights = [0.85, 0.85, 0.85, 1.0]
    for y, h_, lbl in zip(src_y, src_heights, src_labels):
        _box(ax, src_x, y, src_w, h_, lbl,
             fill=GRAY_BG, edge=GRAY, fontsize=9.5)
    # Стрелки между источниками
    for i in range(3):
        y_from = src_y[i]
        y_to = src_y[i + 1] + src_heights[i + 1]
        _arrow(ax, src_x + src_w / 2, y_from,
               src_x + src_w / 2, y_to, lw=1.2)

    # --- Правая полоса: 4 шага + decision + 2 исхода + terminal ---
    step_x = 5.5
    step_w = 4.6
    step_h = 1.0
    step_titles = [
        ("1. Диагностика",
         "оценка качества данных,\nструктура продуктовых метрик"),
        ("2. Прогноз",
         "оценка траекторий ключевых метрик\nна безопасном горизонте"),
        ("3. Сценарии",
         "сценарный анализ управляемых\nпродуктовых рычагов"),
        ("4. Рекомендации",
         "ранжирование действий по\nожидаемому эффекту на метрики"),
    ]
    step_y = [11.4, 9.7, 8.0, 6.3]
    for (title, sub), y in zip(step_titles, step_y):
        patch = FancyBboxPatch(
            (step_x, y), step_w, step_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.8, edgecolor=PRIMARY_BLUE, facecolor=BLUE_BG,
        )
        ax.add_patch(patch)
        ax.text(step_x + step_w / 2, y + step_h * 0.72, title,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="black")
        ax.text(step_x + step_w / 2, y + step_h * 0.28, sub,
                ha="center", va="center",
                fontsize=9.5, color="black")

    # Стрелки между шагами 1→2, 2→3, 3→4
    for i in range(3):
        y_from = step_y[i]
        y_to = step_y[i + 1] + step_h
        _arrow(ax, step_x + step_w / 2, y_from,
               step_x + step_w / 2, y_to, lw=1.5)

    # Аннотации справа от каждого шага (стикеры)
    note_x = 10.4
    note_w = 5.2
    notes_data = [
        ("Итоговый набор данных: 222 × 60\n(подмножество признаков подбирается\nпод конкретную модель)", 1.0),
        ("Целевые метрики: stickiness,\nretention_d7, DAU", 0.85),
        ("Управляемые признаки:\n• crash_rate\n• onboarding_completion_rate\n• median_session_duration", 1.25),
        ("Результат шага: список приоритетных\nпродуктовых изменений с оценкой эффекта", 1.0),
    ]
    for (note_text, n_h), y_step in zip(notes_data, step_y):
        y_note = y_step + (step_h - n_h) / 2
        _note(ax, note_x, y_note, note_w, n_h, note_text, fontsize=9)
        # Связь шаг — стикер
        ax.plot([step_x + step_w, note_x],
                 [y_step + step_h / 2, y_step + step_h / 2],
                 color=OCHRE, linestyle=":", linewidth=0.9, zorder=2)

    # --- Decision diamond ---
    dx, dy = step_x + step_w / 2, 4.6
    diamond_w, diamond_h = 3.6, 1.4
    _diamond(ax, dx, dy, diamond_w, diamond_h,
             "Достаточна ли надёжность\nпрогноза и сценария?",
             fontsize=10.5, weight="bold")
    # Стрелка от шага 4 в верхнюю вершину ромба
    _arrow(ax, dx, step_y[3], dx, dy + diamond_h / 2, lw=1.5)

    # --- Два исхода: вердикт (слева снизу) и итерация (справа снизу) ---
    verdict_x = 5.5
    verdict_y = 2.2
    verdict_w = 4.0
    verdict_h = 1.2
    _box(ax, verdict_x, verdict_y, verdict_w, verdict_h,
         "5. Вердикт\nразвивать / мониторить / закрывать",
         fill=GREEN_BG, edge=GREEN, lw=2.2, fontsize=10.5, weight="bold")

    iter_x = 11.0
    iter_y = 2.2
    iter_w = 4.5
    iter_h = 1.2
    _box(ax, iter_x, iter_y, iter_w, iter_h,
         "Уточнить данные,\nрасширить наблюдения,\nпересобрать сценарии",
         fill=OCHRE_BG, edge=OCHRE, lw=1.5, fontsize=10)

    # Стрелки от ромба к двум исходам с подписями «да» / «нет»
    _arrow(ax, dx - diamond_w / 2, dy,
           verdict_x + verdict_w, verdict_y + verdict_h / 2,
           connectionstyle="arc3,rad=-0.15", lw=1.6)
    ax.text(dx - diamond_w / 2 - 0.8, dy - 0.7, "да",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=GREEN)

    _arrow(ax, dx + diamond_w / 2, dy,
           iter_x, iter_y + iter_h / 2,
           connectionstyle="arc3,rad=0.15", lw=1.6)
    ax.text(dx + diamond_w / 2 + 0.8, dy - 0.7, "нет",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=OCHRE)

    # --- Terminal node (●) ---
    term_x = verdict_x + verdict_w / 2
    term_y = 1.05
    ax.plot([term_x], [term_y], marker="o", markersize=18,
            color="black", zorder=5)
    ax.plot([term_x], [term_y], marker="o", markersize=8,
            color="white", zorder=6)
    _arrow(ax, term_x, verdict_y, term_x, term_y + 0.18, lw=1.5)

    # --- Стрелка от источников данных к шагу 1 ---
    src_join_y = src_y[-1] - 0.3
    ax.plot([src_x + src_w / 2, src_x + src_w / 2],
             [src_y[-1], src_join_y], color=GRAY, lw=1.5, zorder=2)
    _arrow(ax, src_x + src_w / 2, src_join_y,
           step_x, step_y[0] + step_h / 2,
           lw=1.8, color=PRIMARY_BLUE)

    plt.tight_layout()
    out = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig1_1.pdf"
    save_figure(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = main()
    print(f"OK fig1_1 saved: {out} ({out.stat().st_size} bytes)")
