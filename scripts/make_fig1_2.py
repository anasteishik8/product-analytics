"""
make_fig1_2.py — Границы разрабатываемого прототипа и условия применения.

Защитная диаграмма для главы 1: вместо UML use-case с несколькими ролями
показывает границы прототипа и контракт входных данных. Заранее отвечает
на вопрос «почему только два приложения?» — потому что это прототип с
заданным контрактом данных, валидированный на кейсе.

Структура (сверху вниз):
  Исторические данные продукта по заданной схеме признаков
                       ↓
  Прототип системы прогнозирования востребованности
                       ↓
  Выходы: прогноз метрик, доверительные интервалы, сценарный анализ,
         рекомендации, вердикт
                       ↓
  Пользователь системы — аналитик или другой специалист,
  интерпретирующий продуктовые метрики

Условия переноса прототипа на новый продукт описаны в тексте главы,
а не на рисунке: визуально они конкурировали с основной цепочкой.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Цвета из VKR_PALETTE
PRIMARY_BLUE = VKR_PALETTE[0]
RED = VKR_PALETTE[1]
GREEN = VKR_PALETTE[2]
OCHRE = VKR_PALETTE[3]
GRAY = VKR_PALETTE[4]

BLUE_BG = "#e3edf5"
GREEN_BG = "#e6f1ec"
OCHRE_BG = "#fbf2e3"
GRAY_BG = "#f0f0f0"


def _box(ax, x, y, w, h, label, fill=BLUE_BG, edge=PRIMARY_BLUE,
         lw=1.5, fontsize=11, weight="normal"):
    """Прямоугольник со скруглёнными углами + текст по центру."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
        color="black",
    )


def _arrow(ax, x_from, y_from, x_to, y_to, color=GRAY, lw=1.8):
    ax.add_patch(FancyArrowPatch(
        (x_from, y_from), (x_to, y_to),
        arrowstyle="->,head_length=8,head_width=6",
        linewidth=lw, color=color, zorder=2,
    ))


def main() -> Path:
    apply_vkr_style()
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    cx = 5.0
    box_w = 7.0

    # 1) Входные данные
    _box(ax, cx - box_w / 2, 8.7, box_w, 1.2,
         "Исторические данные продукта\nпо заданной схеме признаков",
         fill=GRAY_BG, edge=GRAY, fontsize=11)

    # 2) Прототип системы (главный блок, акцент)
    _box(ax, cx - box_w / 2, 6.4, box_w, 1.3,
         "Прототип системы прогнозирования\nвостребованности",
         fill=GREEN_BG, edge=GREEN, lw=2.5, fontsize=12.5, weight="bold")

    # 3) Выходы
    outputs_lines = [
        "Прогноз метрик   •   Доверительные интервалы",
        "Сценарный анализ   •   Рекомендации   •   Вердикт",
    ]
    _box(ax, cx - box_w / 2, 3.8, box_w, 1.6,
         "\n".join(outputs_lines),
         fill=BLUE_BG, edge=PRIMARY_BLUE, fontsize=11)

    # 4) Пользователь
    _box(ax, cx - box_w / 2, 1.2, box_w, 1.4,
         "Пользователь системы — аналитик или\n"
         "другой специалист, интерпретирующий\n"
         "продуктовые метрики",
         fill=OCHRE_BG, edge=OCHRE, fontsize=11)

    # Стрелки вертикально вниз
    _arrow(ax, cx, 8.65, cx, 7.75)  # 1→2
    _arrow(ax, cx, 6.35, cx, 5.45)  # 2→3
    _arrow(ax, cx, 3.75, cx, 2.65)  # 3→4

    plt.tight_layout()
    out = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig1_2.pdf"
    save_figure(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = main()
    print(f"OK fig1_2 saved: {out} ({out.stat().st_size} bytes)")
