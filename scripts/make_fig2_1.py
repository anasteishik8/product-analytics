"""make_fig2_1.py — fig2_1: схема формирования датасета из 4 источников.

Custom-диаграмма: 4 source-блока (Firebase, Google Play Scraper, Kaggle,
Google Trends + Wikipedia) → центральный блок «floodit_final.parquet 222×60».
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure


def _add_block(ax, x, y, w, h, title, lines, fc, ec, lw=1.2, title_fs=10,
               body_fs=8.5, text_color="#111111"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=0.92,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.13, title,
            ha="center", va="top", fontsize=title_fs, fontweight="bold",
            color=text_color)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.32 - 0.18 * i, line,
                ha="center", va="top", fontsize=body_fs, color=text_color)


def _add_arrow(ax, x1, y1, x2, y2, label, label_offset=(0, 0.06)):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#444444",
                        lw=1.2, mutation_scale=14),
    )
    mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
    ax.text(mx, my, label, ha="center", va="center", fontsize=8,
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.85))


def main() -> Path:
    apply_vkr_style()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Левая колонка — 4 источника (вертикально), по высоте 1.0, разнесены
    src_w, src_h = 2.8, 1.05
    src_x = 0.15
    src_y = [4.55, 3.20, 1.85, 0.50]  # сверху вниз

    # Цвета: in-app=синий (F1), store=охра, market=зелёный, external=серый.
    sources = [
        ("Firebase BigQuery",
         ["In-app телеметрия", "dau, stickiness, retention"],
         VKR_PALETTE[0]),
        ("Google Play Scraper",
         ["Store-метрики", "rating, reviews_count"],
         VKR_PALETTE[3]),
        ("Kaggle (Google Play)",
         ["Конкуренция в категории", "competitor_count, rank"],
         VKR_PALETTE[2]),
        ("Google Trends + Wikipedia",
         ["Внешний интерес", "media_coverage, downloads_index"],
         VKR_PALETTE[4]),
    ]
    # Сумма по 4 источникам = 60 колонок (24 in-app + 4 служебных + 16 + 9 + 7).
    # Служебные (date, app_id, category, weekday) приходят вместе с Firebase-экспортом.
    feature_labels = [
        "24 in-app + 4 служ.",
        "16 признаков",
        "9 признаков",
        "7 признаков",
    ]

    for (title, body, color), y, fl in zip(sources, src_y, feature_labels):
        _add_block(
            ax, src_x, y, src_w, src_h,
            title=title, lines=body,
            fc=_lighten(color, 0.78), ec=color, lw=1.3,
        )

    # Центральный блок — итоговый датасет
    tgt_w, tgt_h = 3.4, 2.0
    tgt_x = 6.25
    tgt_y = 2.0
    _add_block(
        ax, tgt_x, tgt_y, tgt_w, tgt_h,
        title="floodit_final.parquet",
        lines=[
            "222 строки × 60 колонок",
            "2 продукта × 111 дней",
            "0% синтетики",
            "Период: 12.06–30.09.2018",
        ],
        fc="#f5f5f5", ec="#222222", lw=2.0,
        title_fs=11.5, body_fs=9.5,
    )

    # Стрелки от источников к центральному блоку
    arrow_x_to = tgt_x  # левая граница центрального блока
    for y, fl, (_, _, color) in zip(src_y, feature_labels, sources):
        arrow_y_from = y + src_h / 2
        arrow_y_to = tgt_y + tgt_h / 2 + (arrow_y_from - 3.0) * 0.18
        _add_arrow(
            ax,
            src_x + src_w + 0.05, arrow_y_from,
            arrow_x_to - 0.05, arrow_y_to,
            label=fl,
            label_offset=(0.0, 0.08),
        )

    fig.suptitle(
        "Схема формирования итогового датасета: 4 независимых источника",
        fontsize=12, y=0.98,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    out = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig2_1.pdf"
    save_figure(fig, out)
    plt.close(fig)
    return out


def _lighten(hex_color: str, t: float) -> str:
    """Линейно «осветляет» hex-цвет к белому. t∈[0,1], t=0 — оригинал, t=1 — белый."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(round(r + (255 - r) * t))
    g = int(round(g + (255 - g) * t))
    b = int(round(b + (255 - b) * t))
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    out = main()
    size_kb = out.stat().st_size / 1024
    print(f"OK fig2_1 saved: {out} ({size_kb:.1f} KB)")
