"""
make_fig3_7.py — логика вердикта через расстояние Махаланобиса и долю плохих дней.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure  # noqa: E402


OUT_PATH = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig3_7.pdf"


def _box(ax, x, y, w, h, text, face, edge="#333333", lw=1.0,
         fontsize=10.5, bold=False):
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
            fontsize=fontsize, weight=weight)


def _diamond(ax, cx, cy, w, h, text, face="#ffffff", edge="#333"):
    """Возвращает координаты вершин (top, right, bottom, left)."""
    top = (cx, cy + h / 2)
    right = (cx + w / 2, cy)
    bottom = (cx, cy - h / 2)
    left = (cx - w / 2, cy)
    xs = [top[0], right[0], bottom[0], left[0], top[0]]
    ys = [top[1], right[1], bottom[1], left[1], top[1]]
    ax.fill(xs, ys, facecolor=face, edgecolor=edge, linewidth=1.0)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.0)
    return top, right, bottom, left


def _arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0, 0)):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.1,
        color="#444444",
    ))
    if label:
        ax.text((x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1],
                label, ha="center", va="center",
                fontsize=9.5, style="italic", color="#444444",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


def build_figure() -> plt.Figure:
    apply_vkr_style()
    fig, ax = plt.subplots(figsize=(12.5, 11.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.set_axis_off()

    C_DEV = "#cde9d5"
    C_MON = "#fbf2e3"
    C_DIS = "#f4d9d4"
    E_DEV = VKR_PALETTE[2]
    E_MON = VKR_PALETTE[3]
    E_DIS = VKR_PALETTE[1]

    # ─── 1. Вход
    _box(ax, 4.5, 12.5, 5.0, 1.0,
         "Прогнозы и признаки дня t",
         face="#e3edf5", edge="#2c5d8a", bold=True)

    # ─── 2. D² блок
    _box(ax, 4.5, 10.7, 5.0, 1.1,
         r"Расстояние Махаланобиса $D^2$" + "\nотносительно опорной нормы",
         face="#e3edf5", edge="#2c5d8a")
    _arrow(ax, 7.0, 12.5, 7.0, 11.83)

    # ─── 3. Diamond 1: D² < T95?
    d1_top, d1_right, d1_bot, d1_left = _diamond(
        ax, 4.0, 8.8, 2.8, 1.3, r"$D^2 < T_{95}$ ?")
    # стрелка из D² блока вниз — в верх diamond1
    _arrow(ax, 7.0, 10.7, 4.0, 9.45)

    # ─── 4. DEVELOP-день (под/левее diamond1)
    _box(ax, 0.6, 5.6, 3.0, 1.0, "DEVELOP-день",
         face=C_DEV, edge=E_DEV, bold=True, fontsize=11)
    # diamond1 "да" → DEVELOP
    _arrow(ax, d1_left[0], d1_left[1] - 0.05, 2.1, 6.65,
           label="да", label_offset=(-0.2, -0.05))

    # ─── 5. Diamond 2: D² < T99? (справа от diamond1)
    d2_top, d2_right, d2_bot, d2_left = _diamond(
        ax, 10.0, 8.8, 2.8, 1.3, r"$D^2 < T_{99}$ ?")
    # diamond1 "нет" → diamond2 (по горизонтали)
    _arrow(ax, d1_right[0], d1_right[1], d2_left[0] - 0.02, d2_left[1],
           label="нет", label_offset=(0, 0.32))

    # ─── 6. MONITOR-день (под/правее diamond2)
    _box(ax, 10.4, 5.6, 3.0, 1.0, "MONITOR-день",
         face=C_MON, edge=E_MON, bold=True, fontsize=11)
    # diamond2 "да" → MONITOR (вниз-вправо)
    _arrow(ax, d2_right[0] - 0.05, d2_right[1] - 0.05, 11.9, 6.65,
           label="да", label_offset=(0.25, -0.05))

    # ─── 7. DISCONTINUE-день (под diamond2 вниз-влево, по центру схемы)
    _box(ax, 5.5, 5.6, 3.0, 1.0, "DISCONTINUE-день",
         face=C_DIS, edge=E_DIS, bold=True, fontsize=11)
    # diamond2 "нет" → DISCONTINUE
    _arrow(ax, d2_bot[0], d2_bot[1], 7.0, 6.65,
           label="нет", label_offset=(0.45, 0.05))

    # ─── 8. bad_share — объединяющий блок
    _box(ax, 3.5, 3.3, 7.0, 1.0,
         "bad_share = доля плохих дней (MONITOR + DISCONTINUE)",
         face="#eeeeee", edge="#555555", fontsize=10.5)
    # три стрелки от дневных вердиктов → bad_share
    _arrow(ax, 2.1, 5.6, 5.0, 4.35)
    _arrow(ax, 7.0, 5.6, 7.0, 4.35)
    _arrow(ax, 11.9, 5.6, 9.0, 4.35)

    # ─── 9. Итоговый вердикт
    _box(ax, 3.5, 1.3, 7.0, 1.3,
         "Итоговый вердикт продукта:\nDEVELOP / MONITOR / DISCONTINUE",
         face="#e3edf5", edge="#2c5d8a", lw=1.6, bold=True, fontsize=11.5)
    _arrow(ax, 7.0, 3.3, 7.0, 2.65)

    # ─── Легенда цветов (внизу)
    legend_y = 0.1
    items = [("DEVELOP", C_DEV, E_DEV),
             ("MONITOR", C_MON, E_MON),
             ("DISCONTINUE", C_DIS, E_DIS)]
    lx = 1.5
    for name, face, edge in items:
        _box(ax, lx, legend_y, 1.1, 0.45, "", face=face, edge=edge)
        ax.text(lx + 1.25, legend_y + 0.22, name,
                ha="left", va="center", fontsize=10)
        lx += 4.0

    fig.tight_layout()
    return fig


def main() -> None:
    fig = build_figure()
    save_figure(fig, OUT_PATH)
    print(f"OK fig3_7 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
