"""
make_fig3_2.py — алгоритмическая логика семейств моделей-кандидатов.

3-панельный рисунок:
    панель 1: Ridge — глобальная линейная зависимость
    панель 2: NW / LocalLinear / kNN — локальные оценщики
    панель 3: RandomForest / XGBoost — ансамбли
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure  # noqa: E402


OUT_PATH = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig3_2.pdf"
SEED = 42


def _toy_data():
    rng = np.random.default_rng(SEED)
    x = np.linspace(0.5, 9.5, 30)
    y = 0.5 * x + rng.normal(0, 0.6, size=x.size)
    return x, y


def _panel_ridge(ax, x, y):
    ax.scatter(x, y, color=VKR_PALETTE[4], s=28, alpha=0.85, zorder=2)
    # Ridge fit (closed-form linear)
    beta1, beta0 = np.polyfit(x, y, 1)
    xs = np.linspace(0, 10, 50)
    ax.plot(xs, beta0 + beta1 * xs, color=VKR_PALETTE[0],
            linewidth=2.0, zorder=3, label="Ridge")
    ax.text(0.5, 5.3, r"$\min\ \|y - X\beta\|^2 + \lambda\|\beta\|^2$",
            fontsize=11, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="#bbbbbb", boxstyle="round,pad=0.3"))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 6)
    ax.set_title("Ridge")
    ax.set_xlabel("признак x")
    ax.set_ylabel("целевая y")
    ax.text(0.5, -0.22, "Глобальная линейная зависимость",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, style="italic")


def _panel_local(ax, x, y):
    x0 = 5.0
    bw = 1.5
    # окно
    ax.axvspan(x0 - bw, x0 + bw, color=VKR_PALETTE[0], alpha=0.10, zorder=1)
    in_win = np.abs(x - x0) <= bw
    # точки вне окна — бледные, внутри — насыщенные
    ax.scatter(x[~in_win], y[~in_win], color=VKR_PALETTE[4],
               s=22, alpha=0.45, zorder=2)
    ax.scatter(x[in_win], y[in_win], color=VKR_PALETTE[0],
               s=32, alpha=0.95, zorder=3)
    # локальная прямая
    if in_win.sum() >= 2:
        b1, b0 = np.polyfit(x[in_win], y[in_win], 1)
        xs = np.linspace(x0 - bw, x0 + bw, 25)
        ax.plot(xs, b0 + b1 * xs, color=VKR_PALETTE[1],
                linewidth=2.0, linestyle="--",
                label="локальная прямая")
    # точка прогноза
    y_hat = np.mean(y[in_win])
    ax.scatter([x0], [y_hat], color=VKR_PALETTE[0],
               s=140, marker="o", edgecolor="white", linewidth=1.6,
               zorder=5, label=r"прогноз в $x_0$")
    ax.annotate(r"$x_0$", (x0, -0.7), ha="center", fontsize=11)

    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 6)
    ax.set_title("Nadaraya–Watson, LocalLinear, kNN")
    ax.set_xlabel("признак x")
    ax.text(0.5, -0.22, "Прогноз строится по похожим наблюдениям",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, style="italic")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)


def _panel_ensemble(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    # Вход
    box_in = FancyBboxPatch((0.4, 4.3), 1.4, 1.4,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            edgecolor="#333", facecolor="#e3edf5", linewidth=1.0)
    ax.add_patch(box_in)
    ax.text(1.1, 5.0, "x", ha="center", va="center", fontsize=14, weight="bold")

    # 4 дерева — две пары: верх / низ
    tree_centers = [(4.5, 8.0), (4.5, 6.4), (4.5, 4.0), (4.5, 2.4)]
    for i, (tx, ty) in enumerate(tree_centers):
        # маленький треугольник как «дерево»
        ax.plot([tx - 0.5, tx + 0.5, tx, tx - 0.5],
                [ty - 0.4, ty - 0.4, ty + 0.5, ty - 0.4],
                color="#3a8a64", linewidth=1.4)
        ax.plot([tx, tx], [ty - 0.4, ty - 0.85], color="#3a8a64", linewidth=1.2)
        ax.text(tx + 0.75, ty, f"дерево {i + 1}", ha="left", va="center", fontsize=9)
        # стрелка вход -> дерево
        ax.add_patch(FancyArrowPatch(
            (1.9, 5.0), (tx - 0.55, ty),
            arrowstyle="-|>", mutation_scale=10, linewidth=0.8, color="#888888"
        ))

    # Агрегатор
    box_agg = FancyBboxPatch((6.6, 4.3), 1.6, 1.4,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             edgecolor="#333", facecolor="#fbf2e3", linewidth=1.0)
    ax.add_patch(box_agg)
    ax.text(7.4, 5.0, "агрегатор", ha="center", va="center", fontsize=9)

    for (tx, ty) in tree_centers:
        ax.add_patch(FancyArrowPatch(
            (tx + 0.55, ty), (6.65, 5.0),
            arrowstyle="-|>", mutation_scale=10, linewidth=0.8, color="#888888"
        ))

    # Выход
    box_out = FancyBboxPatch((8.6, 4.3), 1.2, 1.4,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             edgecolor="#333", facecolor="#e3edf5", linewidth=1.0)
    ax.add_patch(box_out)
    ax.text(9.2, 5.0, r"$\hat y$", ha="center", va="center", fontsize=14, weight="bold")
    ax.add_patch(FancyArrowPatch(
        (8.25, 5.0), (8.55, 5.0),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#444444"
    ))

    # пояснение
    ax.text(5.0, 0.9,
            "RandomForest: усреднение деревьев\nXGBoost: сумма последовательных поправок",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="#bbbbbb", boxstyle="round,pad=0.3"))

    ax.set_title("RandomForest, XGBoost")


def build_figure() -> plt.Figure:
    apply_vkr_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.4))
    x, y = _toy_data()
    _panel_ridge(axes[0], x, y)
    _panel_local(axes[1], x, y)
    _panel_ensemble(axes[2])

    # Подпись-полоса под всем рисунком
    fig.text(0.5, 0.02,
             "Все семейства — кандидаты. Рабочая модель выбирается per (продукт, метрика) "
             "по recursive forecast validation.",
             ha="center", va="bottom", fontsize=10, style="italic",
             bbox=dict(facecolor="#f4f4f4", edgecolor="#cccccc", boxstyle="round,pad=0.4"))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main() -> None:
    fig = build_figure()
    save_figure(fig, OUT_PATH)
    print(f"OK fig3_2 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
