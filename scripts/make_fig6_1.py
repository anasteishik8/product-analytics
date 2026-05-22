"""make_fig6_1.py — fig6_1: per-day D² для M1 vs M3 (двух-панельная, F1 и F2).

Главный методологический вывод главы 6: M1 (только in-app) даёт DEVELOP для обоих
продуктов, а M3 (полная модель со Store/Market) выявляет F2 как DISCONTINUE.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure


def main() -> Path:
    apply_vkr_style()

    scores = pd.read_csv(ROOT / "results" / "viability_scores.csv")
    summary = pd.read_csv(ROOT / "results" / "viability_summary.csv")

    products = [
        ("com.labpixies.flood", "Flood-It!"),
        ("com.google.flood2", "Flood-It! 2"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)

    legend_handles = None
    legend_labels = None

    for ax, (app_id, title) in zip(axes, products):
        sub = scores[scores["app_id"] == app_id].copy()
        m1 = sub[sub["model"] == "M1"].sort_values("day")
        m2 = sub[sub["model"] == "M2"].sort_values("day")
        m3 = sub[sub["model"] == "M3"].sort_values("day")

        # M2 как тонкий серый контекст
        ax.plot(m2["day"], m2["D2"], color=VKR_PALETTE[4], linewidth=0.9,
                alpha=0.55, label="M2 (in-app + ops)")
        # M1 — синий
        ax.plot(m1["day"], m1["D2"], color=VKR_PALETTE[0], linewidth=1.5,
                label="M1 (только in-app)")
        # M3 — красный, акцент
        ax.plot(m3["day"], m3["D2"], color=VKR_PALETTE[1], linewidth=1.7,
                label="M3 (полная модель)")

        # T95 для M3 — пунктирная горизонтальная линия (порог DISCONTINUE для M3)
        t95_m3 = summary[(summary["app_id"] == app_id) &
                         (summary["model"] == "M3")]["T95"].iloc[0]
        ax.axhline(t95_m3, color=VKR_PALETTE[1], linestyle="--",
                   linewidth=0.9, alpha=0.7,
                   label="T95 (M3)")

        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("День")
        ax.set_ylabel(r"$D^2$ (Mahalanobis), логарифмическая шкала")

        # Расширим верхний y-лимит, чтобы линии не упирались в заголовок panel'а
        y_top = ax.get_ylim()[1]
        ax.set_ylim(top=y_top * 1.8)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Общая легенда — снизу под графиками, не наезжает на данные
    fig.legend(legend_handles, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=False, fontsize=9)

    fig.suptitle(r"Per-day $D^2$ для M1/M2/M3: сравнение блоков признаков",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.18)

    out = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig6_1.pdf"
    save_figure(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = main()
    size_kb = out.stat().st_size / 1024
    print(f"OK fig6_1 saved: {out} ({size_kb:.1f} KB)")
