"""make_fig2_2.py — fig2_2: две траектории продукта (DAU, stickiness, retention_d7).

3 panels (1×3). F1 — синий, F2 — красный. Общая легенда снизу.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure


F1_ID = "com.labpixies.flood"
F2_ID = "com.google.flood2"
F1_LABEL = "Flood-It! (F1)"
F2_LABEL = "Flood-It! 2 (F2)"


def main() -> Path:
    apply_vkr_style()

    df = pd.read_parquet(ROOT / "data" / "processed" / "floodit_final.parquet")
    df["date"] = pd.to_datetime(df["date"])

    f1 = df[df["app_id"] == F1_ID].sort_values("date")
    f2 = df[df["app_id"] == F2_ID].sort_values("date")

    panels = [
        ("dau", "DAU", "Daily Active Users"),
        ("stickiness", "stickiness", "DAU / MAU"),
        ("retention_d7", "retention_d7", "Доля вернувшихся на 7-й день"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    legend_handles, legend_labels = None, None

    for ax, (col, title, ylabel) in zip(axes, panels):
        ax.plot(f1["date"], f1[col], color=VKR_PALETTE[0], lw=1.6,
                label=F1_LABEL)
        ax.plot(f2["date"], f2[col], color=VKR_PALETTE[1], lw=1.6,
                label=F2_LABEL)
        ax.set_title(title)
        ax.set_xlabel("Дата")
        ax.set_ylabel(ylabel)

        # Если масштаб DAU между F1/F2 различается на порядок — log-scale
        if col == "dau":
            ax.set_yscale("log")
            ax.set_ylabel(ylabel + " (лог. шкала)")

        # Поворот x-ticks
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.04),
        ncol=2, frameon=False, fontsize=10,
    )
    fig.suptitle(
        "Две траектории продукта: успех (F1) vs провал (F2)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.86, bottom=0.22)

    out = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig2_2.pdf"
    save_figure(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = main()
    size_kb = out.stat().st_size / 1024
    print(f"OK fig2_2 saved: {out} ({size_kb:.1f} KB)")
