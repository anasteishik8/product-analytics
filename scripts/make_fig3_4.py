"""
make_fig3_4.py — гистограмма bootstrap-выборок с границами 95% доверительного интервала.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure  # noqa: E402


OUT_PATH = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig3_4.pdf"
SEED = 42


def build_figure() -> plt.Figure:
    apply_vkr_style()
    rng = np.random.default_rng(SEED)
    samples = rng.normal(loc=0.0, scale=1.0, size=2000)

    median = float(np.median(samples))
    q_lo = float(np.quantile(samples, 0.025))
    q_hi = float(np.quantile(samples, 0.975))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hist(samples, bins=45, color=VKR_PALETTE[0],
            alpha=0.75, edgecolor="white", linewidth=0.7)
    ax.axvline(median, color=VKR_PALETTE[0],
               linewidth=2.0, linestyle="-",
               label=f"медиана = {median:.2f}")
    ax.axvline(q_lo, color=VKR_PALETTE[1],
               linewidth=1.6, linestyle="--",
               label=f"2.5% квантиль = {q_lo:.2f}")
    ax.axvline(q_hi, color=VKR_PALETTE[1],
               linewidth=1.6, linestyle="--",
               label=f"97.5% квантиль = {q_hi:.2f}")

    ax.set_xlabel("значение оценки")
    ax.set_ylabel("частота")
    ax.set_title("Распределение бутстреп-оценок")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Подпись внутри
    ax.text(0.02, 0.96,
            r"CI$_{95}$ = $[q_{2.5},\, q_{97.5}]$",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#bbbbbb", boxstyle="round,pad=0.3"))

    fig.tight_layout()
    return fig


def main() -> None:
    fig = build_figure()
    save_figure(fig, OUT_PATH)
    print(f"OK fig3_4 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
