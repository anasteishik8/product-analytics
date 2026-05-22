"""Экспорт рисунка 5.1 для ВКР.

Рисунок строится из финального артефакта recursive validation:
results/forecast_validation_recursive_summary.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "forecast_validation_recursive_summary.csv"
OUT_DIR = ROOT / "vkr" / "v2" / "artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "fig5_1_model_quality_aggregate.png"
OUT_PDF = OUT_DIR / "fig5_1_model_quality_aggregate.pdf"

APP_LABELS = {
    "f1": "Flood-It!",
    "f2": "Flood-It! 2",
}

TARGET_LABELS = {
    "stickiness": "stickiness",
    "dau": "DAU",
    "retention_d7": "retention_d7",
}

TARGET_ORDER = ["stickiness", "dau", "retention_d7"]
APP_ORDER = ["f1", "f2"]

MODEL_ORDER = [
    "Ridge",
    "NadarayaWatson",
    "LocalLinear",
    "RandomForest",
    "XGBoost",
]

MODEL_LABELS = {
    "Ridge": "Ridge",
    "RandomForest": "RF",
    "XGBoost": "XGBoost",
    "NadarayaWatson": "NW",
    "LocalLinear": "LL",
}


def build_plot() -> None:
    df = pd.read_csv(SUMMARY_PATH)
    df = df[df["app"].isin(APP_ORDER) & df["target"].isin(TARGET_ORDER)].copy()

    df["pair"] = (
        df["app"].map(APP_LABELS)
        + " / "
        + df["target"].map(TARGET_LABELS)
    )

    pair_order = [f"{app.upper()} / {TARGET_LABELS[target]}" for app in APP_ORDER for target in TARGET_ORDER]
    pair_label_map = {
        f"{APP_LABELS[app]} / {TARGET_LABELS[target]}": f"{app.upper()} / {TARGET_LABELS[target]}"
        for app in APP_ORDER
        for target in TARGET_ORDER
    }
    df["pair_short"] = df["pair"].map(pair_label_map)

    best_rmse = df.groupby("pair")["RMSE"].transform("min")
    df["relative_rmse"] = df["RMSE"] / best_rmse
    df["is_winner"] = np.isclose(df["relative_rmse"], 1.0)

    matrix = (
        df.pivot(index="pair_short", columns="model", values="relative_rmse")
        .reindex(index=pair_order, columns=MODEL_ORDER)
    )
    beats = (
        df.pivot(index="pair_short", columns="model", values="beats_naive")
        .reindex(index=pair_order, columns=MODEL_ORDER)
        .fillna(False)
    )
    winners = (
        df.pivot(index="pair_short", columns="model", values="is_winner")
        .reindex(index=pair_order, columns=MODEL_ORDER)
        .fillna(False)
    )

    values = matrix.to_numpy(dtype=float)
    clipped = np.clip(values, 1.0, 3.0)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_bad("#F2F2F2")

    image = ax.imshow(np.ma.masked_invalid(clipped), cmap=cmap, vmin=1.0, vmax=3.0)

    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticks(np.arange(len(pair_order)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_yticklabels(pair_order)
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.tick_params(axis="both", length=0)

    ax.set_xticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pair_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx, pair in enumerate(pair_order):
        for col_idx, model in enumerate(MODEL_ORDER):
            value = matrix.loc[pair, model]
            if pd.isna(value):
                ax.text(col_idx, row_idx, "—", ha="center", va="center", color="#666666")
                continue

            mark = "*" if bool(beats.loc[pair, model]) else ""
            text_color = "white" if value >= 2.25 else "#111111"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}{mark}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold" if bool(winners.loc[pair, model]) else "normal",
            )

            if bool(winners.loc[pair, model]):
                ax.add_patch(
                    patches.Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=2.2,
                    )
                )

    ax.set_title("Относительный RMSE по парам продукт–метрика")
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Относительный RMSE\n(1.00 = лучшая модель)")
    cbar.set_ticks([1.0, 1.5, 2.0, 2.5, 3.0])
    cbar.set_ticklabels(["1.0", "1.5", "2.0", "2.5", "≥3.0"])

    fig.text(
        0.01,
        0.02,
        "Примечание: F1 — Flood-It!, F2 — Flood-It! 2; NW — Nadaraya-Watson, "
        "LL — Local Linear, RF — Random Forest. Рамка — победитель по RMSE; "
        "* — модель обыгрывает naive baseline.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_plot()
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
