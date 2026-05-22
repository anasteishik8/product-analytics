"""make_fig2_3.py — fig2_3: NaN-карта покрытия по группам признаков.

Heatmap: Y — группы признаков × продукт (8 строк), X — дни наблюдений.
Цвет ячейки = доля непустых значений в группе на этот день.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, save_figure


F1_ID = "com.labpixies.flood"
F2_ID = "com.google.flood2"


GROUP_COLUMNS = {
    "Product (in-app)": [
        "daily_installs", "daily_uninstalls", "net_installs",
        "dau", "mau", "wau", "stickiness",
        "retention_d1", "retention_d7", "retention_d30",
        "total_sessions", "median_session_duration", "avg_session_duration",
        "crash_rate", "onboarding_completion_rate", "peak_hour",
        "crash_rate_ma7", "avg_session_duration_ma7",
        "dau_growth_rate", "dau_growth_rate_7d", "install_growth_rate",
        "retention_d1_change", "retention_trend", "engagement_score",
    ],
    "Store / рейтинги": [
        "rating", "reviews_count", "positive", "negative", "positive_ratio",
        "sentiment_score", "reviews_30d", "positive_30d", "negative_30d",
        "sentiment_30d", "rating_ma30", "positive_ratio_ma30",
        "rating_volatility_7d", "rating_trend", "negative_change",
        "has_store_data",
    ],
    "Market / конкуренция": [
        "competitor_count", "category_rank", "competitor_count_change",
        "competition_volatility_7d", "category_rank_ma14",
        "category_rank_ma30", "rank_volatility_7d", "rank_improvement",
        "market_trend",
    ],
    "External / медиа": [
        "media_coverage", "market_sentiment", "media_coverage_ma7",
        "media_coverage_change", "seasonality_change",
        "downloads_index", "downloads_index_ma30",
    ],
}


def _coverage_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[pd.Timestamp]]:
    """Возвращает матрицу coverage (rows × dates), список row-меток и список дат."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    rows = []
    labels = []
    product_label = {F1_ID: "F1", F2_ID: "F2"}
    for app_id in [F1_ID, F2_ID]:
        sub = df[df["app_id"] == app_id].set_index("date").sort_index()
        for group_name, cols in GROUP_COLUMNS.items():
            cols_present = [c for c in cols if c in sub.columns]
            # доля непустых значений по строке (день) среди cols_present
            non_null = sub[cols_present].notna().mean(axis=1)
            # выровнять к общему списку дат
            non_null = non_null.reindex(dates)
            rows.append(non_null.values)
            labels.append(f"{product_label[app_id]} · {group_name}")

    matrix = np.vstack(rows)
    return matrix, labels, dates


def main() -> Path:
    apply_vkr_style()

    df = pd.read_parquet(ROOT / "data" / "processed" / "floodit_final.parquet")
    matrix, labels, dates = _coverage_matrix(df)

    fig, ax = plt.subplots(figsize=(11, 5))
    # imshow с extent по числовым датам, чтобы x-ось была date-aware
    x_min = mdates.date2num(dates[0])
    x_max = mdates.date2num(dates[-1])
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        extent=(x_min, x_max, len(labels) - 0.5, -0.5),
        interpolation="nearest",
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    # Разделители между F1- и F2-блоками (4 группы)
    ax.axhline(3.5, color="#222222", lw=1.0)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    ax.set_xlabel("Дата наблюдения")
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Доля непустых значений", fontsize=9)

    fig.suptitle(
        "Покрытие данных по группам признаков (F1 / F2)",
        fontsize=12, y=0.98,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)

    out = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig2_3.pdf"
    save_figure(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = main()
    size_kb = out.stat().st_size / 1024
    print(f"OK fig2_3 saved: {out} ({size_kb:.1f} KB)")
