"""Функции построения графиков для Streamlit-приложения.

Все функции возвращают matplotlib.Figure без вызовов st.*.
Это позволяет покрыть их unit-тестами и переиспользовать вне Streamlit.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend для тестов и Streamlit Cloud

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# 5 групп признаков для NaN-heatmap (агрегированная карта)
FEATURE_GROUPS = {
    "engagement": [
        "dau", "mau", "wau", "stickiness", "retention_d1", "retention_d7",
        "retention_d30", "total_sessions", "median_session_duration",
        "avg_session_duration", "engagement_score",
    ],
    "acquisition_quality": [
        "daily_installs", "daily_uninstalls", "net_installs",
        "install_growth_rate", "crash_rate", "onboarding_completion_rate",
    ],
    "store": [
        "rating", "reviews_count", "positive", "negative", "positive_ratio",
        "sentiment_score", "rating_ma30",
    ],
    "market_competition": [
        "competitor_count", "category_rank", "market_trend",
        "media_coverage", "market_sentiment",
    ],
    "engineered_temporal": [
        "crash_rate_ma7", "avg_session_duration_ma7", "dau_growth_rate",
        "dau_growth_rate_7d", "retention_trend", "rating_trend",
    ],
}


def plot_nan_heatmap(df: pd.DataFrame) -> Figure:
    """NaN-heatmap по 5 группам признаков × приложениям.

    Parameters
    ----------
    df : pd.DataFrame
        Датасет с колонкой ``app_id`` и метриками из ``FEATURE_GROUPS``.

    Returns
    -------
    Figure
        Размер 6×4 inches. По оси Y — группы признаков, по X — приложения.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    matrix: list[list[float]] = []
    group_names: list[str] = []
    app_ids = sorted(df["app_id"].unique())
    for group_name, cols in FEATURE_GROUPS.items():
        existing = [c for c in cols if c in df.columns]
        if not existing:
            continue
        row = []
        for app_id in app_ids:
            sub = df[df["app_id"] == app_id][existing]
            row.append(float(sub.isna().mean().mean()))
        matrix.append(row)
        group_names.append(group_name)
    arr = np.asarray(matrix)
    im = ax.imshow(arr, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(app_ids)))
    ax.set_xticklabels(app_ids, rotation=20, ha="right")
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    ax.set_title("Доля пропусков по группам признаков")
    fig.colorbar(im, ax=ax, label="NaN ratio")
    fig.tight_layout()
    return fig


def plot_forecast_with_ci(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    h_safe_A: int,
    h_demo: int,
    target_label: str,
) -> Figure:
    """График history + forecast с CI95-полосой.

    Parameters
    ----------
    history : pd.DataFrame
        Колонки: ``date``, ``y_true``.
    forecast : pd.DataFrame
        Колонки: ``date``, ``y_pred``, ``ci_lo``, ``ci_hi``, ``segment``.
        ``segment ∈ {forecast_safe, forecast_demo}``.
    h_safe_A : int
        Безопасный горизонт прогноза (для подписи).
    h_demo : int
        Демонстрационный горизонт (для подписи).
    target_label : str
        Подпись оси Y.

    Returns
    -------
    Figure
        Размер 10×4 inches.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    # История: визуально отрезаем до последних 30 дней (как в notebook 06b).
    # CSV (demo_forecast_predictions.csv) при этом хранит ВСЮ историю — trim только для графика.
    hist_tail = history.tail(30) if len(history) > 30 else history
    if "y_true" in hist_tail.columns and "date" in hist_tail.columns:
        # Для целей с NaN (например retention_d7) рисуем линию + маркеры ТОЛЬКО по валидным точкам,
        # чтобы не было ложного впечатления «поломанного ряда» из-за разрывов.
        hist_valid = hist_tail.dropna(subset=["y_true"])
        ax.plot(
            hist_valid["date"], hist_valid["y_true"],
            label="history (last 30 days)", color="#1f2937", lw=1.5,
            marker="o", markersize=3,
        )
    safe = forecast[forecast["segment"] == "forecast_safe"]
    demo = forecast[forecast["segment"] == "forecast_demo"]
    if len(safe):
        ax.plot(
            safe["date"], safe["y_pred"],
            color="#3b82f6", lw=2.0,
            label=f"forecast_safe (h≤{h_safe_A})",
        )
        if (
            "ci_lo" in safe.columns and "ci_hi" in safe.columns
            and safe["ci_lo"].notna().any()
        ):
            # Demo CI: построен через MC-ensemble (200 траекторий с residual noise injection,
            # 2.5%/97.5% percentiles). Это НЕ буквально CI95 из notebook 06b (тот использовал
            # residual scaling σ × sqrt(h)). По смыслу близко, но реализация отличается —
            # поэтому подпись "demo CI", чтобы не вводить комиссию в заблуждение.
            ax.fill_between(
                safe["date"], safe["ci_lo"], safe["ci_hi"],
                alpha=0.2, color="#3b82f6", label="demo CI (MC ensemble, 200 траекторий)",
            )
    if len(demo):
        ax.plot(
            demo["date"], demo["y_pred"],
            color="#9ca3af", lw=2.0,
            label=f"forecast_demo (h>{h_safe_A})",
        )
        if (
            "ci_lo" in demo.columns and "ci_hi" in demo.columns
            and demo["ci_lo"].notna().any()
        ):
            ax.fill_between(
                demo["date"], demo["ci_lo"], demo["ci_hi"],
                alpha=0.2, color="#9ca3af",
            )
    # actual markers (overlay) — поверх forecast-сегментов, где y_true не NaN
    fc_all = forecast[forecast["segment"].isin({"forecast_safe", "forecast_demo"})]
    if "y_true" in fc_all.columns:
        actual_pts = fc_all[fc_all["y_true"].notna()]
        if len(actual_pts):
            ax.scatter(
                actual_pts["date"], actual_pts["y_true"],
                s=30, color="black", marker="o", zorder=5, label="actual",
            )
    # вертикальная линия h_safe_A — граница между safe и demo сегментами
    if len(demo) and len(safe):
        boundary_date = demo.sort_values("date").iloc[0]["date"]
        ax.axvline(
            boundary_date, color="#9ca3af", ls=":", lw=1.5,
            label=f"h_safe_A = {h_safe_A}",
        )
    ax.set_ylabel(target_label)
    ax.set_xlabel("Дата")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_delta_histogram(
    delta_pct_distribution: np.ndarray,
    ci_lo: float,
    ci_hi: float,
    median: float,
) -> Figure:
    """Гистограмма распределения delta_pct по MC-симуляциям.

    Parameters
    ----------
    delta_pct_distribution : np.ndarray
        Массив значений Δ% (например, N=1000 MC-симуляций).
    ci_lo, ci_hi : float
        Нижняя/верхняя границы 95% CI.
    median : float
        Медианное значение Δ%.

    Returns
    -------
    Figure
        Размер 6×3 inches. Линии: 0 (красная), median (зелёная), CI95 (синие пунктирные).
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(
        delta_pct_distribution, bins=40,
        color="#bfdbfe", edgecolor="#3b82f6", alpha=0.9,
    )
    ax.axvline(0, color="#ef4444", lw=2, label="нулевой эффект")
    ax.axvline(median, color="#22c55e", lw=2, label=f"median = {median:.2f}%")
    ax.axvline(
        ci_lo, color="#3b82f6", lw=1.5, ls="--",
        label=f"CI95: [{ci_lo:.2f}; {ci_hi:.2f}]%",
    )
    ax.axvline(ci_hi, color="#3b82f6", lw=1.5, ls="--")
    ax.set_xlabel("Δ к baseline, %")
    ax.set_ylabel("Частота")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_verdict_decomposition(decomp: pd.DataFrame) -> Figure:
    """Stacked bar chart days_develop/monitor/discontinue для M1/M2/M3.

    Parameters
    ----------
    decomp : pd.DataFrame
        Колонки: ``model``, ``days_develop``, ``days_monitor``, ``days_discontinue``.

    Returns
    -------
    Figure
        Размер 6×3.5 inches.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.6
    x = np.arange(len(decomp))
    ax.bar(x, decomp["days_develop"], width, label="develop", color="#22c55e")
    ax.bar(
        x, decomp["days_monitor"], width,
        bottom=decomp["days_develop"],
        label="monitor", color="#eab308",
    )
    bot = decomp["days_develop"] + decomp["days_monitor"]
    ax.bar(
        x, decomp["days_discontinue"], width,
        bottom=bot, label="discontinue", color="#ef4444",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(decomp["model"])
    ax.set_ylabel("Days")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
