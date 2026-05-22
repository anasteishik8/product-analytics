"""
make_chapter5_figures.py — генерирует все 8 рисунков главы 5 в едином стиле.

Каждая функция читает источник из results/*.csv, строит график через
matplotlib с единым стилем, сохраняет в vkr/v2/artifacts/figures/figX_Y.pdf
и регистрирует через freeze_artifact.

Usage:
    python scripts/make_chapter5_figures.py             # все 8
    python scripts/make_chapter5_figures.py --only 1 6  # только fig5_1 и fig5_6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


FIGS_OUT = ROOT / "vkr/v2/artifacts/figures"
REGISTRY = ROOT / "vkr/v2/artifacts/registry.csv"


def _freeze(id_: str, source: Path, caption: str) -> None:
    artifact = FIGS_OUT / f"{id_}.pdf"
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/freeze_artifact.py"), id_,
            "--source-kind", "results",
            "--source", str(source),
            "--artifact", str(artifact),
            "--caption", caption,
            "--status", "frozen",
        ],
        check=True,
        cwd=ROOT,
    )


def make_fig5_1() -> None:
    """Relative RMSE heatmap: RMSE / min_RMSE per (app, target). Решает
    проблему несравнимости абсолютных RMSE между DAU (сотни) и долевыми
    метриками (~0.03). Чем темнее цвет, тем хуже модель относительно лучшей в паре."""
    src = ROOT / "results/forecast_validation_recursive_summary.csv"
    df = pd.read_csv(src)
    # RMSE / min RMSE в пределах каждой пары (app, target)
    df["rel_rmse"] = df["RMSE"] / df.groupby(["app", "target"])["RMSE"].transform("min")
    pivot = df.pivot_table(index="model", columns=["app", "target"],
                            values="rel_rmse", aggfunc="min")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=1.0)
    # Цвет акцента для победителей — зелёный из VKR_PALETTE (тот же, что DEVELOP в карточках)
    winner_color = VKR_PALETTE[2]
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            is_winner = abs(val - 1.0) < 1e-6
            if is_winner:
                # Зелёная рамка вокруг ячейки-победителя
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor=winner_color, linewidth=2.4,
                ))
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=winner_color, fontsize=11, weight="bold")
            else:
                text_color = "white" if val > 2.0 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{a}/{t}" for a, t in pivot.columns], rotation=30, ha="right")
    ax.set_title("Относительный RMSE моделей по парам (1.00 = лучшая в паре, выделена зелёным)")
    fig.colorbar(im, ax=ax, label="RMSE / min(RMSE)")
    save_figure(fig, FIGS_OUT / "fig5_1.pdf")
    plt.close(fig)
    _freeze(
        "fig5_1", src,
        "Относительный RMSE моделей по 6 парам (нормирован на лучшую в паре). "
        "Решает несравнимость абсолютных RMSE между DAU и долевыми метриками.",
    )


def make_fig5_2() -> None:
    """Каноническая h_safe-кривая для F1/DAU.
    Источник: results/horizon_curve_f1_dau.csv (dumped in Task 10).
    Multi-origin aggregated RMSE per horizon + threshold + h_safe annotation.

    Adaptation note: Task 10's dump uses columns `horizon`, `rmse_mean`,
    `rmse_std`, `threshold_A`, `h_safe_A` (not `h`, `rmse_median`,
    `rmse_q25/q75`, `threshold_rmse` as in plan). Band uses mean±std."""
    src = ROOT / "results/horizon_curve_f1_dau.csv"
    df = pd.read_csv(src).sort_values("horizon").reset_index(drop=True)
    median_col = "rmse_mean"
    h_safe = int(df["h_safe_A"].iloc[0])
    threshold = (
        float(df["threshold_A"].iloc[0])
        if "threshold_A" in df.columns and pd.notna(df["threshold_A"].iloc[0])
        else None
    )

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df["horizon"], df[median_col],
            color=VKR_PALETTE[0], marker="o", label="RMSE, среднее по origins")
    if "rmse_std" in df.columns:
        lo = df[median_col] - df["rmse_std"]
        hi = df[median_col] + df["rmse_std"]
        ax.fill_between(df["horizon"], lo, hi,
                         alpha=0.2, color=VKR_PALETTE[0], label="± 1 std")
    if threshold is not None:
        ax.axhline(threshold, color=VKR_PALETTE[4], linestyle="--",
                   linewidth=1.0, label=f"Threshold ({threshold:.2f})")
    ax.axvline(h_safe, color=VKR_PALETTE[1], linestyle="--",
               label=f"h_safe = {h_safe}")
    ax.set_xlabel("Горизонт прогноза h, дн.")
    ax.set_ylabel("RMSE (aggregated по origins)")
    ax.set_title("F1/DAU: определение h_safe (точка пересечения с порогом)")
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, FIGS_OUT / "fig5_2.pdf")
    plt.close(fig)
    _freeze(
        "fig5_2", src,
        "F1/DAU: каноническая h_safe-кривая. Multi-origin aggregated RMSE per h "
        "(notebook 05 _walk_forward_pair + _aggregate_rmse), threshold через "
        "_compute_thresholds, h_safe_A=2 дня из horizon_safe.csv.",
    )


HISTORY_TAIL_DAYS = 30  # zoom-окно: последние N дней истории + весь forecast
SAFE_BG = "#e3edf5"     # светло-голубой фон safe-зоны (тон от VKR_PALETTE[0])
RISKY_BG = "#f5e3e3"    # светло-красный фон risky-зоны (тон от VKR_PALETTE[1])


def _plot_forecast_panel(ax, sub: pd.DataFrame, ylabel: str, title: str) -> None:
    """Forecast-панель для fig5_3/fig5_4 с safe/risky background-сегментацией.

    Background per horizon:
      - h <= h_safe_A → SAFE_BG (голубоватый)
      - h >  h_safe_A → RISKY_BG (красноватый)
    + вертикальная линия train/test split + (опц.) пунктир на границе h_safe.
    Zoom на последние HISTORY_TAIL_DAYS дней истории + forecast (данные не обрезаются).
    """
    h = sub[sub["segment"] == "history"]
    fs = sub[sub["segment"] == "forecast_safe"]
    fd = sub[sub["segment"] == "forecast_demo"]
    forecast = sub[sub["segment"] != "history"]

    if forecast.empty:
        return

    forecast_start = int(forecast["day_idx"].min())
    forecast_end = int(forecast["day_idx"].max())
    h_safe_A = int(forecast["h_safe_A"].iloc[0])
    h_demo = int(forecast["h_demo"].iloc[0])

    # Background-сегментация: safe-зона vs risky-зона
    # Safe-зона: day_idx ∈ [forecast_start, forecast_start + h_safe_A - 1]
    # Risky-зона: day_idx ∈ [forecast_start + h_safe_A, forecast_end]
    safe_end_x = forecast_start + h_safe_A - 0.5  # граница safe (внутри последнего safe-дня)
    safe_zone_max_x = min(safe_end_x, forecast_end + 0.5)
    if h_safe_A > 0:
        ax.axvspan(forecast_start - 0.5, safe_zone_max_x,
                   color=SAFE_BG, alpha=0.7, zorder=0,
                   label=f"Safe-зона (h ≤ {h_safe_A})")
    if h_safe_A < h_demo:
        ax.axvspan(max(safe_end_x, forecast_start - 0.5), forecast_end + 0.5,
                   color=RISKY_BG, alpha=0.7, zorder=0,
                   label=f"Risky-зона (h > {h_safe_A})")

    # История (вся, без обрезки)
    ax.plot(h["day_idx"], h["y_true"], color=VKR_PALETTE[4],
            linewidth=1.2, label="История (факт)")

    # Forecast safe — чуть менее крупные маркеры (5)
    if not fs.empty:
        ax.plot(fs["day_idx"], fs["y_true"], color=VKR_PALETTE[0],
                marker="o", markersize=5, linestyle="-",
                linewidth=1.4, label="Forecast-safe (факт)")
        ax.plot(fs["day_idx"], fs["y_pred"], color=VKR_PALETTE[1],
                linestyle="--", linewidth=1.8, label="Прогноз")
        ax.fill_between(fs["day_idx"], fs["ci_lo"], fs["ci_hi"],
                         color=VKR_PALETTE[1], alpha=0.22, label="CI95 (safe)")

    # Forecast demo
    if not fd.empty:
        ax.plot(fd["day_idx"], fd["y_true"], color=VKR_PALETTE[0],
                marker="o", markersize=6, linestyle="-", alpha=0.7,
                linewidth=1.4, label="Forecast-demo (факт)")
        ax.plot(fd["day_idx"], fd["y_pred"], color=VKR_PALETTE[1],
                linestyle=":", linewidth=1.8, alpha=0.85, label="Прогноз-demo")
        ax.fill_between(fd["day_idx"], fd["ci_lo"], fd["ci_hi"],
                         color=VKR_PALETTE[1], alpha=0.10)

    # Train/test split — вертикальная линия с подписью
    ax.axvline(forecast_start - 0.5, color=VKR_PALETTE[4],
               linestyle="-", linewidth=1.0, alpha=0.8)
    ax.annotate(
        "train/test split", xy=(forecast_start - 0.5, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(2, -8), textcoords="offset points",
        fontsize=8.5, color=VKR_PALETTE[4],
        rotation=90, va="top", ha="left",
    )

    # h_safe граница — пунктир, только если safe-зона частична (0 < h_safe < h_demo)
    if 0 < h_safe_A < h_demo:
        ax.axvline(safe_end_x, color=VKR_PALETTE[1],
                   linestyle="--", linewidth=1.2, alpha=0.8)
        ax.annotate(
            f"h_safe = {h_safe_A}", xy=(safe_end_x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -8), textcoords="offset points",
            fontsize=8.5, color=VKR_PALETTE[1], weight="bold",
            rotation=90, va="top", ha="left",
        )

    # ZOOM окно
    ax.set_xlim(forecast_start - HISTORY_TAIL_DAYS - 0.5, forecast_end + 0.5)

    # Y-лимиты под видимое окно
    visible_mask = (sub["day_idx"] >= forecast_start - HISTORY_TAIL_DAYS) & \
                   (sub["day_idx"] <= forecast_end)
    visible = sub[visible_mask]
    y_cols = ["y_true", "y_pred", "ci_lo", "ci_hi"]
    y_vals = pd.concat([visible[c] for c in y_cols if c in visible.columns]).dropna()
    if len(y_vals) > 0:
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        y_pad = (y_max - y_min) * 0.10 if y_max > y_min else abs(y_max) * 0.1 or 1.0
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("День, day_idx")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8.5)


def make_fig5_3() -> None:
    """F1/stickiness — пример рабочего прогноза.
    История обрезана до последних HISTORY_TAIL_DAYS дней, forecast-зона заштрихована."""
    src = ROOT / "results/demo_forecast_predictions.csv"
    df = pd.read_csv(src)
    sub = df[(df["app"] == "f1") & (df["target"] == "stickiness")].sort_values("day_idx")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    _plot_forecast_panel(
        ax, sub,
        ylabel="stickiness",
        title="F1/stickiness: рабочий прогноз (Ridge) — h_safe=24 ≥ demo=14: вся область safe",
    )
    save_figure(fig, FIGS_OUT / "fig5_3.pdf")
    plt.close(fig)
    _freeze(
        "fig5_3", src,
        "F1/stickiness: рабочий recursive forecast (Ridge). h_safe=24 ≥ demo=14, "
        "поэтому весь demo-window попадает в safe-зону (голубой фон). "
        f"X-axis zoom на последние {HISTORY_TAIL_DAYS} дн. истории + forecast; "
        "вертикальная линия — train/test split.",
    )


def make_fig5_4() -> None:
    """F2/retention_d7 — пример рискованного прогноза.
    История обрезана до последних HISTORY_TAIL_DAYS дней, forecast-зона заштрихована."""
    src = ROOT / "results/demo_forecast_predictions.csv"
    df = pd.read_csv(src)
    sub = df[(df["app"] == "f2") & (df["target"] == "retention_d7")].sort_values("day_idx")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    _plot_forecast_panel(
        ax, sub,
        ylabel="retention_d7",
        title="F2/retention_d7: рискованный прогноз (NW) — h_safe=0 < demo=14: вся область risky",
    )
    save_figure(fig, FIGS_OUT / "fig5_4.pdf")
    plt.close(fig)
    _freeze(
        "fig5_4", src,
        "F2/retention_d7: рискованный recursive forecast (NW). h_safe=0 — "
        "безопасного горизонта нет, весь demo-window лежит за пределами (красный фон). "
        f"X-axis zoom на последние {HISTORY_TAIL_DAYS} дн. истории + forecast; "
        "вертикальная линия — train/test split.",
    )


def make_fig5_5() -> None:
    """SHAP top-10 для F2/DAU. Зелёный — положительный, красный — отрицательный."""
    src = ROOT / "results/scenario_shap_summary.csv"
    df = pd.read_csv(src)
    sub = df[(df["app"] == "f2") & (df["target"] == "dau")].copy()
    sub = sub.sort_values("shap_abs", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = [VKR_PALETTE[2] if v > 0 else VKR_PALETTE[1] for v in sub["shap_value"]]
    ax.barh(sub["feature"], sub["shap_value"], color=colors)
    ax.axvline(0, color=VKR_PALETTE[4], linewidth=0.6)
    ax.set_xlabel("SHAP value (вклад в прогноз)")
    ax.set_title("Топ-10 SHAP-драйверов для F2/DAU")
    save_figure(fig, FIGS_OUT / "fig5_5.pdf")
    plt.close(fig)
    _freeze(
        "fig5_5", src,
        "Топ-10 SHAP-драйверов для модели F2/DAU. Зелёный — положительный вклад, "
        "красный — отрицательный.",
    )


def make_fig5_6() -> None:
    """Каноническая MC-гистограмма: cross_product_mimic для F2/DAU.
    Источник — raw N=1000 симуляций из dump_mc_distribution.py."""
    src = ROOT / "results/mc_f2_dau_cross_product_mimic.csv"
    df = pd.read_csv(src)
    delta = df["delta_pct_vs_baseline"]
    median = delta.median()
    lo, hi = delta.quantile([0.025, 0.975])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta, bins=40, color=VKR_PALETTE[0], alpha=0.7, edgecolor="white")
    ax.axvline(0, color=VKR_PALETTE[4], linestyle="-", linewidth=1.0, label="Baseline (0%)")
    ax.axvline(median, color=VKR_PALETTE[1], linestyle="--", linewidth=1.2,
               label=f"Медиана ({median:+.2f}%)")
    ax.axvline(lo, color=VKR_PALETTE[1], linestyle=":", alpha=0.7,
               label=f"CI95 [{lo:+.2f}; {hi:+.2f}]")
    ax.axvline(hi, color=VKR_PALETTE[1], linestyle=":", alpha=0.7)
    ax.set_xlabel("Δ DAU, % к baseline")
    ax.set_ylabel("Частота симуляций")
    ax.set_title("F2/DAU: распределение эффекта сценария cross_product_mimic\n(Monte Carlo, N=1000)")
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, FIGS_OUT / "fig5_6.pdf")
    plt.close(fig)
    _freeze(
        "fig5_6", src,
        "F2/DAU: каноническое MC-распределение эффекта cross_product_mimic (N=1000). "
        "Медиана и CI95 рассчитаны из raw simulation array, дампленного в Task 10.",
    )


def make_fig5_7() -> None:
    """Verdict-карточки F1 и F2 рядом — две колонки в одной фигуре."""
    src = ROOT / "results/viability_summary.csv"
    df = pd.read_csv(src)
    f1 = df[(df["app_id"] == "com.labpixies.flood") & (df["model"] == "M3")].iloc[0]
    f2 = df[(df["app_id"].str.contains("flood2")) & (df["model"] == "M3")].iloc[0]

    verdict_color_map = {
        "DEVELOP": VKR_PALETTE[2],
        "MONITOR": VKR_PALETTE[3],
        "DISCONTINUE": VKR_PALETTE[1],
    }

    def _card(ax, row, product_label: str) -> None:
        ax.axis("off")
        verdict = row["final_verdict"]
        verdict_color = verdict_color_map.get(verdict, VKR_PALETTE[4])
        lines = [
            ("Вердикт", verdict),
            ("T95", f"{row['T95']:.2f}"),
            ("T99", f"{row['T99']:.2f}"),
            ("D² (mean)", f"{row['D2_mean']:.2f}"),
            ("bad share", f"{row['bad_share_mean']:.2%}"),
            ("DEVELOP, дн.", f"{row['days_develop']:.0f}"),
            ("MONITOR, дн.", f"{row['days_monitor']:.0f}"),
            ("DISCONTINUE, дн.", f"{row['days_discontinue']:.0f}"),
        ]
        # Цветная полоса-заголовок по вердикту
        ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, transform=ax.transAxes,
                                    facecolor=verdict_color, alpha=0.15, linewidth=0))
        ax.text(0.5, 0.965, product_label, fontsize=12, weight="bold",
                ha="center", va="center", transform=ax.transAxes)
        # Поля
        y0 = 0.85
        for i, (label, value) in enumerate(lines):
            y = y0 - i * 0.095
            ax.text(0.05, y, label + ":", fontsize=10.5, weight="bold",
                    transform=ax.transAxes)
            color = verdict_color if label == "Вердикт" else "black"
            weight = "bold" if label == "Вердикт" else "normal"
            ax.text(0.55, y, value, fontsize=10.5, color=color, weight=weight,
                    transform=ax.transAxes)
        # Рамка
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, edgecolor="#999999", linewidth=0.8))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _card(axes[0], f1, "Flood-It! (M3)")
    _card(axes[1], f2, "Flood-It! 2 (M3)")
    fig.subplots_adjust(wspace=0.15, left=0.03, right=0.97, top=0.95, bottom=0.05)
    save_figure(fig, FIGS_OUT / "fig5_7.pdf")
    plt.close(fig)
    _freeze(
        "fig5_7", src,
        "Verdict-карточки обоих продуктов (M3): Flood-It! (DEVELOP) слева, "
        "Flood-It! 2 (DISCONTINUE) справа. Цветовой акцент в заголовке = вердикт. "
        "Полный разрез по M1/M2/M3 — см. tab5_6.",
    )


def make_fig5_8() -> None:
    """Сводный dashboard: stacked-bar дней + топ-3 сценария F2/DAU."""
    src = ROOT / "results/viability_summary.csv"
    df_v = pd.read_csv(src)
    df_s = pd.read_csv(ROOT / "results/scenario_analysis_summary.csv")
    df_s = df_s[df_s["scenario_type"] != "baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    pivot = df_v[df_v["model"] == "M3"].set_index("app_id")[
        ["days_develop", "days_monitor", "days_discontinue"]
    ]
    pivot.columns = ["DEVELOP", "MONITOR", "DISCONTINUE"]
    pivot.plot(kind="bar", stacked=True, ax=axes[0],
               color=[VKR_PALETTE[2], VKR_PALETTE[3], VKR_PALETTE[1]])
    axes[0].set_title("Декомпозиция дней по продуктам (M3)")
    axes[0].set_ylabel("Дней")
    axes[0].set_xlabel("")
    # Легенду выносим под график, чтобы не наезжала на бары
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=3, fontsize=9)
    axes[0].set_xticklabels(["Flood-It!", "Flood-It! 2"], rotation=0)

    f2_dau = df_s[(df_s["app"] == "f2") & (df_s["target"] == "dau")].nlargest(
        3, "delta_pct_vs_baseline"
    )
    axes[1].barh(f2_dau["scenario_name"], f2_dau["delta_pct_vs_baseline"],
                  color=VKR_PALETTE[0])
    axes[1].axvline(0, color=VKR_PALETTE[4], linewidth=0.6)
    axes[1].set_title("F2/DAU: топ-3 сценария по эффекту")
    axes[1].set_xlabel("Δ, %")
    # Подложим место под легенду левой панели
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, FIGS_OUT / "fig5_8.pdf")
    plt.close(fig)
    _freeze(
        "fig5_8", src,
        "Сводный dashboard: декомпозиция дней по продуктам (M3) + топ-3 сценария F2/DAU.",
    )


FIGURE_MAKERS = {
    1: make_fig5_1, 2: make_fig5_2, 3: make_fig5_3, 4: make_fig5_4,
    5: make_fig5_5, 6: make_fig5_6, 7: make_fig5_7, 8: make_fig5_8,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=int, nargs="+", default=None,
                   help="Список номеров рисунков (1..8). По умолчанию — все.")
    args = p.parse_args(argv)
    apply_vkr_style()
    nums = args.only or sorted(FIGURE_MAKERS.keys())
    for n in nums:
        print(f"Generating fig5_{n}...")
        FIGURE_MAKERS[n]()
    print(f"OK done: {len(nums)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
