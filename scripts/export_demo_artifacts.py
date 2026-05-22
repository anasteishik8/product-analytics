"""Экспорт demo-артефактов для Streamlit-приложения.

Производит:
    results/demo_forecast_predictions.csv

Скрипт воспроизводит логику notebook 06b на основе зафиксированных
конфигураций и модулей проекта. Зависимости: WhatIfEngine helpers
(scaler, bandwidth, lag features), winner_registry, final_config.json,
forecast_validation_recursive_summary.csv, horizon_safe.csv.

Запуск: python scripts/export_demo_artifacts.py
Детерминизм: ВЕСЬ pipeline без MC-шума (deterministic recursive forecast).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.streamlit.lib.winner_registry import resolve_winner_model, WhatIfUnavailable
from app.streamlit.lib.what_if_engine import (
    APP_ID_MAP,
    TRAIN_END_IDX,
    TARGET_LAGS,
    L,
    SEED,
    WhatIfEngine,
    _resolve_bandwidth,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "floodit_final.parquet"
SUMMARY_PATH = PROJECT_ROOT / "results" / "forecast_validation_recursive_summary.csv"
HORIZON_PATH = PROJECT_ROOT / "results" / "horizon_safe.csv"
FINAL_CONFIG_PATH = PROJECT_ROOT / "results" / "final_config.json"
OUT_PATH = PROJECT_ROOT / "results" / "demo_forecast_predictions.csv"

H_DEMO = 14
TARGETS = ("stickiness", "dau", "retention_d7")
APPS = ("f1", "f2")
COLUMNS = [
    "app", "target", "model", "date", "day_idx", "horizon",
    "segment", "y_true", "y_pred", "ci_lo", "ci_hi",
    "h_safe_A", "h_demo", "warning",
]


def _deterministic_forecast(fitted, state, h_demo):
    """Без шума: один recursive forecast на h_demo шагов вперёд.

    Использует fitted.scaler для StandardScaler-преобразования (как в notebook 07)
    и x_base_raw из state (без MC-шума). Лаги обновляются рекурсивно из y_pred.
    """
    y_hist = list(state.y_history)
    x_cur = state.x_base_raw.copy()
    preds = []
    for _ in range(h_demo):
        lag_vals = [y_hist[-lag] for lag in TARGET_LAGS]
        x_with_lags = np.concatenate([x_cur, np.asarray(lag_vals, dtype=float)])
        x_scaled = fitted.scaler.transform(x_with_lags.reshape(1, -1))
        y_pred = float(fitted.model.predict(x_scaled)[0])
        preds.append(y_pred)
        y_hist.append(y_pred)
        y_hist.pop(0)
    return np.asarray(preds, dtype=float)


def _mc_forecast_with_ci(
    model, scaler, residual_std, x_base, y_hist_arr,
    base_features, h_demo, n_trajectories=200, seed=42,
):
    """MC-ensemble recursive forecast — для оценки CI per forecast day.

    На каждом шаге траектории к y_pred добавляется residual noise ~N(0, residual_std).
    Feature noise НЕ применяется — operational features остаются на baseline level,
    потому что это plain forecast (не counterfactual What-if).

    Returns
    -------
    medians : np.ndarray shape (h_demo,)
    ci_lo : np.ndarray shape (h_demo,)
    ci_hi : np.ndarray shape (h_demo,)
    """
    rng = np.random.RandomState(seed)
    trajectories = np.zeros((n_trajectories, h_demo), dtype=float)
    for i in range(n_trajectories):
        y_hist = list(y_hist_arr[-L:])
        for h in range(h_demo):
            lag_vals = [y_hist[-lag] for lag in TARGET_LAGS]
            x_with_lags = np.concatenate([x_base, np.asarray(lag_vals, dtype=float)])
            x_scaled = scaler.transform(x_with_lags.reshape(1, -1))
            y_pred = float(model.predict(x_scaled)[0])
            if residual_std > 0:
                y_pred += rng.normal(0.0, residual_std)
            trajectories[i, h] = y_pred
            y_hist.append(y_pred)
            y_hist.pop(0)
    medians = np.median(trajectories, axis=0)
    ci_lo = np.percentile(trajectories, 2.5, axis=0)
    ci_hi = np.percentile(trajectories, 97.5, axis=0)
    return medians, ci_lo, ci_hi


def main() -> int:
    df = pd.read_parquet(DATA_PATH)
    with open(FINAL_CONFIG_PATH) as f:
        final_config = json.load(f)
    horizon_safe = pd.read_csv(HORIZON_PATH).set_index(["app_key", "target"])
    summary = pd.read_csv(SUMMARY_PATH).set_index(["app", "target", "model"])

    engine = WhatIfEngine(df, SUMMARY_PATH, final_config)

    rows: list[dict] = []
    for app in APPS:
        df_app = (
            df[df["app_id"] == APP_ID_MAP[app]]
            .sort_values("date")
            .reset_index(drop=True)
        )
        for target in TARGETS:
            try:
                model_name = resolve_winner_model(app, target, SUMMARY_PATH)
            except WhatIfUnavailable:
                continue

            base_features = list(final_config[app][target]["features"])
            state = engine._extract_baseline_state(app, target, base_features)
            bandwidth = _resolve_bandwidth(model_name, app, target, SUMMARY_PATH)
            fitted = engine._train_winner_model(
                app, target, model_name, state, base_features, bandwidth=bandwidth,
            )

            try:
                h_safe_A = int(horizon_safe.loc[(app, target), "h_safe_A"])
            except KeyError:
                h_safe_A = 0
            try:
                warn = str(summary.loc[(app, target, model_name), "warning"])
            except KeyError:
                warn = ""

            # history segment
            for i in range(TRAIN_END_IDX + 1):
                y_true = df_app.iloc[i][target]
                rows.append({
                    "app": app, "target": target, "model": model_name,
                    "date": str(df_app.iloc[i]["date"]),
                    "day_idx": i, "horizon": 0, "segment": "history",
                    "y_true": float(y_true) if pd.notna(y_true) else np.nan,
                    "y_pred": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                    "h_safe_A": h_safe_A, "h_demo": H_DEMO, "warning": warn,
                })

            # forecast segments (deterministic recursive forecast)
            preds = _deterministic_forecast(fitted, state, H_DEMO)
            # MC CI per forecast day (n_trajectories=200, residual noise only)
            _, ci_lo, ci_hi = _mc_forecast_with_ci(
                model=fitted.model,
                scaler=fitted.scaler,
                residual_std=fitted.residual_std,
                x_base=state.x_base_raw,
                y_hist_arr=state.y_history,
                base_features=base_features,
                h_demo=H_DEMO,
                seed=SEED,
            )
            for h in range(1, H_DEMO + 1):
                day_idx = TRAIN_END_IDX + h
                if day_idx < len(df_app):
                    date_str = str(df_app.iloc[day_idx]["date"])
                    y_true_raw = df_app.iloc[day_idx][target]
                    y_true = float(y_true_raw) if pd.notna(y_true_raw) else np.nan
                else:
                    date_str = ""
                    y_true = np.nan
                segment = "forecast_safe" if h <= h_safe_A else "forecast_demo"
                rows.append({
                    "app": app, "target": target, "model": model_name,
                    "date": date_str, "day_idx": day_idx, "horizon": h,
                    "segment": segment,
                    "y_true": y_true,
                    "y_pred": float(preds[h - 1]),
                    "ci_lo": float(ci_lo[h - 1]),
                    "ci_hi": float(ci_hi[h - 1]),
                    "h_safe_A": h_safe_A, "h_demo": H_DEMO, "warning": warn,
                })

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False, float_format="%.10g")
    print(f"Wrote {len(out_df)} rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
