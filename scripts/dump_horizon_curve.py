"""
dump_horizon_curve.py — экспортирует multi-origin per-horizon error stats
для одной пары (app, target). По умолчанию F1/DAU с winning-моделью из
results/horizon_safe.csv (LocalLinear для F1/DAU по winner_model_at_h_safe_C).

Запускается ОДИН раз — результат коммитится как
results/horizon_curve_<app>_<target>.csv. Используется fig5_2 для
канонической h_safe-кривой: aggregated RMSE по h + threshold + h_safe.

Логика повторяет cells 6, 8, 10, 12, 15, 17 из notebook 05_forecast_horizon.ipynb.

Usage:
    python scripts/dump_horizon_curve.py
    python scripts/dump_horizon_curve.py --app f2 --target stickiness
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import Ridge, ElasticNetCV  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from src.feature_engineering import load_dataset  # noqa: E402
from src.kernel_regression import (  # noqa: E402
    NadarayaWatsonRegressor,
    LocalLinearRegressor,
)
from notebooks._common import select_features_clean, build_xy  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Константы из notebook 05 cells 2/6
SEED = 42
APP_F1 = "com.labpixies.flood"
APP_F2 = "com.google.flood2"
APP_IDS = {"f1": APP_F1, "f2": APP_F2}

ORIGINS = [58, 65, 72, 79, 85]
HORIZONS = list(range(1, 25))  # h = 1..24
STRATEGIES = ["hold_last", "linear_trend"]
MODELS_BASE = [
    "NadarayaWatson", "LocalLinear", "kNN",
    "Ridge", "ElasticNet",
    "RandomForest", "XGBoost",
]
MIN_TRAIN_NONNULL = 15
BW_GRID = np.logspace(-2, 1, 10)

# Загрузка final_config — нужен для threshold A
FINAL_CONFIG = json.loads(
    (ROOT / "results" / "final_config.json").read_text(encoding="utf-8")
)


# === COPY FROM notebook 05 cell 8 ===
def _extrapolate_features(X_last, strategy, horizon, slopes=None, bounds=None):
    """Экстраполирует вектор признаков на `horizon` дней вперёд."""
    if strategy == "hold_last":
        X_h = X_last.copy()
    elif strategy == "linear_trend":
        if slopes is None:
            raise ValueError("slopes required for linear_trend")
        X_h = X_last + slopes * horizon
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if bounds is not None:
        X_min, X_max = bounds
        X_h = np.clip(X_h, X_min, X_max)
    return X_h


def _origin_train_slice(df_app, origin):
    """Возвращает train-подвыборку: rows 0..origin (включительно) of df_app."""
    return df_app.iloc[: origin + 1].copy()


def _valid_origin(df_app, origin, target, min_nonnull=MIN_TRAIN_NONNULL):
    """True если на этом origin'е train содержит ≥ min_nonnull non-null target."""
    if origin >= len(df_app):
        return False
    train = _origin_train_slice(df_app, origin)
    return train[target].notna().sum() >= min_nonnull


# === COPY FROM notebook 05 cell 10 ===
def _fit_model(model_name, X_train_sc, y_train, random_state=42):
    """Фитит свежую модель указанного типа на (X_train_sc, y_train)."""
    try:
        n_train = len(X_train_sc)

        if model_name == "NadarayaWatson":
            m = NadarayaWatsonRegressor(kernel="gaussian", bandwidth=1.0)
            n_splits = min(5, max(2, n_train // 5))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            m.bandwidth_cv(X_train_sc, y_train, cv=cv, bandwidths=BW_GRID,
                           random_state=random_state)
            m.fit(X_train_sc, y_train)
            return m

        if model_name == "LocalLinear":
            m = LocalLinearRegressor(kernel="gaussian", bandwidth=1.0)
            n_splits = min(5, max(2, n_train // 5))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            m.bandwidth_cv(X_train_sc, y_train, cv=cv, bandwidths=BW_GRID,
                           random_state=random_state)
            m.fit(X_train_sc, y_train)
            return m

        if model_name == "kNN":
            k = 5 if n_train >= 25 else 3
            m = KNeighborsRegressor(n_neighbors=k)
            m.fit(X_train_sc, y_train)
            return m

        if model_name == "Ridge":
            m = Ridge(alpha=1.0, random_state=random_state)
            m.fit(X_train_sc, y_train)
            return m

        if model_name == "ElasticNet":
            m = ElasticNetCV(
                l1_ratio=[0.5], alphas=np.logspace(-2, 1, 10),
                cv=min(3, max(2, n_train // 10)), max_iter=20000,
                random_state=random_state, n_jobs=1,
            )
            m.fit(X_train_sc, y_train)
            return m

        if model_name == "RandomForest":
            m = RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=1
            )
            m.fit(X_train_sc, y_train)
            return m

        if model_name == "XGBoost":
            m = XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=random_state, verbosity=0,
            )
            m.fit(X_train_sc, y_train)
            return m

        return None
    except Exception as e:
        warnings.warn(f"_fit_model failed for {model_name}: {e}", stacklevel=2)
        return None


# === COPY FROM notebook 05 cell 12 ===
def _walk_forward_pair(df, app_id, target, strategy,
                       origins=None, horizons=None, models=None,
                       max_features=15, random_state=42):
    """Walk-forward validation для одной (app, target, strategy) комбинации."""
    origins = origins or ORIGINS
    horizons = horizons or HORIZONS
    models_to_use = list(models) if models else list(MODELS_BASE)

    # AR(1) только для retention_d7
    if target == "retention_d7" and "AR(1)" not in models_to_use:
        models_to_use = models_to_use + ["AR(1)"]

    df_app = df[df["app_id"] == app_id].sort_values("date").reset_index(drop=True)
    rows = []

    for origin in origins:
        if not _valid_origin(df_app, origin, target):
            warnings.warn(
                f"Skip origin={origin} for {app_id}/{target}: "
                f"n_train_nonnull < {MIN_TRAIN_NONNULL}"
            )
            continue

        train_df = _origin_train_slice(df_app, origin)
        n_train_nonnull = int(train_df[target].notna().sum())

        try:
            fnames = select_features_clean(
                train_df, target, max_features=max_features,
                random_state=random_state,
            )
        except ValueError as e:
            warnings.warn(f"select_features_clean failed at origin={origin}: {e}")
            continue

        X_train_raw, y_train, _medians = build_xy(train_df, target, fnames)
        if len(X_train_raw) < MIN_TRAIN_NONNULL:
            warnings.warn(f"Empty X_train at origin={origin} after build_xy")
            continue

        scaler = StandardScaler().fit(X_train_raw)
        X_train_sc = scaler.transform(X_train_raw)

        X_last = X_train_raw[-1]
        bounds = (X_train_raw.min(axis=0), X_train_raw.max(axis=0))

        slopes = None
        if strategy == "linear_trend":
            n_recent = min(14, len(X_train_raw))
            X_recent = X_train_raw[-n_recent:]
            t_idx = np.arange(n_recent, dtype=float)
            slopes = np.array([
                np.polyfit(t_idx, X_recent[:, j], 1)[0]
                for j in range(X_recent.shape[1])
            ])

        X_at_h_scaled = {}
        for h in horizons:
            X_h_raw = _extrapolate_features(X_last, strategy, h, slopes, bounds)
            X_at_h_scaled[h] = scaler.transform(X_h_raw.reshape(1, -1))

        for model_name in models_to_use:
            if model_name == "AR(1)":
                from statsmodels.tsa.arima.model import ARIMA
                try:
                    ar1 = ARIMA(y_train, order=(1, 0, 0)).fit()
                    y_pred_all = ar1.forecast(steps=max(horizons))
                except Exception as e:
                    warnings.warn(f"AR(1) fit failed at origin={origin}: {e}")
                    continue

                for h in horizons:
                    actual_idx = origin + h
                    if actual_idx >= len(df_app):
                        break
                    y_actual = df_app.iloc[actual_idx][target]
                    if pd.isna(y_actual):
                        continue
                    y_p = float(y_pred_all[h - 1])
                    rows.append({
                        "app_id": app_id, "target": target, "strategy": strategy,
                        "origin": origin, "model": model_name, "horizon": h,
                        "y_actual": float(y_actual), "y_pred": y_p,
                        "sq_error": (y_actual - y_p) ** 2,
                        "n_train_nonnull": n_train_nonnull,
                    })
                continue

            model = _fit_model(model_name, X_train_sc, y_train,
                               random_state=random_state)
            if model is None:
                continue

            for h in horizons:
                actual_idx = origin + h
                if actual_idx >= len(df_app):
                    break
                y_actual = df_app.iloc[actual_idx][target]
                if pd.isna(y_actual):
                    continue
                try:
                    y_p = float(model.predict(X_at_h_scaled[h])[0])
                except Exception as e:
                    warnings.warn(f"{model_name} predict failed h={h}: {e}")
                    continue
                rows.append({
                    "app_id": app_id, "target": target, "strategy": strategy,
                    "origin": origin, "model": model_name, "horizon": h,
                    "y_actual": float(y_actual), "y_pred": y_p,
                    "sq_error": (y_actual - y_p) ** 2,
                    "n_train_nonnull": n_train_nonnull,
                })

    return pd.DataFrame(rows)


# === COPY FROM notebook 05 cell 15 ===
def _aggregate_rmse(results_df):
    """results_df -> agg_df с колонками
    [app_id, target, strategy, model, horizon, rmse_mean, rmse_std, n_origins].
    """
    grouped = results_df.groupby(
        ["app_id", "target", "strategy", "model", "horizon"], as_index=False
    ).agg(
        sq_error_mean=("sq_error", "mean"),
        sq_error_std=("sq_error", "std"),
        n_origins=("origin", "nunique"),
    )
    grouped["rmse_mean"] = np.sqrt(grouped["sq_error_mean"])
    grouped["rmse_std"] = grouped["sq_error_std"] / (
        2 * np.sqrt(grouped["sq_error_mean"] + 1e-12)
    )
    return grouped[[
        "app_id", "target", "strategy", "model", "horizon",
        "rmse_mean", "rmse_std", "n_origins",
    ]]


# === COPY FROM notebook 05 cell 17 ===
def _compute_thresholds(df_app, target, agg_pair):
    """Возвращает значения трёх порогов (A, B, C) для пары.

    A: 1.5 * CV_RMSE_train (из final_config)
    B: std(y) train + test window — naive RMSE
    C: 0.707 * std(y_test_window) — равно R²=0.5
    """
    app_key = "f1" if df_app["app_id"].iloc[0] == APP_F1 else "f2"
    cfg = FINAL_CONFIG[app_key][target]
    thr_A = 1.5 * cfg["cv_metrics"]["rmse_mean"]

    y_full = df_app[target].dropna().values
    thr_B = float(np.std(y_full))

    y_test_window = df_app.iloc[
        ORIGINS[0] + 1: ORIGINS[-1] + max(HORIZONS) + 1
    ][target].dropna().values
    if len(y_test_window) > 1:
        thr_C = 0.707 * float(np.std(y_test_window))
    else:
        thr_C = float("nan")

    return {"A": thr_A, "B": thr_B, "C": thr_C}


def dump(app: str, target: str, strategy: str = "hold_last") -> pd.DataFrame:
    """Запускает walk-forward для одной (app, target, strategy), агрегирует
    по horizon, фильтрует к winning-модели из horizon_safe.csv.
    """
    df = load_dataset()
    app_id = APP_IDS[app]
    df_app = df[df.app_id == app_id].sort_values("date").reset_index(drop=True)

    # Winning model для (app, target) из horizon_safe.csv
    hs = pd.read_csv(ROOT / "results/horizon_safe.csv")
    row = hs[(hs.app_key == app) & (hs.target == target)]
    if row.empty:
        sys.exit(f"Не найдено (app={app}, target={target}) в horizon_safe.csv")
    model_name = row.iloc[0]["winner_model_at_h_safe_C"]
    best_strategy_csv = row.iloc[0]["best_strategy"]
    if strategy != best_strategy_csv:
        print(
            f"NOTE: requested strategy={strategy}, "
            f"horizon_safe.csv best_strategy={best_strategy_csv}"
        )

    print(f"Running walk-forward: app={app}, target={target}, "
          f"strategy={strategy}, model={model_name}...")

    # Run walk-forward (все модели, чтобы воспроизвести агрегаты из ноутбука)
    results_df = _walk_forward_pair(df, app_id, target, strategy)
    agg_df = _aggregate_rmse(results_df)

    # Filter к winning-модели
    agg_pair = agg_df[
        (agg_df["app_id"] == app_id)
        & (agg_df["target"] == target)
        & (agg_df["strategy"] == strategy)
        & (agg_df["model"] == model_name)
    ].copy().sort_values("horizon").reset_index(drop=True)

    # Compute thresholds
    thresholds = _compute_thresholds(df_app, target, agg_pair)

    # Финальный DataFrame для дампа
    out = agg_pair.copy()
    out["app"] = app
    # колонки app_id, target, strategy, model, horizon, rmse_mean, rmse_std, n_origins уже есть
    out["threshold_A"] = thresholds["A"]
    out["threshold_B"] = thresholds["B"]
    out["threshold_C"] = thresholds["C"]
    out["h_safe_A"] = int(row.iloc[0]["h_safe_A"])
    out["h_safe_B"] = int(row.iloc[0]["h_safe_B"])
    out["h_safe_C"] = int(row.iloc[0]["h_safe_C"])

    cols = [
        "app", "app_id", "target", "strategy", "model", "horizon",
        "rmse_mean", "rmse_std", "n_origins",
        "threshold_A", "threshold_B", "threshold_C",
        "h_safe_A", "h_safe_B", "h_safe_C",
    ]
    return out[cols]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="f1")
    p.add_argument("--target", default="dau")
    p.add_argument("--strategy", default="hold_last")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.out is None:
        args.out = ROOT / f"results/horizon_curve_{args.app}_{args.target}.csv"

    df_out = dump(args.app, args.target, args.strategy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"OK wrote {args.out}: {len(df_out)} rows "
          f"(h=1..{int(df_out['horizon'].max())})")

    # Sanity: h_safe_A в дампе должен совпадать с h_safe_A в horizon_safe.csv
    expected = pd.read_csv(ROOT / "results/horizon_safe.csv")
    expected_row = expected[
        (expected.app_key == args.app) & (expected.target == args.target)
    ]
    if not expected_row.empty:
        h_safe_expected = int(expected_row.iloc[0]["h_safe_A"])
        h_safe_dump = int(df_out["h_safe_A"].iloc[0])
        ok = h_safe_dump == h_safe_expected
        print(
            f"h_safe sanity: dump={h_safe_dump}, "
            f"horizon_safe.csv={h_safe_expected}, ok={ok}"
        )
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
