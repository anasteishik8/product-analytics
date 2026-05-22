"""
dump_mc_distribution.py — раз-генератор raw MC trajectories для одного сценария.

По умолчанию: F2/DAU cross_product_mimic, N=1000, seed=42, h_demo=14.
Запускается ОДИН раз, результат коммитится в results/mc_<app>_<target>_<scenario>.csv.

Воспроизводит logic notebook 07 cell 9 (_recursive_forecast + _run_scenario_mc_trajectory).
НЕ использует ScenarioAnalyzer.run_scenario — он одношаговый.
НЕ использует define_scenarios_floodit — там старые сценарии без cross_product_mimic.

Usage:
    python scripts/dump_mc_distribution.py                    # F2/DAU/cross_product_mimic (default)
    python scripts/dump_mc_distribution.py --app f1 --target dau --scenario combined_improvement
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

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from src.feature_engineering import load_dataset  # noqa: E402
from src.kernel_regression import (  # noqa: E402
    NadarayaWatsonRegressor,
    LocalLinearRegressor,
)
from src.scenario_analysis import ScenarioAnalyzer  # noqa: E402
from notebooks._common import build_xy  # noqa: E402
from app.streamlit.lib.winner_registry import resolve_winner_model  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Константы из notebook 07 cell 2 (verified)
SEED = 42
TRAIN_END = 86            # последний train-день (хардкод, в final_config нет meta.TRAIN_END)
H_DEMO = 14               # горизонт траектории
N_BOOTSTRAP = 2000        # для BCa CI
N_MC_SIMULATIONS = 1000   # MC симуляций per scenario
N_SIMULATIONS = N_MC_SIMULATIONS  # alias для совместимости с сигнатурами
CI_LEVEL = 0.95
TARGET_LAGS = [1, 7]      # consistent с 06b (verified в cell 2)
NOISE_PCT = 0.05          # 5% × std признака для гауссова шума

APP_F1 = "com.labpixies.flood"
APP_F2 = "com.google.flood2"
APP_IDS = {"f1": APP_F1, "f2": APP_F2}

# Выбор winner-модели для сценария — через единый источник правды
# (app/streamlit/lib/winner_registry.py: resolve_winner_model). Логика
# идентична notebook 07 cell 16 и согласована с tab5_1.


def bca_bootstrap_ci(data, alpha=0.05, n_bootstrap=2000, seed=42):
    """Тонкая обёртка над ScenarioAnalyzer.bca_bootstrap_ci (метод не использует self)."""
    return ScenarioAnalyzer.bca_bootstrap_ci(
        None, data, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed,
    )


# === COPY FROM notebook 07 cell 6 (_add_lag_features, _build_extended_xy, _retune_bandwidth) ===
def _add_lag_features(df_app: pd.DataFrame, target: str,
                      lags=TARGET_LAGS) -> pd.DataFrame:
    """Adds target_lag<l> columns. ffill+bfill target before shift to keep
    history for sparse targets.
    """
    out = df_app.copy().sort_values("date").reset_index(drop=True)
    target_filled = out[target].ffill().bfill()
    for lag in lags:
        out[f"target_lag{lag}"] = target_filled.shift(lag)
    return out


def _build_extended_xy(train_df: pd.DataFrame, target: str, base_features: list,
                       train_medians: dict, lags=TARGET_LAGS):
    """Extended feature set [base + target_lag1 + target_lag7]."""
    df_with_lags = _add_lag_features(train_df, target, lags=lags)
    df_target_ok = df_with_lags[df_with_lags[target].notna()].copy()

    X_base, y_full, _ = build_xy(df_target_ok, target, base_features,
                                 train_medians=train_medians)

    lag_names = [f"target_lag{l}" for l in lags]
    lag_values = df_target_ok[lag_names].values

    X_extended = np.hstack([X_base, lag_values])
    mask = ~np.isnan(X_extended).any(axis=1)
    X_clean = X_extended[mask]
    y_clean = np.asarray(y_full)[mask]

    feature_names = list(base_features) + lag_names
    return X_clean, y_clean, feature_names


def _retune_bandwidth(model_class, X_sc, y, kernel,
                      h_grid=None, n_splits=5, random_state=42) -> float:
    """KFold CV bandwidth selection for NW/LocalLinear on extended feature set."""
    if h_grid is None:
        h_grid = np.logspace(-1.5, 1.0, 12)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_rmse = np.full(len(h_grid), np.inf)
    for i, h in enumerate(h_grid):
        fold_rmse = []
        for tr_idx, vl_idx in kf.split(X_sc):
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    m = model_class(kernel=kernel, bandwidth=float(h))
                    m.fit(X_sc[tr_idx], y[tr_idx])
                    pred = np.asarray(m.predict(X_sc[vl_idx]), dtype=float)
                    fold_rmse.append(
                        float(np.sqrt(np.mean((y[vl_idx] - pred) ** 2)))
                    )
            except Exception:
                fold_rmse.append(np.inf)
        cv_rmse[i] = float(np.mean(fold_rmse))
    return float(h_grid[int(np.argmin(cv_rmse))])


# === COPY FROM notebook 07 cell 7 (_train_scenario_model) ===
def _train_scenario_model(app_id: str, target: str, model_name: str,
                          train_df: pd.DataFrame, final_config: dict) -> dict:
    """Trains one of 06b winners on extended feature set."""
    app_key = "f1" if app_id == APP_F1 else "f2"
    cfg = final_config[app_key][target]
    base_features = cfg["features"]
    medians = cfg["feature_train_medians"]

    X_raw, y, feature_names = _build_extended_xy(
        train_df, target, base_features, medians, lags=TARGET_LAGS
    )
    if len(X_raw) < 10:
        raise ValueError(f"Train too small for {app_key}.{target}/{model_name}")

    scaler = StandardScaler().fit(X_raw)
    X_sc = scaler.transform(X_raw)

    bandwidth_used = None
    kernel_eff = cfg.get("kernel") or "gaussian"

    if model_name == "NadarayaWatson":
        bandwidth_used = _retune_bandwidth(
            NadarayaWatsonRegressor, X_sc, y, kernel_eff
        )
        model = NadarayaWatsonRegressor(kernel=kernel_eff, bandwidth=bandwidth_used)
    elif model_name == "LocalLinear":
        bandwidth_used = _retune_bandwidth(
            LocalLinearRegressor, X_sc, y, kernel_eff
        )
        model = LocalLinearRegressor(kernel=kernel_eff, bandwidth=bandwidth_used)
    elif model_name == "Ridge":
        model = Ridge(alpha=1.0, random_state=42)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    elif model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_sc, y)
    # ops_features берутся из final_config (cell 16 notebook 07) — добавляем
    # в return-dict, чтобы _run_scenario_mc_trajectory имел всё необходимое.
    ops_features = cfg.get("operational_features", [])
    return {
        "model": model, "model_name": model_name, "scaler": scaler,
        "feature_names": feature_names, "base_features": base_features,
        "X_train_raw": X_raw, "X_train_sc": X_sc, "y_train": y,
        "bandwidth_used": bandwidth_used,
        "ops_features": ops_features,
    }


# === COPY FROM notebook 07 cell 8 (_get_baseline_state) ===
def _get_baseline_state(df: pd.DataFrame, app_id: str, target: str,
                        base_features: list, train_medians: dict) -> dict:
    """Extracts day-T baseline: feature values at T=TRAIN_END + y_history."""
    df_app = df[df.app_id == app_id].sort_values("date").reset_index(drop=True)

    # Base features at T=86 with median fallback for NaN
    row_T = df_app.iloc[TRAIN_END]
    x_base = np.array([
        row_T[f] if (f in df_app.columns and pd.notna(row_T.get(f, np.nan)))
        else train_medians.get(f, 0.0)
        for f in base_features
    ], dtype=float)

    # y_history: forward-fill from last valid in train
    y_full = df_app[target].values
    train_y = y_full[: TRAIN_END + 1]
    train_valid = train_y[~pd.isna(train_y)]
    if len(train_valid) == 0:
        raise ValueError(f"No valid {target} in train for {app_id}")

    last_valid = float(train_valid[-1])
    history_slice = y_full[TRAIN_END - max(TARGET_LAGS) + 1: TRAIN_END + 1]
    y_history = []
    for v in history_slice:
        if pd.isna(v):
            y_history.append(last_valid)
        else:
            y_history.append(float(v))
            last_valid = float(v)

    return {"x_base_raw": x_base, "y_history": y_history}


# === COPY FROM notebook 07 cell 9 (_recursive_forecast, _run_scenario_mc_trajectory) ===
def _recursive_forecast(model, scaler, x_base_raw: np.ndarray,
                        base_feature_names: list, ops_changes: dict,
                        y_history: list, h_demo: int = H_DEMO,
                        lags=TARGET_LAGS, noise_rng=None,
                        noise_scale_per_op: dict = None,
                        residual_std: float = 0.0) -> np.ndarray:
    """Recursive multi-step forecast with optional Gaussian noise on ops features."""
    max_lag = max(lags)
    assert len(y_history) >= max_lag

    # Build counterfactual x: apply ops_changes
    x_counter = x_base_raw.copy()
    for feature_name, new_val in ops_changes.items():
        if feature_name in base_feature_names:
            idx = base_feature_names.index(feature_name)
            x_counter[idx] = new_val

    y_pred = np.zeros(h_demo, dtype=float)
    y_buffer = list(y_history)

    for h in range(1, h_demo + 1):
        x_step = x_counter.copy()
        # Add noise on ops features (independent per step)
        if noise_rng is not None and noise_scale_per_op is not None:
            for feature_name, sigma in noise_scale_per_op.items():
                if feature_name in base_feature_names and sigma > 0:
                    idx = base_feature_names.index(feature_name)
                    x_step[idx] = x_step[idx] + noise_rng.normal(0.0, sigma)

        # Compose: base + lag features
        lag_vals = np.array([y_buffer[-l] for l in lags], dtype=float)
        x_full_raw = np.concatenate([x_step, lag_vals])
        x_full_sc = scaler.transform(x_full_raw.reshape(1, -1))
        y_pred[h - 1] = float(model.predict(x_full_sc)[0])
        # Residual noise injection
        if noise_rng is not None and residual_std > 0:
            y_pred[h - 1] += noise_rng.normal(0.0, residual_std)
        y_buffer.append(y_pred[h - 1])

    return y_pred


def _run_scenario_mc_trajectory(model, scaler, x_base_raw: np.ndarray,
                                base_feature_names: list,
                                ops_changes: dict, ops_features: list,
                                X_train_raw: np.ndarray,
                                y_history: list,
                                X_train_sc=None, y_train=None,
                                n_mc: int = N_SIMULATIONS,
                                random_state: int = SEED):
    """Monte Carlo trajectory: n_mc recursive forecasts with Gaussian noise on ops."""
    rng = np.random.RandomState(random_state)

    # Per-feature noise std: NOISE_PCT * std on raw train (only base features)
    n_base = len(base_feature_names)
    base_train_stds = np.std(X_train_raw[:, :n_base], axis=0)
    noise_scale_per_op = {}
    for f in ops_features:
        if f in base_feature_names:
            idx = base_feature_names.index(f)
            noise_scale_per_op[f] = float(base_train_stds[idx] * NOISE_PCT)

    # Compute residual std for predictive noise injection (train residuals)
    residual_std = 0.0
    if X_train_sc is not None and y_train is not None:
        try:
            y_pred_train = np.asarray(model.predict(X_train_sc), dtype=float)
            residuals = np.asarray(y_train, dtype=float) - y_pred_train
            residual_std = float(np.std(residuals))
        except Exception:
            residual_std = 0.0

    trajectories = np.zeros((n_mc, H_DEMO), dtype=float)
    for k in range(n_mc):
        traj = _recursive_forecast(
            model=model, scaler=scaler, x_base_raw=x_base_raw,
            base_feature_names=base_feature_names, ops_changes=ops_changes,
            y_history=y_history, h_demo=H_DEMO, lags=TARGET_LAGS,
            noise_rng=rng, noise_scale_per_op=noise_scale_per_op,
            residual_std=residual_std,
        )
        trajectories[k] = traj

    # Per-step CI via percentile, median trajectory
    alpha = (1 - CI_LEVEL) / 2
    median_traj = np.median(trajectories, axis=0)
    ci_lo_traj = np.percentile(trajectories, 100 * alpha, axis=0)
    ci_hi_traj = np.percentile(trajectories, 100 * (1 - alpha), axis=0)

    # Final-day h=14: BCa CI
    final_samples = trajectories[:, -1]
    ci_lo_h14, ci_hi_h14 = bca_bootstrap_ci(
        final_samples, alpha=1 - CI_LEVEL,
        n_bootstrap=N_BOOTSTRAP, seed=random_state,
    )
    median_h14 = float(np.median(final_samples))

    return {
        "trajectories": trajectories,
        "median_traj": median_traj,
        "ci_lo_traj": ci_lo_traj,
        "ci_hi_traj": ci_hi_traj,
        "median_h14": median_h14,
        "ci_lo_h14": float(ci_lo_h14),
        "ci_hi_h14": float(ci_hi_h14),
    }


def dump(app: str, target: str, scenario_name: str,
         n: int = N_SIMULATIONS) -> pd.DataFrame:
    """Воспроизводит notebook 07 cell 16 для одного (app, target, scenario)."""
    # 1. final_config — ключ "model", не "model_name"
    final_config = json.loads(
        (ROOT / "results/final_config.json").read_text(encoding="utf-8")
    )
    app_id = APP_IDS[app]

    # Winner-модель — через единый registry (читает recursive_summary.csv,
    # та же логика что и в notebook 07 cell 16, согласовано с tab5_1).
    summary_path = ROOT / "results/forecast_validation_recursive_summary.csv"
    model_name = resolve_winner_model(app, target, summary_path)

    # 2. Load data + train slice
    df = load_dataset()
    train_df = (
        df[df.app_id == app_id]
        .sort_values("date")
        .iloc[: TRAIN_END + 1]
        .copy()
    )

    # 3. Train model (per notebook 07 cell 7)
    trained = _train_scenario_model(app_id, target, model_name, train_df, final_config)

    # 4. Baseline state (per notebook 07 cell 8)
    base_features = trained["base_features"]
    medians = final_config[app][target].get("feature_train_medians", {})
    baseline = _get_baseline_state(df, app_id, target, base_features, medians)

    # 5. Scenario changes — из summary CSV
    summary = pd.read_csv(ROOT / "results/scenario_analysis_summary.csv")
    m = summary[
        (summary["app"] == app)
        & (summary["target"] == target)
        & (summary["scenario_name"] == scenario_name)
    ]
    if m.empty:
        sys.exit(
            f"Не найдено (app={app}, target={target}, scenario={scenario_name}) "
            f"в summary"
        )
    changes = json.loads(m.iloc[0]["scenario_changes_json"])
    summary_delta = float(m.iloc[0]["delta_pct_vs_baseline"])
    print(f"Model for ({app}, {target}): {model_name}")
    print(f"Loaded changes for {scenario_name}: {changes}")
    print(f"Expected median delta_pct (per summary): {summary_delta:+.2f}%")

    # 6. Run baseline (changes={}) — SEED=42, как в notebook 07.
    baseline_trajectories = _run_scenario_mc_trajectory(
        model=trained["model"],
        scaler=trained.get("scaler"),
        x_base_raw=baseline["x_base_raw"],
        base_feature_names=base_features,
        ops_changes={},
        ops_features=trained["ops_features"],
        X_train_raw=trained["X_train_raw"],
        y_history=baseline["y_history"],
        X_train_sc=trained.get("X_train_sc"),
        y_train=trained.get("y_train"),
        n_mc=n,
        random_state=SEED,
    )
    baseline_h14 = baseline_trajectories["trajectories"][:, -1]
    baseline_median = float(np.median(baseline_h14))

    # 7. Run scenario — тот же SEED=42 (notebook 07 не использует разные seeds).
    scenario_trajectories = _run_scenario_mc_trajectory(
        model=trained["model"],
        scaler=trained.get("scaler"),
        x_base_raw=baseline["x_base_raw"],
        base_feature_names=base_features,
        ops_changes=changes,
        ops_features=trained["ops_features"],
        X_train_raw=trained["X_train_raw"],
        y_history=baseline["y_history"],
        X_train_sc=trained.get("X_train_sc"),
        y_train=trained.get("y_train"),
        n_mc=n,
        random_state=SEED,
    )
    scenario_h14 = scenario_trajectories["trajectories"][:, -1]

    # 8. Per-simulation delta_pct (по h14 финальным значениям)
    out = pd.DataFrame({
        "sim_idx": np.arange(n),
        "baseline_h14": baseline_h14,
        "scenario_h14": scenario_h14,
        "delta_pct_vs_baseline": (scenario_h14 - baseline_median)
        / abs(baseline_median) * 100.0,
    })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="f2")
    p.add_argument("--target", default="dau")
    p.add_argument("--scenario", default="cross_product_mimic")
    p.add_argument("--n", type=int, default=N_SIMULATIONS)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.out is None:
        args.out = (
            ROOT / f"results/mc_{args.app}_{args.target}_{args.scenario}.csv"
        )

    df_out = dump(args.app, args.target, args.scenario, n=args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    median_delta = df_out["delta_pct_vs_baseline"].median()
    print(f"OK wrote {args.out}: {len(df_out)} sims, "
          f"median delta = {median_delta:+.2f}%")

    # Sanity check
    summary = pd.read_csv(ROOT / "results/scenario_analysis_summary.csv")
    m = summary[
        (summary["app"] == args.app)
        & (summary["target"] == args.target)
        & (summary["scenario_name"] == args.scenario)
    ]
    if not m.empty:
        s_delta = float(m.iloc[0]["delta_pct_vs_baseline"])
        diff = abs(median_delta - s_delta)
        print(
            f"Summary delta = {s_delta:+.2f}%, "
            f"dump delta = {median_delta:+.2f}%, diff = {diff:.2f}pp"
        )
        if diff > 1.0:
            print(
                f"WARNING: расходится >1pp — проверь копию helper-функций "
                f"из notebook 07",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
