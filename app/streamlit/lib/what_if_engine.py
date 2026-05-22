"""WhatIfEngine — единственный live-компонент Streamlit-демо.

Повторяет траекторную логику notebook 07: MC-рекурсивный прогноз
на h=14 с гауссовским шумом на operational features и residual noise.
Источник winner-модели — winner_registry.resolve_winner_model.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import dataclass as _dc
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.kernel_regression import LocalLinearRegressor, NadarayaWatsonRegressor

# Константы — идентичны notebook 07
TRAIN_END_IDX: int = 86
H_DEMO: int = 14
N_MC: int = 1000
SEED: int = 42
TARGET_LAGS: tuple[int, ...] = (1, 7)
L: int = max(TARGET_LAGS)  # 7
NOISE_PCT: float = 0.05

OPERATIONAL_FEATURES: frozenset[str] = frozenset({
    "crash_rate",
    "onboarding_completion_rate",
    "median_session_duration",
})

APP_ID_MAP: dict[str, str] = {
    "f1": "com.labpixies.flood",
    "f2": "com.google.flood2",
}


@dataclass(frozen=True)
class WhatIfRequest:
    """Запрос к WhatIfEngine.run()."""
    app: Literal["f1", "f2"]
    target: Literal["stickiness", "dau", "retention_d7"]
    changes: dict[str, float]
    n_simulations: int = N_MC
    seed: int = SEED


@dataclass(frozen=True)
class WhatIfResult:
    """Результат WhatIfEngine.run()."""
    expected_delta_pct: float
    ci95_lo_pct: float
    ci95_hi_pct: float
    ci95_abs_lo: float
    ci95_abs_hi: float
    p_success: float
    baseline_h14: float
    median_scenario: float
    scenario_final: np.ndarray
    delta_pct_distribution: np.ndarray
    model_name: str
    extrapolation_warning: bool


@_dc(frozen=True)
class _BaselineState:
    """Baseline state на день T=TRAIN_END_IDX=86 (Invariant 2)."""
    x_base_raw: np.ndarray         # shape (n_base_features,)
    y_history: np.ndarray          # shape (L,)
    train_slice: pd.DataFrame      # df_app.iloc[:TRAIN_END_IDX+1]
    feature_names: list[str]


@_dc(frozen=True)
class _FittedWinner:
    """Обученная winner-модель + std остатков и порядок признаков.

    Включает StandardScaler — notebook 07 обучает Ridge/RF/NW на ScalerX,
    поэтому inference тоже должен идти через scaler.transform().
    """
    model: object               # sklearn-compatible
    residual_std: float
    feature_order: list[str]    # base_features + lag columns
    scaler: object              # StandardScaler, fit на X_raw
    X_train_raw: np.ndarray     # raw train (для noise_scale_per_op)


def _make_model(model_name: str, bandwidth: float = 1.0):
    """Фабрика моделей по имени из 06b winner.

    Parameters
    ----------
    model_name : str
        Имя winner-модели из forecast_validation_recursive_summary.csv.
    bandwidth : float, optional
        Ширина окна для kernel-моделей (NW/LocalLinear). По умолчанию 1.0.
        Для kernel-моделей рекомендуется прокидывать `bandwidth_used` из
        forecast_validation_recursive_summary.csv (см. `_resolve_bandwidth`).
    """
    if model_name == "Ridge":
        return Ridge(alpha=1.0, random_state=SEED)
    if model_name == "RandomForest":
        return RandomForestRegressor(n_estimators=100, random_state=SEED)
    if model_name == "NadarayaWatson":
        return NadarayaWatsonRegressor(kernel="gaussian", bandwidth=bandwidth)
    if model_name == "LocalLinear":
        return LocalLinearRegressor(kernel="gaussian", bandwidth=bandwidth)
    if model_name == "kNN":
        return KNeighborsRegressor(n_neighbors=5)
    raise ValueError(f"Unsupported model: {model_name}")


def _resolve_bandwidth(
    model_name: str, app: str, target: str, summary_path
) -> float:
    """Достаёт `bandwidth_used` из forecast_validation_recursive_summary.csv.

    Для не-kernel моделей (Ridge/RandomForest/XGBoost/kNN) возвращает 1.0
    (значение игнорируется фабрикой). Для kernel-моделей (NW/LocalLinear)
    читает столбец `bandwidth_used` из строки (app, target, model).

    Если bandwidth не найден (NaN/отсутствует) — возвращает 1.0 как fallback.
    """
    if model_name not in {"NadarayaWatson", "LocalLinear"}:
        return 1.0
    df = pd.read_csv(summary_path)
    row = df[
        (df["app"] == app)
        & (df["target"] == target)
        & (df["model"] == model_name)
    ]
    if len(row) == 0:
        return 1.0
    val = row.iloc[0].get("bandwidth_used", np.nan)
    if pd.isna(val):
        return 1.0
    return float(val)


class WhatIfEngine:
    """Адаптер для counterfactual sandbox.

    Skeleton; реальная логика добавляется в задачах 1.5–1.11.
    """

    def __init__(self, df: pd.DataFrame, summary_path, final_config: dict) -> None:
        self.df = df
        self.summary_path = summary_path
        self.final_config = final_config

    def _extract_baseline_state(
        self, app: str, target: str, base_features: list[str]
    ) -> _BaselineState:
        """Извлекает baseline state на день T=TRAIN_END_IDX=86.

        Применяет forward-fill для NaN в base_features; остаточные NaN
        заменяются на feature_train_medians из final_config.
        """
        app_id = APP_ID_MAP[app]
        df_app = (
            self.df[self.df["app_id"] == app_id]
            .sort_values("date")
            .reset_index(drop=True)
        )
        if len(df_app) <= TRAIN_END_IDX:
            raise ValueError(
                f"df_app for {app} has only {len(df_app)} rows; need > {TRAIN_END_IDX}"
            )

        train_slice = df_app.iloc[: TRAIN_END_IDX + 1].copy()

        # baseline row на день 86 (RAW, без ffill — notebook 07 _get_baseline_state
        # читает row_T = df_app.iloc[TRAIN_END] и fall-back'ит на медианы только
        # если значение NaN).
        baseline_row = train_slice.iloc[TRAIN_END_IDX]
        medians = self.final_config[app][target].get("feature_train_medians", {})
        x_base_raw = np.array([
            float(baseline_row[f])
            if (f in train_slice.columns and pd.notna(baseline_row.get(f, np.nan)))
            else float(medians.get(f, train_slice[f].median()))
            for f in base_features
        ], dtype=float)

        # y_history по notebook 07: history_slice = y_full[T-L+1:T+1] с заменой
        # NaN на last_valid (forward-fill во времени от последнего valid в train).
        y_full = train_slice[target].to_numpy(dtype=float)
        train_valid = y_full[~pd.isna(y_full)]
        if len(train_valid) == 0:
            raise ValueError(f"No valid {target} values in train for {app}")
        last_valid = float(train_valid[-1])
        history_slice = y_full[TRAIN_END_IDX - L + 1 : TRAIN_END_IDX + 1]
        y_history_list: list[float] = []
        for v in history_slice:
            if pd.isna(v):
                y_history_list.append(last_valid)
            else:
                y_history_list.append(float(v))
                last_valid = float(v)
        y_history = np.array(y_history_list, dtype=float)
        assert y_history.shape == (L,), (
            f"y_history shape {y_history.shape}, expected ({L},)"
        )

        return _BaselineState(
            x_base_raw=x_base_raw,
            y_history=y_history,
            train_slice=train_slice,
            feature_names=list(base_features),
        )

    def _add_lag_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Добавляет target_lag1..target_lagL колонки (ffill+bfill перед shift)."""
        out = df.copy()
        target_filled = out[target].ffill().bfill()
        for lag in TARGET_LAGS:
            out[f"{target}_lag{lag}"] = target_filled.shift(lag)
        return out

    def _train_winner_model(
        self,
        app: str,
        target: str,
        model_name: str,
        state: "_BaselineState",
        base_features: list[str],
        bandwidth: float = 1.0,
    ) -> "_FittedWinner":
        """Обучает winner-модель на train_slice + лаги; считает residual_std.

        Воспроизводит notebook 07 _build_extended_xy / _train_scenario_model:
          1. ffill+bfill таргета, добавление target_lag1/_lag7.
          2. Фильтр df[target].notna() (как в build_xy).
          3. Fill базовых фич медианами из final_config (train_medians).
          4. Drop остаточных NaN в X (любая колонка).
          5. StandardScaler.fit(X_raw); model.fit(X_sc, y).
        """
        df_with_lags = self._add_lag_features(state.train_slice, target)
        lag_cols = [f"{target}_lag{l}" for l in TARGET_LAGS]
        feature_order = list(base_features) + lag_cols

        # Фильтр по target.notna() (build_xy contract)
        df_with_lags = df_with_lags[df_with_lags[target].notna()].copy()

        # Fill базовых фич train_medians из final_config (как build_xy(..., train_medians=...))
        medians = self.final_config[app][target].get("feature_train_medians", {})
        for f in base_features:
            fill_val = float(medians.get(f, df_with_lags[f].median()))
            df_with_lags[f] = df_with_lags[f].fillna(fill_val)

        # Drop строк, где остались NaN в X (например, первые max(LAGS) строк по лагам)
        df_with_lags = df_with_lags.dropna(subset=feature_order)

        X = df_with_lags[feature_order].to_numpy(dtype=float)
        y = df_with_lags[target].to_numpy(dtype=float)

        # Notebook 07 fits a StandardScaler on X_raw перед обучением (см.
        # _train_scenario_model). Без этого Ridge даёт совершенно другой fit,
        # из-за чего baseline_h14 расходится на ~25% (см. диагностику задачи 1.10).
        scaler = StandardScaler().fit(X)
        X_sc = scaler.transform(X)

        model = _make_model(model_name, bandwidth=bandwidth)
        model.fit(X_sc, y)

        residuals = y - model.predict(X_sc)
        residual_std = float(np.std(residuals, ddof=0))

        return _FittedWinner(
            model=model,
            residual_std=residual_std,
            feature_order=feature_order,
            scaler=scaler,
            X_train_raw=X,
        )

    def _compute_noise_scales(
        self,
        fitted: "_FittedWinner",
        base_features: list[str],
        app: str,
        target: str,
    ) -> list[tuple[int, float]]:
        """Список (index_in_base_features, sigma) ТОЛЬКО для ops с sigma>0.

        Порядок повторяет порядок ops_features из final_config — это критично,
        потому что notebook 07 итерирует `for f, sigma in noise_scale_per_op.items()`,
        и dict insertion order совпадает с порядком ops_features в _run_scenario_mc_trajectory.
        """
        n_base = len(base_features)
        base_train_stds = np.std(fitted.X_train_raw[:, :n_base], axis=0)
        ops_features = self.final_config[app][target].get(
            "operational_features", list(OPERATIONAL_FEATURES)
        )
        seq: list[tuple[int, float]] = []
        for f in ops_features:
            if f in base_features:
                idx = base_features.index(f)
                sigma = float(base_train_stds[idx] * NOISE_PCT)
                if sigma > 0:
                    seq.append((idx, sigma))
        return seq

    def _simulate_one_trajectory(
        self,
        x_base: np.ndarray,
        y_history: np.ndarray,
        fitted: "_FittedWinner",
        base_features: list[str],
        noise_scales: list[tuple[int, float]],
        rng: np.random.RandomState,
        target: str,
    ) -> np.ndarray:
        """Одна MC-траектория длиной H_DEMO с гауссовским шумом на ops features
        и residual noise на каждом шаге.

        Возвращает массив прогнозов таргета shape (H_DEMO,).
        """
        traj = np.zeros(H_DEMO, dtype=float)
        y_hist = list(y_history)  # mutable, длиной L
        x_cur = x_base.copy()

        for h in range(H_DEMO):
            # Notebook 07 итерирует только по ops_features (порядок из
            # final_config) и каждый draws ОДИН rng.normal(0, sigma).
            # Любое другое количество/порядок rng-draws сдвинет MC-поток.
            x_noisy = x_cur.copy()
            for idx, sigma in noise_scales:
                x_noisy[idx] = x_noisy[idx] + rng.normal(0.0, sigma)

            # построить лаги: lag1 = y_hist[-1], lag7 = y_hist[-7]
            lag_vals = [y_hist[-lag] for lag in TARGET_LAGS]
            x_with_lags = np.concatenate([x_noisy, np.asarray(lag_vals, dtype=float)])

            # notebook 07 предсказывает на масштабированных признаках
            x_scaled = fitted.scaler.transform(x_with_lags.reshape(1, -1))
            y_pred = float(fitted.model.predict(x_scaled)[0])

            # residual noise
            if fitted.residual_std > 0:
                y_pred += rng.normal(0.0, fitted.residual_std)

            traj[h] = y_pred
            # обновить историю: добавить новый прогноз, удалить самый старый
            y_hist.append(y_pred)
            y_hist.pop(0)

        return traj

    def run(self, request: WhatIfRequest) -> WhatIfResult:
        from app.streamlit.lib.winner_registry import resolve_winner_model
        from src.scenario_analysis import ScenarioAnalyzer

        model_name = resolve_winner_model(request.app, request.target, self.summary_path)
        bandwidth = _resolve_bandwidth(
            model_name, request.app, request.target, self.summary_path
        )
        base_features = list(self.final_config[request.app][request.target]["features"])

        state = self._extract_baseline_state(request.app, request.target, base_features)
        fitted = self._train_winner_model(
            request.app, request.target, model_name, state, base_features,
            bandwidth=bandwidth,
        )
        noise_scales = self._compute_noise_scales(
            fitted, base_features, request.app, request.target
        )

        # применить changes только для фичей из OPERATIONAL_FEATURES ∩ base_features
        x_scenario = state.x_base_raw.copy()
        valid_keys = (set(request.changes) & OPERATIONAL_FEATURES) & set(base_features)
        for fname in valid_keys:
            idx = base_features.index(fname)
            x_scenario[idx] = float(request.changes[fname])

        # В notebook 07 _run_scenario_mc_trajectory ВЫЗЫВАЕТСЯ ОТДЕЛЬНО для каждого
        # сценария (включая baseline) и каждый раз создаёт свой np.random.RandomState(42).
        # Поэтому baseline и scenario получают идентичные последовательности шумов
        # на тех индексах, где x_base совпадает.
        rng_baseline = np.random.RandomState(request.seed)
        rng_scenario = np.random.RandomState(request.seed)

        # baseline trajectories
        baseline_finals = np.zeros(request.n_simulations, dtype=float)
        for i in range(request.n_simulations):
            traj = self._simulate_one_trajectory(
                x_base=state.x_base_raw,
                y_history=state.y_history.copy(),
                fitted=fitted,
                base_features=base_features,
                noise_scales=noise_scales,
                rng=rng_baseline,
                target=request.target,
            )
            baseline_finals[i] = traj[-1]

        # scenario trajectories
        scenario_finals = np.zeros(request.n_simulations, dtype=float)
        for i in range(request.n_simulations):
            traj = self._simulate_one_trajectory(
                x_base=x_scenario,
                y_history=state.y_history.copy(),
                fitted=fitted,
                base_features=base_features,
                noise_scales=noise_scales,
                rng=rng_scenario,
                target=request.target,
            )
            scenario_finals[i] = traj[-1]

        # метрики (формулы notebook 07)
        baseline_h14 = float(np.median(baseline_finals))
        median_scenario = float(np.median(scenario_finals))
        expected_delta_pct = (median_scenario - baseline_h14) / abs(baseline_h14) * 100.0
        p_success = float((scenario_finals > baseline_h14).mean())

        # bca_bootstrap_ci — метод ScenarioAnalyzer, не использует self,
        # поэтому вызываем как unbound через явный None (см. src/scenario_analysis.py:216).
        ci_abs_lo, ci_abs_hi = ScenarioAnalyzer.bca_bootstrap_ci(
            None, scenario_finals, alpha=0.05, seed=request.seed
        )
        ci95_lo_pct = (ci_abs_lo - baseline_h14) / abs(baseline_h14) * 100.0
        ci95_hi_pct = (ci_abs_hi - baseline_h14) / abs(baseline_h14) * 100.0

        delta_pct_distribution = (scenario_finals - baseline_h14) / abs(baseline_h14) * 100.0

        return WhatIfResult(
            expected_delta_pct=expected_delta_pct,
            ci95_lo_pct=ci95_lo_pct,
            ci95_hi_pct=ci95_hi_pct,
            ci95_abs_lo=float(ci_abs_lo),
            ci95_abs_hi=float(ci_abs_hi),
            p_success=p_success,
            baseline_h14=baseline_h14,
            median_scenario=median_scenario,
            scenario_final=scenario_finals,
            delta_pct_distribution=delta_pct_distribution,
            model_name=model_name,
            extrapolation_warning=abs(expected_delta_pct) > 100.0,
        )
