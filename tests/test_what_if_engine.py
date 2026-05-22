"""Тесты WhatIfEngine для Streamlit-демо.

Файл будет наполняться в задачах 1.4–1.9 (baseline state, lag features,
MC-симуляция, WhatIfEngine.run() и invariants 2/3).
Пока — только smoke-тест фикстуры real_floodit_df.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.streamlit.lib.what_if_engine import (
    APP_ID_MAP,
    L,
    TRAIN_END_IDX,
    WhatIfEngine,
    WhatIfRequest,
    WhatIfResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "results" / "forecast_validation_recursive_summary.csv"
FINAL_CONFIG_PATH = PROJECT_ROOT / "results" / "final_config.json"


@pytest.fixture(scope="session")
def final_config():
    with open(FINAL_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def engine(real_floodit_df, final_config):
    return WhatIfEngine(real_floodit_df, SUMMARY_PATH, final_config)


def test_real_floodit_df_loads(real_floodit_df):
    assert real_floodit_df.shape == (222, 60)
    assert set(real_floodit_df["app_id"].unique()) == {"com.labpixies.flood", "com.google.flood2"}


def test_what_if_request_defaults():
    req = WhatIfRequest(app="f1", target="stickiness", changes={"crash_rate": 0.02})
    assert req.n_simulations == 1000
    assert req.seed == 42
    assert req.changes == {"crash_rate": 0.02}


def test_what_if_request_is_frozen():
    req = WhatIfRequest(app="f1", target="stickiness", changes={})
    with pytest.raises((AttributeError, Exception)):
        req.seed = 100  # type: ignore[misc]


def test_what_if_result_construct_minimal():
    result = WhatIfResult(
        expected_delta_pct=7.65,
        ci95_lo_pct=6.41,
        ci95_hi_pct=9.38,
        ci95_abs_lo=0.057628,
        ci95_abs_hi=0.059235,
        p_success=0.658,
        baseline_h14=0.054156,
        median_scenario=0.058298,
        scenario_final=np.zeros(1000),
        delta_pct_distribution=np.zeros(1000),
        model_name="Ridge",
        extrapolation_warning=False,
    )
    assert result.expected_delta_pct == 7.65
    assert result.scenario_final.shape == (1000,)


def test_invariant_2a_baseline_row_is_iloc_86(engine, real_floodit_df, final_config):
    """x_base_raw должен быть строкой df_app.iloc[TRAIN_END_IDX=86]."""
    base_features = final_config["f1"]["stickiness"]["features"]
    state = engine._extract_baseline_state("f1", "stickiness", base_features)
    df_app = real_floodit_df[real_floodit_df["app_id"] == APP_ID_MAP["f1"]].sort_values("date").reset_index(drop=True)
    expected = df_app.iloc[TRAIN_END_IDX][base_features].to_numpy(dtype=float)
    # после возможного forward-fill значения могут отличаться только в NaN-позициях
    # этот тест проверяет ИНДЕКС, а не точные значения
    assert state.x_base_raw.shape == expected.shape
    # проверка не-NaN значений совпадают
    mask = ~pd.isna(expected)
    np.testing.assert_allclose(state.x_base_raw[mask], expected[mask], rtol=1e-9)


def test_invariant_2b_y_history_length_is_L(engine, real_floodit_df, final_config):
    """y_history длиной L=7, последние значения таргета из train_slice."""
    base_features = final_config["f1"]["stickiness"]["features"]
    state = engine._extract_baseline_state("f1", "stickiness", base_features)
    assert state.y_history.shape == (L,)
    assert L == 7
    df_app = real_floodit_df[real_floodit_df["app_id"] == APP_ID_MAP["f1"]].sort_values("date").reset_index(drop=True)
    # после forward-fill — последние L значений до TRAIN_END_IDX (включительно)
    expected_slice = df_app.iloc[TRAIN_END_IDX - L + 1 : TRAIN_END_IDX + 1]["stickiness"].ffill().bfill().to_numpy()
    np.testing.assert_allclose(state.y_history, expected_slice, rtol=1e-9)


def test_baseline_uses_ffill_for_nan(engine, real_floodit_df, final_config):
    """NaN в base_features заменяется forward-fill (затем train median fallback)."""
    base_features = final_config["f1"]["stickiness"]["features"]
    state = engine._extract_baseline_state("f1", "stickiness", base_features)
    assert not np.any(np.isnan(state.x_base_raw)), "x_base_raw must not contain NaN after ffill+fallback"


def test_build_lag_features_adds_columns(engine, real_floodit_df):
    df_app = real_floodit_df[real_floodit_df["app_id"] == APP_ID_MAP["f1"]].sort_values("date").reset_index(drop=True)
    train_slice = df_app.iloc[: TRAIN_END_IDX + 1].copy()
    with_lags = engine._add_lag_features(train_slice, "stickiness")
    assert "stickiness_lag1" in with_lags.columns
    assert "stickiness_lag7" in with_lags.columns


def test_train_winner_returns_fitted_model_and_residual_std(engine, final_config):
    base_features = final_config["f1"]["stickiness"]["features"]
    state = engine._extract_baseline_state("f1", "stickiness", base_features)
    fitted = engine._train_winner_model("f1", "stickiness", "Ridge", state, base_features)
    assert hasattr(fitted.model, "predict")
    assert fitted.residual_std >= 0.0
    assert isinstance(fitted.residual_std, float)


def test_mc_trajectory_shape_and_finite(engine, final_config):
    """Одна траектория — массив длиной H_DEMO=14, без NaN/inf."""
    base_features = final_config["f1"]["stickiness"]["features"]
    state = engine._extract_baseline_state("f1", "stickiness", base_features)
    fitted = engine._train_winner_model("f1", "stickiness", "Ridge", state, base_features)
    rng = np.random.RandomState(42)
    noise_scales = engine._compute_noise_scales(fitted, base_features, "f1", "stickiness")
    traj = engine._simulate_one_trajectory(
        x_base=state.x_base_raw,
        y_history=state.y_history.copy(),
        fitted=fitted,
        base_features=base_features,
        noise_scales=noise_scales,
        rng=rng,
        target="stickiness",
    )
    assert traj.shape == (14,)
    assert np.all(np.isfinite(traj))


def test_noise_scales_only_operational_nonzero(engine, final_config):
    """Возвращаем list[(idx, sigma)] только для ops features с sigma>0."""
    base_features = final_config["f1"]["stickiness"]["features"]
    state = engine._extract_baseline_state("f1", "stickiness", base_features)
    fitted = engine._train_winner_model("f1", "stickiness", "Ridge", state, base_features)
    scales = engine._compute_noise_scales(fitted, base_features, "f1", "stickiness")
    # Все возвращённые индексы должны соответствовать operational features
    ops = {"crash_rate", "onboarding_completion_rate", "median_session_duration"}
    for idx, sigma in scales:
        assert sigma > 0.0
        assert base_features[idx] in ops, (
            f"non-operational feature {base_features[idx]} got nonzero noise"
        )


def test_run_baseline_no_changes_returns_zero_delta(engine):
    """Без changes scenario_final ≈ baseline_final → delta_pct ≈ 0."""
    req = WhatIfRequest(app="f1", target="stickiness", changes={})
    result = engine.run(req)
    # baseline vs baseline → median(scenario_final) очень близко к baseline_h14
    assert abs(result.expected_delta_pct) < 5.0, (
        f"baseline-vs-baseline delta should be ~0, got {result.expected_delta_pct}"
    )
    assert result.model_name == "Ridge"
    assert result.scenario_final.shape == (1000,)
    assert result.delta_pct_distribution.shape == (1000,)


def test_invariant_3_determinism(engine):
    """Два прогона с seed=42 → идентичные метрики до 4 знаков."""
    req = WhatIfRequest(app="f1", target="stickiness", changes={"crash_rate": 0.02})
    r1 = engine.run(req)
    r2 = engine.run(req)
    np.testing.assert_allclose([r1.expected_delta_pct], [r2.expected_delta_pct], atol=1e-4)
    np.testing.assert_allclose([r1.p_success], [r2.p_success], atol=1e-4)
    np.testing.assert_allclose([r1.ci95_lo_pct], [r2.ci95_lo_pct], atol=1e-4)
    np.testing.assert_allclose([r1.ci95_hi_pct], [r2.ci95_hi_pct], atol=1e-4)
    np.testing.assert_allclose(r1.scenario_final, r2.scenario_final, atol=1e-9)
