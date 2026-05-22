"""Smoke-тесты функций отрисовки: возвращают Figure без ошибок."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from app.streamlit.lib.plots import (
    plot_nan_heatmap,
    plot_forecast_with_ci,
    plot_delta_histogram,
    plot_verdict_decomposition,
)


def test_plot_nan_heatmap_returns_figure(sample_df):
    fig = plot_nan_heatmap(sample_df)
    assert fig is not None
    assert hasattr(fig, "savefig")


def test_plot_forecast_with_ci_returns_figure():
    dates = pd.date_range("2024-01-01", periods=20)
    history = pd.DataFrame(
        {
            "date": dates[:14],
            "y_true": np.random.randn(14),
            "segment": ["history"] * 14,
        }
    )
    forecast = pd.DataFrame(
        {
            "date": dates[14:],
            "y_pred": np.random.randn(6),
            "ci_lo": np.random.randn(6) - 0.5,
            "ci_hi": np.random.randn(6) + 0.5,
            "segment": ["forecast_safe"] * 6,
        }
    )
    fig = plot_forecast_with_ci(
        history, forecast, h_safe_A=3, h_demo=6, target_label="stickiness"
    )
    assert fig is not None


def test_plot_delta_histogram_returns_figure():
    delta = np.random.randn(1000) * 10
    fig = plot_delta_histogram(delta, ci_lo=-5, ci_hi=8, median=2.5)
    assert fig is not None


def test_plot_verdict_decomposition_returns_figure():
    rows = pd.DataFrame(
        {
            "model": ["M1", "M2", "M3"],
            "days_develop": [91, 55, 57],
            "days_monitor": [10, 48, 46],
            "days_discontinue": [10, 7, 7],
        }
    )
    fig = plot_verdict_decomposition(rows)
    assert fig is not None
