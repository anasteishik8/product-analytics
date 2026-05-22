"""
Общие фикстуры для тестов системы прогнозирования востребованности IT-продуктов.

Все фикстуры используют синтетические данные (20–30 строк),
реальный parquet в тестах НЕ загружается (медленно).
Seed=42 зафиксирован для воспроизводимости.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.kernel_regression import NadarayaWatsonRegressor

DEFAULT_SEED = 42
N_SAMPLES = 25
N_FEATURES = 8


@pytest.fixture(scope="session")
def rng() -> np.random.RandomState:
    """Генератор случайных чисел с seed=42."""
    return np.random.RandomState(DEFAULT_SEED)


@pytest.fixture(scope="session")
def sample_X(rng) -> np.ndarray:
    """Синтетический массив признаков (25, 8)."""
    return rng.randn(N_SAMPLES, N_FEATURES).astype(float)


@pytest.fixture(scope="session")
def sample_y(sample_X, rng) -> np.ndarray:
    """Синтетические целевые значения (25,) — нелинейная функция + шум."""
    y = np.sin(sample_X[:, 0]) + 0.5 * sample_X[:, 1] ** 2 + rng.randn(N_SAMPLES) * 0.1
    return y.astype(float)


@pytest.fixture(scope="session")
def sample_y_binary(sample_y) -> np.ndarray:
    """Бинарные метки (25,) — медиана как порог."""
    return (sample_y > np.median(sample_y)).astype(int)


@pytest.fixture(scope="session")
def feature_names() -> list[str]:
    """Имена признаков для синтетического датасета."""
    return [f"feature_{i}" for i in range(N_FEATURES)]


@pytest.fixture(scope="session")
def fitted_nw_model(sample_X, sample_y) -> NadarayaWatsonRegressor:
    """Обученный NadarayaWatsonRegressor на синтетических данных."""
    model = NadarayaWatsonRegressor(kernel="gaussian", bandwidth=1.0)
    model.fit(sample_X, sample_y)
    return model


@pytest.fixture(scope="session")
def sample_df(rng) -> pd.DataFrame:
    """Синтетический DataFrame с теми же колонками, что floodit_final.parquet.

    25 строк × 60 колонок. Используется для тестов feature_engineering.
    Содержит два приложения (10 строк каждого + 5 общих).
    """
    n = N_SAMPLES
    dates = pd.date_range("2018-06-12", periods=n, freq="D")
    app_ids = ["com.labpixies.flood"] * 15 + ["com.google.flood2"] * 10

    df = pd.DataFrame({
        "date": dates,
        "app_id": app_ids,
        "category": ["Puzzle"] * n,
        "weekday": np.tile(range(7), n // 7 + 1)[:n],
        "has_store_data": rng.randint(0, 2, n),
        # Engagement
        "dau": rng.uniform(100, 900, n),
        "mau": rng.uniform(500, 3000, n),
        "wau": rng.uniform(200, 1000, n),
        "stickiness": rng.uniform(0.03, 1.0, n),
        "retention_d1": rng.uniform(0.1, 0.6, n),
        "retention_d7": rng.uniform(0.02, 0.25, n),
        "retention_d30": rng.uniform(0.01, 0.15, n),
        "total_sessions": rng.uniform(500, 5000, n),
        "median_session_duration": rng.uniform(60, 600, n),
        "avg_session_duration": rng.uniform(60, 600, n),
        "engagement_score": rng.uniform(0.1, 1.0, n),
        # Acquisition
        "daily_installs": rng.randint(10, 100, n).astype(float),
        "daily_uninstalls": rng.randint(5, 80, n).astype(float),
        "net_installs": rng.randint(-50, 50, n).astype(float),
        "install_growth_rate": rng.uniform(-0.5, 0.5, n),
        # Quality
        "crash_rate": rng.uniform(0.001, 0.7, n),
        "onboarding_completion_rate": rng.uniform(0.2, 0.9, n),
        # Temporal
        "peak_hour": rng.randint(0, 24, n).astype(float),
        "crash_rate_ma7": rng.uniform(0.001, 0.5, n),
        "avg_session_duration_ma7": rng.uniform(60, 600, n),
        "dau_growth_rate": rng.uniform(-0.3, 0.3, n),
        "dau_growth_rate_7d": rng.uniform(-0.3, 0.3, n),
        "retention_d1_change": rng.uniform(-0.1, 0.1, n),
        "retention_trend": rng.uniform(-0.05, 0.05, n),
        "seasonality_change": rng.uniform(-0.2, 0.2, n),
        "downloads_index": rng.uniform(0.5, 2.0, n),
        "downloads_index_ma30": rng.uniform(0.5, 2.0, n),
        # Store
        "rating": np.where(np.arange(n) < 15, rng.uniform(3.5, 5.0, n), np.nan),
        "reviews_count": np.where(np.arange(n) < 15, rng.uniform(100, 2000, n), np.nan),
        "positive": np.where(np.arange(n) < 15, rng.uniform(50, 1500, n), np.nan),
        "negative": np.where(np.arange(n) < 15, rng.uniform(10, 200, n), np.nan),
        "positive_ratio": np.where(np.arange(n) < 15, rng.uniform(0.6, 0.99, n), np.nan),
        "sentiment_score": np.where(np.arange(n) < 15, rng.uniform(0.3, 0.9, n), np.nan),
        "reviews_30d": np.where(np.arange(n) < 15, rng.uniform(5, 50, n), np.nan),
        "positive_30d": np.where(np.arange(n) < 15, rng.uniform(3, 40, n), np.nan),
        "negative_30d": np.where(np.arange(n) < 15, rng.uniform(1, 10, n), np.nan),
        "sentiment_30d": np.where(np.arange(n) < 15, rng.uniform(0.3, 0.9, n), np.nan),
        "rating_ma30": np.where(np.arange(n) < 15, rng.uniform(3.5, 5.0, n), np.nan),
        "positive_ratio_ma30": np.where(np.arange(n) < 15, rng.uniform(0.6, 0.99, n), np.nan),
        "rating_volatility_7d": rng.uniform(0.0, 0.1, n),
        "rating_trend": rng.uniform(-0.05, 0.05, n),
        "negative_change": rng.uniform(-0.1, 0.1, n),
        # Competitive
        "competitor_count": rng.randint(5000, 25000, n).astype(float),
        "category_rank": rng.randint(1, 500, n).astype(float),
        "competitor_count_change": rng.uniform(-100, 100, n),
        "competition_volatility_7d": rng.uniform(0, 0.05, n),
        "category_rank_ma14": rng.randint(1, 500, n).astype(float),
        "category_rank_ma30": rng.randint(1, 500, n).astype(float),
        "rank_volatility_7d": rng.uniform(0, 0.1, n),
        "rank_improvement": rng.uniform(-0.5, 0.5, n),
        # Market
        "market_trend": rng.uniform(0.5, 2.0, n),
        "media_coverage": rng.uniform(0, 1, n),
        "market_sentiment": rng.uniform(-1, 1, n),
        "media_coverage_ma7": rng.uniform(0, 1, n),
        "media_coverage_change": rng.uniform(-0.2, 0.2, n),
    })

    # Убеждаемся, что есть ровно 60 колонок
    assert len(df.columns) == 60, f"Ожидалось 60 колонок, получено {len(df.columns)}"
    return df


import pyarrow.parquet  # noqa: F401, E402  # ensure pyarrow available


@pytest.fixture(scope="session")
def real_floodit_df() -> pd.DataFrame:
    """Реальный датасет floodit_final.parquet, 222×60.

    Используется только в тестах WhatIfEngine и streamlit consistency.
    Большинство юнит-тестов используют sample_df (синтетика).
    """
    from pathlib import Path
    parquet_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "floodit_final.parquet"
    df = pd.read_parquet(parquet_path)
    assert df.shape == (222, 60), f"expected (222, 60), got {df.shape}"
    return df
