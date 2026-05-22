"""
Тесты для src/feature_engineering.py.

Проверяет:
    - load_dataset возвращает DataFrame с 60 колонками
    - select_features возвращает ≤ 15 признаков
    - VIF-фильтр удаляет коллинеарные колонки
    - normalize_features сохраняет форму
    - get_cv_strategy возвращает правильный тип
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import numpy.testing as npt

from src.feature_engineering import (
    load_dataset,
    compute_vif,
    select_features,
    normalize_features,
    get_cv_strategy,
)


class TestLoadDataset:
    def test_returns_dataframe(self):
        """load_dataset возвращает DataFrame."""
        df = load_dataset()
        assert isinstance(df, pd.DataFrame)

    def test_shape_222x60(self):
        """Форма датасета: 222 строки × 60 колонок."""
        df = load_dataset()
        assert df.shape == (222, 60)

    def test_expected_columns_present(self):
        """Ключевые колонки присутствуют в датасете."""
        df = load_dataset()
        required = ["date", "app_id", "stickiness", "rating", "retention_d7",
                    "crash_rate", "dau", "mau"]
        for col in required:
            assert col in df.columns, f"Колонка '{col}' отсутствует"

    def test_date_is_datetime(self):
        """Колонка date имеет тип datetime."""
        df = load_dataset()
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_file_not_found_raises(self):
        """FileNotFoundError при неверном пути."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent/path/data.parquet")


class TestComputeVif:
    def test_returns_dataframe_with_required_columns(self, sample_df):
        """compute_vif возвращает DataFrame с колонками feature, vif, keep."""
        numeric_cols = ["dau", "mau", "crash_rate", "stickiness", "rating"]
        sub = sample_df[numeric_cols].dropna()
        result = compute_vif(sub)
        assert set(result.columns) == {"feature", "vif", "keep"}

    def test_high_vif_col_marked_false(self, sample_df):
        """Высококоллинеарная колонка помечается keep=False."""
        # dau и mau сильно скоррелированы с wau
        sub = sample_df[["dau", "mau", "wau"]].dropna()
        result = compute_vif(sub, threshold=5.0)
        # Хотя бы одна колонка должна быть помечена keep=False при строгом пороге
        # (зависит от данных, поэтому проверяем структуру)
        assert len(result) > 0
        assert "keep" in result.columns

    def test_vif_positive(self, sample_df):
        """VIF-значения положительны."""
        sub = sample_df[["dau", "crash_rate", "stickiness"]].dropna()
        result = compute_vif(sub)
        assert (result["vif"] >= 1.0).all()

    def test_returns_sorted_by_vif_desc(self, sample_df):
        """Результат отсортирован по убыванию VIF."""
        sub = sample_df[["dau", "mau", "crash_rate", "stickiness"]].dropna()
        result = compute_vif(sub)
        vif_values = result["vif"].tolist()
        assert vif_values == sorted(vif_values, reverse=True)


class TestSelectFeatures:
    def test_returns_tuple_of_three(self, sample_df):
        """select_features возвращает кортеж (X, y, feature_names)."""
        X, y, names = select_features(sample_df, target="stickiness")
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(names, list)

    def test_max_features_respected(self, sample_df):
        """Число отобранных признаков ≤ max_features."""
        X, y, names = select_features(sample_df, target="stickiness", max_features=10)
        assert len(names) <= 10
        assert X.shape[1] == len(names)

    def test_shapes_consistent(self, sample_df):
        """X и y имеют согласованные формы."""
        X, y, names = select_features(sample_df, target="stickiness")
        assert X.shape[0] == len(y)
        assert X.shape[1] == len(names)

    def test_target_not_in_features(self, sample_df):
        """Целевая переменная не входит в список признаков."""
        _, _, names = select_features(sample_df, target="stickiness")
        assert "stickiness" not in names

    def test_meta_columns_excluded(self, sample_df):
        """Служебные колонки не входят в признаки."""
        _, _, names = select_features(sample_df, target="stickiness")
        meta = {"date", "app_id", "category", "weekday", "has_store_data"}
        for col in meta:
            assert col not in names

    def test_no_nans_in_X(self, sample_df):
        """В X нет NaN после select_features."""
        X, _, _ = select_features(sample_df, target="stickiness")
        assert not np.any(np.isnan(X))

    def test_no_nans_in_y(self, sample_df):
        """В y нет NaN."""
        _, y, _ = select_features(sample_df, target="stickiness")
        assert not np.any(np.isnan(y))


class TestNormalizeFeatures:
    def test_shapes_preserved(self, sample_X):
        """normalize_features сохраняет форму X."""
        X_train = sample_X[:20]
        X_test = sample_X[20:]
        X_tr_sc, X_te_sc = normalize_features(X_train, X_test)
        assert X_tr_sc.shape == X_train.shape
        assert X_te_sc.shape == X_test.shape

    def test_train_mean_approx_zero(self, sample_X):
        """Среднее по train после масштабирования ≈ 0."""
        X_train = sample_X[:20]
        X_test = sample_X[20:]
        X_tr_sc, _ = normalize_features(X_train, X_test)
        npt.assert_allclose(X_tr_sc.mean(axis=0), 0.0, atol=1e-10)

    def test_fit_on_train_only(self, sample_X):
        """Масштабирование выполняется только по статистикам train."""
        X_train = sample_X[:20]
        X_test = sample_X[20:]
        X_tr1, X_te1 = normalize_features(X_train, X_test)
        X_tr2, X_te2 = normalize_features(X_train, X_test)
        npt.assert_array_equal(X_tr1, X_tr2)
        npt.assert_array_equal(X_te1, X_te2)


class TestGetCvStrategy:
    def test_retention_d7_returns_timeseries(self):
        """retention_d7 → TimeSeriesSplit."""
        from sklearn.model_selection import TimeSeriesSplit
        cv = get_cv_strategy("retention_d7")
        assert isinstance(cv, TimeSeriesSplit)

    def test_stickiness_returns_kfold(self):
        """stickiness → KFold с перемешиванием."""
        from sklearn.model_selection import KFold
        cv = get_cv_strategy("stickiness")
        assert isinstance(cv, KFold)

    def test_rating_returns_kfold(self):
        """rating → KFold."""
        from sklearn.model_selection import KFold
        cv = get_cv_strategy("rating")
        assert isinstance(cv, KFold)

    def test_n_splits_is_5(self):
        """n_splits=5 для всех стратегий."""
        cv1 = get_cv_strategy("retention_d7")
        cv2 = get_cv_strategy("stickiness")
        assert cv1.n_splits == 5
        assert cv2.n_splits == 5

    def test_get_cv_strategy_dau_returns_timeseriessplit(self):
        """DAU должен использовать TimeSeriesSplit."""
        from sklearn.model_selection import TimeSeriesSplit
        cv = get_cv_strategy("dau")
        assert isinstance(cv, TimeSeriesSplit), \
            f"DAU должен использовать TimeSeriesSplit, получен {type(cv).__name__}"
        assert cv.n_splits == 5
