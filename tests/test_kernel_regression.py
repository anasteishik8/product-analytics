"""
Тесты для src/kernel_regression.py.

Проверяет:
    - Smoke-тест fit + predict
    - bandwidth_cv возвращает валидное float
    - Формы выходных массивов ядерных функций
    - Детерминизм с seed
    - Работу с 1D входными данными
    - Корректность предсказания на известных функциях
"""
from __future__ import annotations

import numpy as np
import pytest
import numpy.testing as npt

from src.kernel_regression import (
    NadarayaWatsonRegressor,
    LocalLinearRegressor,
    gaussian_kernel,
    epanechnikov_kernel,
    triangular_kernel,
)


# ---- Тесты ядерных функций ----

class TestGaussianKernel:
    def test_shape_matches_input(self):
        """Форма вывода совпадает с формой ввода."""
        u = np.array([0.0, 1.0, -1.0, 2.0])
        result = gaussian_kernel(u)
        assert result.shape == u.shape

    def test_maximum_at_zero(self):
        """Максимум гауссовского ядра достигается в u=0."""
        u = np.array([0.0, 0.5, 1.0, 2.0])
        values = gaussian_kernel(u)
        assert values[0] == max(values)

    def test_nonnegative(self):
        """Значения неотрицательны."""
        u = np.linspace(-3, 3, 50)
        assert (gaussian_kernel(u) >= 0).all()

    def test_known_value(self):
        """Проверяет известное значение в u=0: K(0) = 1/sqrt(2*pi)."""
        expected = 1.0 / np.sqrt(2 * np.pi)
        npt.assert_allclose(gaussian_kernel(np.array([0.0]))[0], expected, rtol=1e-6)


class TestEpanechnikovKernel:
    def test_zero_outside_support(self):
        """Нули за пределами [-1, 1]."""
        u = np.array([-2.0, -1.5, 1.5, 2.0])
        npt.assert_array_equal(epanechnikov_kernel(u), np.zeros(4))

    def test_maximum_at_zero(self):
        """Максимум 0.75 в u=0."""
        result = epanechnikov_kernel(np.array([0.0]))
        npt.assert_allclose(result[0], 0.75, rtol=1e-6)

    def test_shape(self):
        u = np.random.randn(100)
        result = epanechnikov_kernel(u)
        assert result.shape == u.shape


class TestTriangularKernel:
    def test_zero_outside_support(self):
        u = np.array([-1.5, 1.5])
        npt.assert_array_equal(triangular_kernel(u), np.zeros(2))

    def test_maximum_at_zero(self):
        result = triangular_kernel(np.array([0.0]))
        npt.assert_allclose(result[0], 1.0, rtol=1e-6)


# ---- Тесты NadarayaWatsonRegressor ----

class TestNadarayaWatsonRegressor:
    def test_smoke_fit_predict(self, sample_X, sample_y):
        """Smoke: fit + predict проходит без ошибок."""
        model = NadarayaWatsonRegressor(kernel="gaussian", bandwidth=1.0)
        model.fit(sample_X, sample_y)
        preds = model.predict(sample_X)
        assert preds.shape == (len(sample_X),)

    def test_predict_shape_matches_input(self, sample_X, sample_y):
        """Форма предсказания совпадает с числом строк X."""
        model = NadarayaWatsonRegressor().fit(sample_X, sample_y)
        X_new = sample_X[:10]
        preds = model.predict(X_new)
        assert preds.shape == (10,)

    def test_predict_before_fit_raises(self, sample_X):
        """predict() до fit() должен поднимать RuntimeError."""
        model = NadarayaWatsonRegressor()
        with pytest.raises(RuntimeError, match="не обучена"):
            model.predict(sample_X)

    def test_bandwidth_cv_returns_valid_float(self, sample_X, sample_y):
        """bandwidth_cv возвращает положительное float из сетки."""
        from sklearn.model_selection import KFold
        model = NadarayaWatsonRegressor()
        bandwidths = np.array([0.1, 0.5, 1.0, 2.0])
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        best_h = model.bandwidth_cv(sample_X, sample_y, cv=cv, bandwidths=bandwidths)
        assert isinstance(best_h, float)
        assert best_h > 0
        assert best_h in bandwidths

    def test_seed_determinism(self, sample_X, sample_y):
        """Два прогона с одним seed дают идентичный результат."""
        model1 = NadarayaWatsonRegressor(bandwidth=0.5).fit(sample_X, sample_y)
        model2 = NadarayaWatsonRegressor(bandwidth=0.5).fit(sample_X, sample_y)
        npt.assert_array_equal(model1.predict(sample_X), model2.predict(sample_X))

    def test_different_kernels_give_different_predictions(self, sample_X, sample_y):
        """Разные ядра дают разные предсказания (на одинаковых данных)."""
        m_gauss = NadarayaWatsonRegressor(kernel="gaussian", bandwidth=1.0).fit(sample_X, sample_y)
        m_epan = NadarayaWatsonRegressor(kernel="epanechnikov", bandwidth=1.0).fit(sample_X, sample_y)
        p1 = m_gauss.predict(sample_X)
        p2 = m_epan.predict(sample_X)
        # Должны отличаться хотя бы немного
        assert not np.allclose(p1, p2)

    def test_invalid_kernel_raises(self):
        """Неизвестное ядро — ValueError."""
        with pytest.raises(ValueError, match="Неизвестное ядро"):
            NadarayaWatsonRegressor(kernel="unknown_kernel")

    def test_invalid_bandwidth_raises(self):
        """Неположительная ширина окна — ValueError."""
        with pytest.raises(ValueError):
            NadarayaWatsonRegressor(bandwidth=-1.0)

    def test_1d_input(self, sample_y):
        """Работает с 1D входными данными."""
        X_1d = np.linspace(0, 5, len(sample_y))
        model = NadarayaWatsonRegressor(bandwidth=0.5)
        model.fit(X_1d, sample_y)
        preds = model.predict(X_1d)
        assert preds.shape == (len(sample_y),)

    def test_no_nans_in_predictions(self, sample_X, sample_y):
        """Предсказания не содержат NaN."""
        model = NadarayaWatsonRegressor(bandwidth=1.0).fit(sample_X, sample_y)
        preds = model.predict(sample_X)
        assert not np.any(np.isnan(preds))


# ---- Тесты LocalLinearRegressor ----

class TestLocalLinearRegressor:
    def test_smoke_fit_predict(self, sample_X, sample_y):
        """Smoke: fit + predict без ошибок."""
        model = LocalLinearRegressor(kernel="gaussian", bandwidth=2.0)
        model.fit(sample_X, sample_y)
        preds = model.predict(sample_X[:5])
        assert preds.shape == (5,)

    def test_predict_shape(self, sample_X, sample_y):
        model = LocalLinearRegressor(bandwidth=1.0).fit(sample_X, sample_y)
        preds = model.predict(sample_X)
        assert preds.shape == (len(sample_X),)

    def test_bandwidth_cv_returns_float(self, sample_X, sample_y):
        """bandwidth_cv возвращает валидное float."""
        from sklearn.model_selection import KFold
        model = LocalLinearRegressor()
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        best_h = model.bandwidth_cv(sample_X, sample_y, cv=cv,
                                     bandwidths=np.array([0.5, 1.0, 2.0]))
        assert isinstance(best_h, float)
        assert best_h > 0

    def test_no_nans(self, sample_X, sample_y):
        """Нет NaN в предсказаниях."""
        model = LocalLinearRegressor(bandwidth=2.0).fit(sample_X, sample_y)
        preds = model.predict(sample_X)
        assert not np.any(np.isnan(preds))
