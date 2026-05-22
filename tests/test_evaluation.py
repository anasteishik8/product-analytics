"""
Тесты для src/evaluation.py.

Проверяет:
    - compute_regression_metrics возвращает правильные ключи
    - MAPE = None при нулях в y_true
    - Тест Дибольда-Мариано возвращает словарь с p_value
    - ROC-AUC в диапазоне [0, 1]
    - compare_models возвращает DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import numpy.testing as npt
import warnings

from src.evaluation import (
    compute_regression_metrics,
    compute_classification_metrics,
    diebold_mariano_test,
    compare_models,
)


class TestComputeRegressionMetrics:
    def test_returns_all_keys(self):
        """Возвращает словарь со всеми ожидаемыми ключами."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        result = compute_regression_metrics(y_true, y_pred)
        expected_keys = {"rmse", "mae", "mape", "smape", "r2", "mase"}
        assert expected_keys.issubset(set(result.keys()))

    def test_rmse_is_correct(self):
        """RMSE вычислен правильно."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # RMSE = sqrt((1+0+1)/3) = sqrt(2/3)
        expected_rmse = np.sqrt(2 / 3)
        result = compute_regression_metrics(y_true, y_pred)
        npt.assert_allclose(result["rmse"], expected_rmse, rtol=1e-6)

    def test_mape_none_when_zeros(self):
        """MAPE = None при наличии нулей в y_true."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 0.9, 2.1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = compute_regression_metrics(y_true, y_pred, target_name="test")
        assert result["mape"] is None

    def test_mape_warning_issued(self):
        """UserWarning выдаётся при нулях в y_true."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 0.9, 2.1])
        with pytest.warns(UserWarning, match="MAPE"):
            compute_regression_metrics(y_true, y_pred, target_name="retention_d7")

    def test_mape_computed_when_nonzero(self):
        """MAPE вычисляется правильно при ненулевых y_true."""
        y_true = np.array([1.0, 2.0, 4.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        result = compute_regression_metrics(y_true, y_pred)
        npt.assert_allclose(result["mape"], 0.0, atol=1e-9)

    def test_r2_perfect_prediction(self):
        """R² = 1.0 при идеальном предсказании."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_regression_metrics(y_true, y_true.copy())
        npt.assert_allclose(result["r2"], 1.0, rtol=1e-6)

    def test_mae_nonnegative(self):
        """MAE неотрицателен."""
        rng = np.random.RandomState(42)
        y_true = rng.randn(50)
        y_pred = rng.randn(50)
        result = compute_regression_metrics(y_true + 5, y_pred + 5)  # сдвиг чтобы не было нулей
        assert result["mae"] >= 0

    def test_smape_range(self):
        """sMAPE находится в диапазоне [0, 200]."""
        rng = np.random.RandomState(42)
        y_true = np.abs(rng.randn(50)) + 0.1
        y_pred = np.abs(rng.randn(50)) + 0.1
        result = compute_regression_metrics(y_true, y_pred)
        assert 0 <= result["smape"] <= 200

    def test_mase_with_ytrain(self):
        """MASE вычисляется при наличии y_train."""
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0])
        y_pred = np.array([6.0, 7.0])  # идеальное предсказание
        result = compute_regression_metrics(y_true, y_pred, y_train=y_train)
        # MAE=0, MASE=0
        npt.assert_allclose(result["mase"], 0.0, atol=1e-9)


class TestComputeClassificationMetrics:
    def test_roc_auc_in_range(self):
        """ROC-AUC в диапазоне [0, 1]."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 30)
        y_proba = rng.uniform(0, 1, 30)
        result = compute_classification_metrics(y_true, y_proba)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_returns_all_keys(self):
        """Возвращает ожидаемые ключи."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.7, 0.3, 0.9])
        result = compute_classification_metrics(y_true, y_proba)
        assert set(result.keys()) == {"roc_auc", "precision", "recall", "f1"}

    def test_perfect_classification(self):
        """ROC-AUC = 1.0 при идеальном классификаторе."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_classification_metrics(y_true, y_proba)
        npt.assert_allclose(result["roc_auc"], 1.0, rtol=1e-6)

    def test_f1_in_range(self):
        """F1 в диапазоне [0, 1]."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 50)
        y_proba = rng.uniform(0, 1, 50)
        result = compute_classification_metrics(y_true, y_proba)
        assert 0.0 <= result["f1"] <= 1.0


class TestDieboldMarianoTest:
    def test_returns_dict_with_p_value(self):
        """Возвращает словарь с p_value."""
        rng = np.random.RandomState(42)
        errors1 = rng.randn(50)
        errors2 = rng.randn(50) * 0.8
        result = diebold_mariano_test(errors1, errors2)
        assert "p_value" in result
        assert "dm_stat" in result
        assert "winner" in result

    def test_p_value_in_range(self):
        """p_value в диапазоне [0, 1]."""
        rng = np.random.RandomState(42)
        errors1 = rng.randn(100)
        errors2 = rng.randn(100)
        result = diebold_mariano_test(errors1, errors2)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_identical_errors_give_tie(self):
        """Идентичные ошибки — ничья (winner='tie', p_value≈1)."""
        errors = np.array([0.1, -0.2, 0.15, -0.05, 0.3])
        result = diebold_mariano_test(errors, errors)
        assert result["winner"] == "tie"

    def test_winner_is_valid_string(self):
        """winner ∈ {'model1', 'model2', 'tie'}."""
        rng = np.random.RandomState(42)
        errors1 = rng.randn(50) * 2
        errors2 = rng.randn(50) * 0.5  # явно лучше
        result = diebold_mariano_test(errors1, errors2)
        assert result["winner"] in {"model1", "model2", "tie"}

    def test_mismatched_lengths_raise(self):
        """Разные длины errors1 и errors2 — ValueError."""
        with pytest.raises(ValueError):
            diebold_mariano_test(np.array([1.0, 2.0]), np.array([1.0]))

    def test_better_model_wins(self):
        """Модель с меньшими ошибками выигрывает (при достаточном n)."""
        rng = np.random.RandomState(42)
        errors_good = rng.randn(200) * 0.1   # маленькие ошибки
        errors_bad = rng.randn(200) * 5.0    # большие ошибки
        result = diebold_mariano_test(errors_bad, errors_good)
        # errors_bad > errors_good → модель 2 лучше
        assert result["p_value"] < 0.05
        assert result["winner"] == "model2"


class TestCompareModels:
    def test_returns_dataframe(self):
        """compare_models возвращает DataFrame."""
        results = {
            "Ridge": {"metrics": {"rmse": 0.05, "mae": 0.04, "r2": 0.9}},
            "RF": {"metrics": {"rmse": 0.03, "mae": 0.025, "r2": 0.95}},
        }
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)

    def test_sorted_by_rmse(self):
        """Таблица отсортирована по RMSE (ascending)."""
        results = {
            "A": {"metrics": {"rmse": 0.1}},
            "B": {"metrics": {"rmse": 0.05}},
            "C": {"metrics": {"rmse": 0.08}},
        }
        df = compare_models(results)
        rmse_vals = df["rmse"].tolist()
        assert rmse_vals == sorted(rmse_vals)

    def test_model_column_present(self):
        """Колонка 'model' присутствует."""
        results = {"Test": {"metrics": {"rmse": 0.01}}}
        df = compare_models(results)
        assert "model" in df.columns
