"""
Тесты для src/scenario_analysis.py.

Проверяет:
    - run_scenario возвращает правильные ключи
    - BCa CI: lower < upper
    - delta_pct является float
    - define_scenarios_floodit возвращает 4 сценария
    - Детерминизм с seed
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import numpy.testing as npt

from src.scenario_analysis import ScenarioAnalyzer, define_scenarios_floodit


@pytest.fixture
def nw_scenario_analyzer(sample_X, sample_y, fitted_nw_model):
    """ScenarioAnalyzer на основе обученной NW-модели."""
    return ScenarioAnalyzer(
        model=fitted_nw_model,
        X_train=sample_X,
        feature_names=[f"feature_{i}" for i in range(sample_X.shape[1])],
        y_train=sample_y,
    )


class TestScenarioAnalyzer:
    def test_run_scenario_returns_correct_keys(self, nw_scenario_analyzer, sample_X):
        """run_scenario возвращает словарь с обязательными ключами."""
        result = nw_scenario_analyzer.run_scenario(
            X_baseline=sample_X,
            changes={},
            n_simulations=50,
            seed=42,
        )
        required_keys = {"median", "mean", "ci_95", "distribution",
                         "p_exceed_target", "baseline_median", "delta_pct",
                         "n_simulations", "changes"}
        assert required_keys.issubset(set(result.keys()))

    def test_distribution_shape(self, nw_scenario_analyzer, sample_X):
        """distribution имеет n_simulations элементов."""
        n_sim = 100
        result = nw_scenario_analyzer.run_scenario(
            X_baseline=sample_X,
            changes={},
            n_simulations=n_sim,
            seed=42,
        )
        assert len(result["distribution"]) == n_sim

    def test_ci_95_lower_less_than_upper(self, nw_scenario_analyzer, sample_X):
        """BCa CI: нижняя граница < верхней границы."""
        result = nw_scenario_analyzer.run_scenario(
            X_baseline=sample_X,
            changes={},
            n_simulations=200,
            seed=42,
        )
        lower, upper = result["ci_95"]
        assert lower <= upper

    def test_delta_pct_is_float(self, nw_scenario_analyzer, sample_X):
        """delta_pct имеет тип float."""
        result = nw_scenario_analyzer.run_scenario(
            X_baseline=sample_X,
            changes={},
            n_simulations=50,
            seed=42,
        )
        assert isinstance(result["delta_pct"], float)

    def test_p_exceed_in_range(self, nw_scenario_analyzer, sample_X):
        """p_exceed_target в диапазоне [0, 1]."""
        result = nw_scenario_analyzer.run_scenario(
            X_baseline=sample_X,
            changes={},
            n_simulations=100,
            seed=42,
        )
        assert 0.0 <= result["p_exceed_target"] <= 1.0

    def test_seed_determinism(self, nw_scenario_analyzer, sample_X):
        """Два прогона с одним seed дают идентичный результат."""
        result1 = nw_scenario_analyzer.run_scenario(
            X_baseline=sample_X, changes={}, n_simulations=100, seed=42
        )
        result2 = nw_scenario_analyzer.run_scenario(
            X_baseline=sample_X, changes={}, n_simulations=100, seed=42
        )
        npt.assert_allclose(result1["distribution"], result2["distribution"])

    def test_1d_baseline_works(self, nw_scenario_analyzer, sample_X):
        """Работает с 1D baseline (одна строка)."""
        x_1d = sample_X[0]
        result = nw_scenario_analyzer.run_scenario(
            X_baseline=x_1d, changes={}, n_simulations=50, seed=42
        )
        assert "median" in result

    def test_unknown_feature_warning(self, nw_scenario_analyzer, sample_X):
        """Предупреждение при неизвестном признаке в changes."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nw_scenario_analyzer.run_scenario(
                X_baseline=sample_X,
                changes={"nonexistent_feature": 0.5},
                n_simulations=50,
                seed=42,
            )
        # Проверяем, что было предупреждение
        assert any("не найден" in str(warning.message) for warning in w)

    def test_baseline_delta_zero(self, nw_scenario_analyzer, sample_X):
        """Базовый сценарий без изменений: delta_pct близко к нулю (нет входного шума).

        Важно: residual_std_ может быть ненулевым, если y_train был передан.
        Тест проверяет, что без изменений и без входного шума средне предсказание
        не меняется (дисперсия обусловлена только выходным шумом).
        """
        n_simulations = 200
        result = nw_scenario_analyzer.run_scenario(
            X_baseline=sample_X,
            changes={},
            n_simulations=n_simulations,
            noise_std=0.0,  # без входного шума
            seed=42,
        )
        # Среднее предсказание должно быть близко к baseline (дисперсия от residual_std_)
        baseline_pred = result["baseline_median"]
        mean_pred = result["mean"]
        # При baseline (changes={}) выходной шум центрирован вокруг baseline_pred.
        # Стандартная ошибка среднего: residual_std / sqrt(n). Допуск 4*SE — ~99.99%.
        se = nw_scenario_analyzer.residual_std_ / math.sqrt(n_simulations)
        npt.assert_allclose(mean_pred, baseline_pred, atol=4 * se)


class TestBcaBootstrapCI:
    def test_lower_less_than_upper(self, nw_scenario_analyzer):
        """BCa CI: lower < upper."""
        rng = np.random.RandomState(42)
        data = rng.randn(200) + 5.0
        lower, upper = nw_scenario_analyzer.bca_bootstrap_ci(data, seed=42)
        assert lower < upper

    def test_ci_contains_median(self, nw_scenario_analyzer):
        """Медиана входит в 95% CI."""
        rng = np.random.RandomState(42)
        data = rng.randn(500) + 3.0
        lower, upper = nw_scenario_analyzer.bca_bootstrap_ci(data, seed=42)
        median = float(np.median(data))
        assert lower <= median <= upper

    def test_wider_for_smaller_alpha(self, nw_scenario_analyzer):
        """Меньший alpha → более широкий CI."""
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        lo_95, hi_95 = nw_scenario_analyzer.bca_bootstrap_ci(data, alpha=0.05, seed=42)
        lo_80, hi_80 = nw_scenario_analyzer.bca_bootstrap_ci(data, alpha=0.20, seed=42)
        assert (hi_95 - lo_95) >= (hi_80 - lo_80) * 0.9  # допуск из-за bootstrap стохастики


class TestDefineScenarios:
    def test_returns_4_scenarios(self):
        """Функция возвращает ровно 4 сценария."""
        scenarios = define_scenarios_floodit()
        assert len(scenarios) == 4

    def test_baseline_has_empty_changes(self):
        """Сценарий 'baseline' содержит пустой словарь changes."""
        scenarios = define_scenarios_floodit()
        baseline = next(s for s in scenarios if s["name"] == "baseline")
        assert baseline["changes"] == {}

    def test_required_keys_in_each_scenario(self):
        """Каждый сценарий имеет ключи name, description, changes."""
        for s in define_scenarios_floodit():
            assert "name" in s
            assert "description" in s
            assert "changes" in s

    def test_combined_has_two_changes(self):
        """Комбинированный сценарий содержит два изменения."""
        scenarios = define_scenarios_floodit()
        combined = next(s for s in scenarios if s["name"] == "combined_improvement")
        assert len(combined["changes"]) == 2


class TestResidualStd:
    def test_residual_std_uses_real_residuals(self, sample_X, sample_y):
        """residual_std_ должен равняться std(y_train - model.predict(X_train))."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0).fit(sample_X, sample_y)
        feature_names = [f"f{i}" for i in range(sample_X.shape[1])]

        analyzer = ScenarioAnalyzer(
            model, sample_X, feature_names, y_train=sample_y
        )

        expected = float(np.std(sample_y - model.predict(sample_X)))
        npt.assert_allclose(analyzer.residual_std_, expected, rtol=1e-6)

    def test_residual_std_without_y_train_warns(self, sample_X):
        """Без y_train должно быть предупреждение и residual_std_=0.0."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0).fit(sample_X, sample_X[:, 0])
        feature_names = [f"f{i}" for i in range(sample_X.shape[1])]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer = ScenarioAnalyzer(model, sample_X, feature_names)
            assert any("y_train не передан" in str(warning.message) for warning in w)

        assert analyzer.residual_std_ == 0.0


def test_scenario_changes_actually_move_prediction_with_scaler():
    """Когда scaler передан, изменения признаков в RAW-шкале сдвигают прогноз."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    rng = np.random.RandomState(0)
    X_raw = rng.uniform(0, 1, size=(100, 3))
    # y зависит от первого признака линейно
    y = 2.0 * X_raw[:, 0] + 0.1 * rng.randn(100)

    scaler = StandardScaler().fit(X_raw)
    X_sc = scaler.transform(X_raw)
    model = Ridge(alpha=0.1).fit(X_sc, y)

    analyzer = ScenarioAnalyzer(
        model, X_raw, ["f0", "f1", "f2"],
        y_train=y, scaler=scaler,
    )

    base = analyzer.run_scenario(X_raw, changes={}, n_simulations=200, seed=42)
    high_f0 = analyzer.run_scenario(X_raw, changes={"f0": 0.9},
                                    n_simulations=200, seed=42)

    # f0 поднялся с медианы (~0.5) до 0.9, прогноз должен заметно вырасти
    assert high_f0["median"] > base["median"] + 0.5, (
        f"Изменение f0 не сдвинуло прогноз: base={base['median']:.3f}, "
        f"high={high_f0['median']:.3f}"
    )
