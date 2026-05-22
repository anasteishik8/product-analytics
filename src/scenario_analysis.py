"""
Сценарный анализ востребованности IT-продукта.

Реализует:
    - Симуляции Монте-Карло для оценки распределения прогноза при
      изменении входных признаков (what-if анализ)
    - BCa bootstrap (Bias Corrected and Accelerated) доверительные интервалы
    - Стандартные сценарии для Flood-It!: базовый, снижение crash_rate,
      рост retention_d7, комбинированный

BCa bootstrap предпочтительнее percentile bootstrap на малых выборках
(Efron & Tibshirani, 1993, An Introduction to the Bootstrap):
точнее покрытие, учитывает смещение оценщика и асимметрию распределения.

Ссылки:
    [9] Efron B., Tibshirani R.J. (1993). An Introduction to the Bootstrap.
    [10] Hyndman R.J. (2006). arxiv:2403.20182.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
DEFAULT_N_SIMULATIONS = 1000
DEFAULT_N_BOOTSTRAP = 2000


class ScenarioAnalyzer:
    """Анализатор сценариев на основе симуляций Монте-Карло.

    Использует обученную регрессионную модель для генерации распределения
    прогнозных значений при изменении входных признаков.

    Parameters
    ----------
    model : обученная sklearn-совместимая модель
        Должна реализовывать метод predict(X).
    X_train : np.ndarray, shape (n, d)
        Обучающая выборка (для оценки шума из остатков).
    feature_names : list[str]
        Имена признаков в порядке колонок X_train.
    y_train : np.ndarray, shape (n,), optional
        Обучающая целевая переменная. Если передана — используется
        для вычисления residual_std_ из реальных остатков.
        Если не передана — residual_std_ = 0.0 с предупреждением.
    scaler : object, optional
        Объект масштабирования (StandardScaler и т.п.).
        Сохраняется для последующего использования.

    Attributes
    ----------
    residual_std_ : float
        Стандартное отклонение остатков на обучающей выборке.
        Равно 0.0, если y_train не передан в __init__.

    Examples
    --------
    >>> analyzer = ScenarioAnalyzer(model, X_train, feature_names)
    >>> result = analyzer.run_scenario(X_baseline, {'crash_rate': 0.05})
    >>> result['median']
    0.127
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: list[str],
        y_train: Optional[np.ndarray] = None,
        scaler=None,
    ) -> None:
        self.model = model
        self.X_train = np.array(X_train)
        self.feature_names = list(feature_names)
        self.scaler = scaler

        if y_train is not None:
            residuals = np.asarray(y_train) - model.predict(X_train)
            self.residual_std_: float = float(np.std(residuals))
        else:
            warnings.warn(
                "y_train не передан — residual_std_ не вычисляется (=0). "
                "Передайте y_train для корректной оценки шума на выходе.",
                stacklevel=2,
            )
            self.residual_std_ = 0.0

    def run_scenario(
        self,
        X_baseline: np.ndarray,
        changes: dict[str, float],
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        noise_std: Optional[float] = None,
        seed: int = DEFAULT_SEED,
    ) -> dict[str, object]:
        """Запускает сценарный анализ Монте-Карло.

        Для каждой симуляции:
            1. Берём базовую точку X_baseline (или медиану X_train, если None)
            2. Применяем изменения из changes
            3. Добавляем гауссовский шум к признакам (noise_std)
            4. Предсказываем целевое значение

        Parameters
        ----------
        X_baseline : np.ndarray, shape (n_baseline, d) или (d,)
            Базовые значения признаков. Если 2D — используем медиану по строкам.
        changes : dict[str, float]
            Словарь изменений: {'crash_rate': 0.05, 'retention_d7': 0.15}.
            Ключи должны быть в self.feature_names.
        n_simulations : int, optional
            Число симуляций Монте-Карло. По умолчанию 1000.
        noise_std : float или None, optional
            Стандартное отклонение шума на признаках. Если None — 5% от std признаков.
        seed : int, optional
            Зерно генератора. По умолчанию 42.

        Returns
        -------
        dict
            {
                'median': медиана прогнозов,
                'mean': среднее прогнозов,
                'ci_95': (нижняя, верхняя граница BCa 95% CI),
                'distribution': np.ndarray прогнозов,
                'p_exceed_target': вероятность превышения baseline,
                'baseline_median': прогноз без изменений,
                'delta_pct': изменение в % от baseline,
                'n_simulations': n_simulations,
                'changes': changes,
            }
        """
        rng = np.random.RandomState(seed)

        # Базовая точка (в RAW-шкале если scaler есть, иначе в той же что X_baseline)
        if X_baseline.ndim == 1:
            x_base_raw = X_baseline.copy()
        else:
            x_base_raw = np.median(X_baseline, axis=0)

        x_scenario_raw = x_base_raw.copy()
        for feature_name, new_value in changes.items():
            if feature_name in self.feature_names:
                idx = self.feature_names.index(feature_name)
                x_scenario_raw[idx] = new_value
            else:
                warnings.warn(
                    f"Признак '{feature_name}' не найден в feature_names. "
                    f"Доступные: {self.feature_names[:5]}...",
                    stacklevel=2,
                )

        # Шум — в RAW-шкале по std признаков из X_train
        if noise_std is None:
            feature_stds = np.std(self.X_train, axis=0)
            noise_std_per_feature = feature_stds * 0.05
        else:
            noise_std_per_feature = np.full(len(x_scenario_raw), noise_std)

        # Базовый прогноз (трансформируем в шкалу модели если есть scaler)
        def _to_model_space(x_raw_1d: np.ndarray) -> np.ndarray:
            x_2d = x_raw_1d.reshape(1, -1)
            if self.scaler is not None:
                x_2d = self.scaler.transform(x_2d)
            return x_2d

        baseline_pred = float(self.model.predict(_to_model_space(x_base_raw))[0])

        # n_simulations симуляций
        simulations = np.zeros(n_simulations)
        for i in range(n_simulations):
            noise = rng.normal(0, noise_std_per_feature)
            x_noisy_raw = x_scenario_raw + noise
            pred = float(self.model.predict(_to_model_space(x_noisy_raw))[0])
            # Подмешиваем шум остатков на выходе
            if self.residual_std_ > 0:
                pred += rng.normal(0, self.residual_std_)
            simulations[i] = pred

        # BCa CI
        ci_lower, ci_upper = self.bca_bootstrap_ci(
            simulations,
            alpha=0.05,
            n_bootstrap=DEFAULT_N_BOOTSTRAP,
            seed=seed,
        )

        median_pred = float(np.median(simulations))
        mean_pred = float(np.mean(simulations))

        # Вероятность превышения baseline
        p_exceed = float(np.mean(simulations > baseline_pred))

        # Изменение в %
        if abs(baseline_pred) > 1e-10:
            delta_pct = float((median_pred - baseline_pred) / abs(baseline_pred) * 100)
        else:
            delta_pct = 0.0

        return {
            "median": median_pred,
            "mean": mean_pred,
            "ci_95": (ci_lower, ci_upper),
            "distribution": simulations,
            "p_exceed_target": p_exceed,
            "baseline_median": baseline_pred,
            "delta_pct": delta_pct,
            "n_simulations": n_simulations,
            "changes": changes,
        }

    def bca_bootstrap_ci(
        self,
        data: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        seed: int = DEFAULT_SEED,
    ) -> tuple[float, float]:
        """BCa bootstrap доверительный интервал (Bias Corrected and Accelerated).

        BCa bootstrap предпочтительнее percentile bootstrap:
        учитывает смещение оценки (bias correction) и изменение
        дисперсии при сдвиге параметра (acceleration).

        Формула:
            z0 = Φ^{-1}(#{θ* < θ_obs} / B)
            a  = sum(θ_jk - θ.)^3 / (6 * (sum(θ_jk - θ.)^2)^{3/2})
            α1 = Φ(z0 + (z0 + z_α/2)/(1 - a*(z0 + z_α/2)))
            α2 = Φ(z0 + (z0 + z_{1-α/2})/(1 - a*(z0 + z_{1-α/2})))

        Parameters
        ----------
        data : np.ndarray
            Выборка (прогнозы из Монте-Карло).
        alpha : float, optional
            Уровень значимости. По умолчанию 0.05 (95% CI).
        n_bootstrap : int, optional
            Число bootstrap-репликаций. По умолчанию 2000.
        seed : int, optional
            Зерно генератора.

        Returns
        -------
        tuple[float, float]
            (lower, upper) — границы доверительного интервала.
        """
        from scipy.stats import norm

        rng = np.random.RandomState(seed)
        n = len(data)
        theta_obs = np.median(data)

        # Bootstrap статистики
        boot_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            boot_sample = rng.choice(data, size=n, replace=True)
            boot_stats[i] = np.median(boot_sample)

        # Bias correction z0
        prop_less = np.mean(boot_stats < theta_obs)
        if prop_less <= 0:
            prop_less = 1e-10
        if prop_less >= 1:
            prop_less = 1 - 1e-10
        z0 = norm.ppf(prop_less)

        # Acceleration a (jackknife)
        jack_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jack_stats[i] = np.median(jack_sample)

        jack_mean = np.mean(jack_stats)
        diffs = jack_mean - jack_stats
        numer = np.sum(diffs ** 3)
        denom = np.sum(diffs ** 2)

        if denom < 1e-12:
            a = 0.0
        else:
            a = float(numer / (6.0 * denom ** 1.5))

        # Скорректированные квантили
        z_alpha_lo = norm.ppf(alpha / 2)
        z_alpha_hi = norm.ppf(1 - alpha / 2)

        def _adjusted_quantile(z_alpha_k: float) -> float:
            num = z0 + z_alpha_k
            denom_q = 1 - a * (z0 + z_alpha_k)
            if abs(denom_q) < 1e-12:
                return alpha / 2 if z_alpha_k < 0 else 1 - alpha / 2
            return float(norm.cdf(z0 + num / denom_q))

        alpha1 = _adjusted_quantile(z_alpha_lo)
        alpha2 = _adjusted_quantile(z_alpha_hi)

        # Ограничиваем в (0, 1)
        alpha1 = np.clip(alpha1, 1e-6, 1 - 1e-6)
        alpha2 = np.clip(alpha2, 1e-6, 1 - 1e-6)

        lower = float(np.quantile(boot_stats, alpha1))
        upper = float(np.quantile(boot_stats, alpha2))

        if lower > upper:
            lower, upper = upper, lower

        return lower, upper


def define_scenarios_floodit() -> list[dict]:
    """Возвращает 4 стандартных сценария для Flood-It!.

    Сценарии:
        1. Baseline — текущие метрики (crash_rate=0.132, retention_d7=0.081)
        2. Снижение crash_rate: 13.2% → 5%
        3. Рост retention_d7: 8.1% → 15%
        4. Комбинированный: оба изменения одновременно

    Returns
    -------
    list[dict]
        Список сценариев с ключами: name, description, changes.

    Examples
    --------
    >>> scenarios = define_scenarios_floodit()
    >>> len(scenarios)
    4
    >>> scenarios[0]['name']
    'baseline'
    """
    return [
        {
            "name": "baseline",
            "description": "Текущие метрики Flood-It! (crash_rate=13.2%, retention_d7=8.1%)",
            "changes": {},
        },
        {
            "name": "low_crash_rate",
            "description": "Снижение crash_rate с 13.2% до 5% (целевой порог Google Play Vitals)",
            "changes": {"crash_rate": 0.05},
        },
        {
            "name": "high_retention_d7",
            "description": "Рост retention_d7 с 8.1% до 15% (выше медианы категории Puzzle)",
            "changes": {"retention_d7": 0.15},
        },
        {
            "name": "combined_improvement",
            "description": "Совместное снижение crash_rate до 5% и рост retention_d7 до 15%",
            "changes": {"crash_rate": 0.05, "retention_d7": 0.15},
        },
    ]
