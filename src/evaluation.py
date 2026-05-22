"""
Метрики качества и статистические тесты для оценки моделей прогнозирования.

Реализует:
    - Регрессионные метрики: RMSE, MAE, MAPE, sMAPE, MASE, R²
    - Метрики классификации: ROC-AUC, Precision, Recall, F1
    - Тест Дибольда–Мариано для сравнения прогнозных моделей
    - Сводная таблица сравнения моделей

Ссылки:
    [6] Hyndman R.J., Koehler A.B. (2006). Another look at measures of forecast accuracy.
    [7] Diebold F.X., Mariano R.S. (1995). Comparing predictive accuracy.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "",
    y_train: Optional[np.ndarray] = None,
) -> dict[str, float | None | str]:
    """Вычисляет набор метрик качества регрессии.

    Метрики:
        - RMSE = sqrt(mean((y_true - y_pred)^2))
        - MAE  = mean(|y_true - y_pred|)
        - MAPE = mean(|y_true - y_pred| / |y_true|) * 100   [None при нулях]
        - sMAPE = mean(2*|y-y_hat| / (|y| + |y_hat|)) * 100
        - R²   = 1 - SS_res / SS_tot
        - MASE = MAE / MAE_naive (наивный прогноз = предыдущее значение)

    Parameters
    ----------
    y_true : np.ndarray
        Истинные значения.
    y_pred : np.ndarray
        Предсказанные значения.
    target_name : str, optional
        Имя целевой переменной (для предупреждений в лог).
    y_train : np.ndarray или None, optional
        Обучающая выборка для расчёта MASE. Если None — MASE=None.

    Returns
    -------
    dict[str, float | None | str]
        Словарь с ключами: rmse, mae, mape, smape, r2, mase.
        mape может быть None при y_true с нулями.

    Examples
    --------
    >>> m = compute_regression_metrics(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
    >>> 'rmse' in m
    True
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residuals = y_true - y_pred

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # MAPE — None при нулях
    mape: Optional[float] = None
    if np.abs(y_true).min() < 1e-6:
        warnings.warn(
            f"MAPE для '{target_name}' не вычислен: y_true содержит значения ≈ 0. "
            "Используйте RMSE или MASE как основную метрику.",
            UserWarning,
            stacklevel=2,
        )
    else:
        mape = float(np.mean(np.abs(residuals / y_true)) * 100)

    # sMAPE
    denom = np.abs(y_true) + np.abs(y_pred)
    smape_vals = np.where(denom > 1e-12, 2 * np.abs(residuals) / denom, 0.0)
    smape = float(np.mean(smape_vals) * 100)

    # MASE (относительно наивного прогноза в обучающей выборке)
    mase: Optional[float] = None
    if y_train is not None and len(y_train) > 1:
        naive_errors = np.abs(np.diff(y_train))
        mae_naive = float(np.mean(naive_errors))
        if mae_naive > 1e-12:
            mase = float(mae / mae_naive)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "mase": mase,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Вычисляет метрики бинарной классификации.

    Parameters
    ----------
    y_true : np.ndarray
        Истинные бинарные метки (0/1).
    y_pred_proba : np.ndarray
        Предсказанные вероятности класса 1.
    threshold : float, optional
        Порог бинаризации предсказаний. По умолчанию 0.5.

    Returns
    -------
    dict[str, float]
        Словарь: roc_auc, precision, recall, f1.

    Examples
    --------
    >>> m = compute_classification_metrics(np.array([0, 1, 1]), np.array([0.2, 0.8, 0.7]))
    >>> 0.0 <= m['roc_auc'] <= 1.0
    True
    """
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, f1_score
    )

    y_true = np.array(y_true, dtype=int)
    y_pred_proba = np.array(y_pred_proba, dtype=float)
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
) -> dict[str, float | str]:
    """Тест Дибольда–Мариано для сравнения прогнозных моделей.

    H₀: Обе модели имеют одинаковую прогностическую точность.
    H₁: Модели различаются по точности (двусторонний тест).

    Статистика:
        d_t = loss1(t) - loss2(t), где loss = squared error
        DM = mean(d) / sqrt(Var(d) / n)

    Parameters
    ----------
    errors1 : np.ndarray
        Остатки (y_true - y_pred) первой модели.
    errors2 : np.ndarray
        Остатки второй модели.
    h : int, optional
        Горизонт прогноза (влияет на оценку дисперсии). По умолчанию 1.

    Returns
    -------
    dict[str, float | str]
        Словарь: dm_stat (DM-статистика), p_value, winner ('model1', 'model2', 'tie').

    References
    ----------
    Diebold F.X., Mariano R.S. (1995). Comparing predictive accuracy.
    Journal of Business and Economic Statistics.

    Examples
    --------
    >>> res = diebold_mariano_test(np.array([0.1, 0.2]), np.array([0.15, 0.1]))
    >>> 'p_value' in res
    True
    """
    errors1 = np.array(errors1, dtype=float)
    errors2 = np.array(errors2, dtype=float)

    if len(errors1) != len(errors2):
        raise ValueError("errors1 и errors2 должны иметь одинаковую длину")

    # Потери — квадратичные ошибки
    loss1 = errors1 ** 2
    loss2 = errors2 ** 2
    d = loss1 - loss2

    n = len(d)
    d_mean = np.mean(d)

    # Оценка дисперсии с учётом автокорреляции (Newey-West для h>1)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_lag = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        gamma_sum += (1 - lag / h) * gamma_lag

    var_d = (gamma0 + 2 * gamma_sum) / n

    if var_d < 1e-20:
        # Практически идентичные ошибки
        return {"dm_stat": 0.0, "p_value": 1.0, "winner": "tie"}

    dm_stat = float(d_mean / np.sqrt(var_d))
    p_value = float(2 * (1 - stats.norm.cdf(np.abs(dm_stat))))

    if p_value > 0.05:
        winner = "tie"
    elif dm_stat > 0:
        winner = "model2"  # модель 2 лучше (меньшие потери)
    else:
        winner = "model1"

    return {
        "dm_stat": dm_stat,
        "p_value": p_value,
        "winner": winner,
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Строит сводную таблицу метрик по всем моделям.

    Parameters
    ----------
    results : dict
        Словарь {model_name: {metrics: {rmse, mae, mape, ...}, ...}}.

    Returns
    -------
    pd.DataFrame
        Таблица с колонками [model, rmse, mae, mape, smape, r2, mase].
        Отсортирована по rmse (ascending).

    Examples
    --------
    >>> df = compare_models({'Ridge': {'metrics': {'rmse': 0.01}}})
    >>> 'model' in df.columns
    True
    """
    rows = []
    for model_name, model_data in results.items():
        metrics = model_data.get("metrics", {})
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["model", "rmse", "mae", "mape", "smape", "r2", "mase"])

    df = pd.DataFrame(rows)

    # Сортируем по RMSE если есть
    if "rmse" in df.columns:
        df = df.sort_values("rmse", ascending=True).reset_index(drop=True)

    return df
