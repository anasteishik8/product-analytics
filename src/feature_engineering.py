"""
Feature engineering: загрузка данных, VIF-анализ, отбор признаков.

Реализует трёхступенчатый отбор признаков перед ядерной регрессией:
    1. Удаление служебных и нечисловых колонок
    2. VIF-фильтрация (Variance Inflation Factor > 10)
    3. Permutation importance — топ-15 признаков

Обоснование: при d=60, n=222 ядерная регрессия страдает от проклятия
размерности. Сокращение до d <= 15 критически важно для качества
оценки Надарая–Ватсона.

Ссылки:
    [4] Belsley D.A. et al. (1980). Regression Diagnostics.
    [5] Breiman L. (2001). Random Forests (permutation importance).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_PATH = Path("data/processed/floodit_final.parquet")
DEFAULT_SEED = 42

# Колонки, которые никогда не используются как признаки
_META_COLUMNS = {"date", "app_id", "category", "weekday", "has_store_data"}

# Целевые переменные — исключаются из признаков
# retention_d7_high включён явно: это бинарная производная от retention_d7,
# которая создаётся на лету в pipeline и не должна попадать в матрицу X
_ALL_TARGETS = {"stickiness", "rating", "retention_d7", "retention_d30",
                "retention_d1", "dau", "mau", "wau",
                "retention_d7_high", "retention_d1_high", "retention_d30_high"}


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Загружает финальный датасет из parquet-файла.

    Parameters
    ----------
    path : str или Path, optional
        Путь к файлу floodit_final.parquet. По умолчанию data/processed/floodit_final.parquet.

    Returns
    -------
    pd.DataFrame
        Датафрейм 222 × 60. Колонка date приведена к datetime64.

    Raises
    ------
    FileNotFoundError
        Если файл не найден по указанному пути.

    Examples
    --------
    >>> df = load_dataset()
    >>> df.shape
    (222, 60)
    """
    path = Path(path)
    if not path.is_absolute():
        # Ищем относительно корня проекта (два уровня вверх от src/)
        root = Path(__file__).parent.parent
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    df = pd.read_parquet(path)
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df


def compute_vif(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """Вычисляет VIF (Variance Inflation Factor) для числовых признаков.

    VIF_j = 1 / (1 - R^2_j), где R^2_j — коэффициент детерминации
    при регрессии j-го признака на все остальные.

    Признак с VIF > threshold имеет сильную мультиколлинеарность
    и рекомендован к удалению.

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с числовыми признаками (только признаки, без таргетов).
    threshold : float, optional
        Порог VIF. По умолчанию 10.0.

    Returns
    -------
    pd.DataFrame
        Датафрейм с колонками [feature, vif, keep], отсортированный
        по убыванию VIF. keep=True если VIF <= threshold.

    Examples
    --------
    >>> result = compute_vif(df[feature_cols])
    >>> low_vif_features = result.loc[result['keep'], 'feature'].tolist()
    """
    from sklearn.linear_model import LinearRegression

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Удаляем колонки с нулевой дисперсией или константы
    valid_cols = [c for c in numeric_cols if df[c].std() > 1e-10 and df[c].notna().sum() > 5]

    if len(valid_cols) < 2:
        return pd.DataFrame(columns=["feature", "vif", "keep"])

    # Заполняем NaN медианой для расчёта VIF
    X = df[valid_cols].copy()
    for col in valid_cols:
        X[col] = X[col].fillna(X[col].median())

    vif_values = []
    for i, col in enumerate(valid_cols):
        y_vif = X[col].values
        X_others = X.drop(columns=[col]).values

        # Регрессия col на все остальные
        reg = Ridge(alpha=1e-6).fit(X_others, y_vif)
        y_pred = reg.predict(X_others)
        ss_res = np.sum((y_vif - y_pred) ** 2)
        ss_tot = np.sum((y_vif - y_vif.mean()) ** 2)

        if ss_tot < 1e-12:
            r2 = 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        # Ограничиваем R^2 снизу чтобы избежать деления на ~0
        r2 = np.clip(r2, 0.0, 1.0 - 1e-9)
        vif = 1.0 / (1.0 - r2)
        vif_values.append(vif)

    result = pd.DataFrame(
        columns=["feature", "vif", "keep"],
        data={
            "feature": valid_cols,
            "vif": vif_values,
            "keep": [v <= threshold for v in vif_values],
        }
    )
    return result.sort_values("vif", ascending=False).reset_index(drop=True)


def get_cv_strategy(target: str) -> KFold | TimeSeriesSplit:
    """Возвращает стратегию кросс-валидации для заданного таргета.

    Для retention_d7 и dau используется TimeSeriesSplit (автокорреляция),
    для остальных — KFold с перемешиванием (IID допущение).

    Parameters
    ----------
    target : str
        Имя целевой переменной: 'stickiness', 'dau', 'retention_d7'.

    Returns
    -------
    KFold или TimeSeriesSplit
        Объект кросс-валидации с n_splits=5.

    Examples
    --------
    >>> cv = get_cv_strategy('retention_d7')
    >>> type(cv).__name__
    'TimeSeriesSplit'
    """
    if target in {"retention_d7", "dau"}:
        return TimeSeriesSplit(n_splits=5)
    return KFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Нормализует признаки StandardScaler (fit только на train).

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, d)
        Обучающие признаки.
    X_test : np.ndarray, shape (n_test, d)
        Тестовые признаки (трансформируются по статистикам train).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Пара (X_train_scaled, X_test_scaled) той же формы.

    Notes
    -----
    Не возвращает scaler-объект. Если downstream-код передаёт X в
    ScenarioAnalyzer (или иную функцию, требующую scaler для трансформации
    RAW-точек), используйте явный StandardScaler().fit(X_train) и сохраняйте
    объект; см. src/modeling.py:train_all_models.

    Examples
    --------
    >>> X_tr_sc, X_te_sc = normalize_features(X_train, X_test)
    >>> X_tr_sc.shape == X_train.shape
    True
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def select_features(
    df: pd.DataFrame,
    target: str,
    max_features: int = 15,
    app_id: Optional[str] = None,
    vif_threshold: float = 10.0,
    random_state: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Отбирает признаки через VIF-фильтрацию и permutation importance.

    Алгоритм:
        1. Отфильтровать строки по app_id (если задан)
        2. Исключить служебные колонки, все потенциальные таргеты, бинарные
        3. Заполнить NaN медианой
        4. VIF-фильтрация: удалить признаки с VIF > vif_threshold
        5. Обучить Ridge на 80% выборки
        6. Permutation importance → отобрать топ-max_features
        7. Вернуть X (все строки), y, feature_names

    Parameters
    ----------
    df : pd.DataFrame
        Полный датафрейм (все 222 строки × 60 колонок).
    target : str
        Целевая переменная. Должна присутствовать в df.
    max_features : int, optional
        Максимальное число признаков после отбора. По умолчанию 15.
    app_id : str или None, optional
        Если задан — фильтрует строки по app_id перед обучением.
    vif_threshold : float, optional
        Порог VIF. По умолчанию 10.0.
    random_state : int, optional
        Зерно генератора. По умолчанию 42.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        (X, y, feature_names), где X shape (n_valid, d_selected),
        y shape (n_valid,), feature_names — список d_selected имён.

    Notes
    -----
    Строки с NaN в target удаляются перед возвратом.
    Функция не меняет исходный датафрейм (copy()).
    """
    df_work = df.copy()

    # Фильтрация по app_id
    if app_id is not None:
        df_work = df_work[df_work["app_id"] == app_id].copy()

    # Удаляем строки с NaN в таргете
    df_work = df_work[df_work[target].notna()].copy()

    if len(df_work) < 10:
        raise ValueError(f"Слишком мало строк ({len(df_work)}) после фильтрации для target='{target}'")

    y = df_work[target].values.astype(float)

    # Определяем множество признаков: все числовые, кроме метаданных и таргетов
    exclude = _META_COLUMNS | _ALL_TARGETS
    candidate_cols = [
        c for c in df_work.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df_work[c])
        and df_work[c].std() > 1e-10
    ]

    if len(candidate_cols) == 0:
        raise ValueError("Нет кандидатов для признаков после фильтрации служебных колонок")

    # Заполняем NaN медианой
    X_df = df_work[candidate_cols].copy()
    for col in candidate_cols:
        X_df[col] = X_df[col].fillna(X_df[col].median())

    # VIF-фильтрация
    vif_result = compute_vif(X_df)
    low_vif_cols = vif_result.loc[vif_result["keep"], "feature"].tolist()

    if len(low_vif_cols) == 0:
        warnings.warn("VIF-фильтрация удалила все признаки, используем топ-20 по VIF", stacklevel=2)
        low_vif_cols = vif_result.head(20)["feature"].tolist()

    X_vif = X_df[low_vif_cols].values

    # Permutation importance на валидации
    n = len(X_vif)
    split_idx = int(n * 0.8)
    X_tr, X_val = X_vif[:split_idx], X_vif[split_idx:]
    y_tr, y_val = y[:split_idx], y[split_idx:]

    if len(X_val) < 5:
        # Слишком мало данных для val — используем всё для обучения
        X_tr, y_tr = X_vif, y
        X_val, y_val = X_vif, y

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(X_tr_sc, y_tr)

    perm = sklearn_permutation_importance(
        ridge, X_val_sc, y_val,
        n_repeats=10,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )

    # Сортируем по убыванию важности
    imp_means = perm.importances_mean
    sorted_idx = np.argsort(imp_means)[::-1]
    top_idx = sorted_idx[:max_features]
    selected_cols = [low_vif_cols[i] for i in top_idx]

    # Финальный X — только отобранные признаки
    X_final = X_df[selected_cols].values

    return X_final, y, selected_cols
