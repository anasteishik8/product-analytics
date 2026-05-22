"""Общие утилиты для EDA-ноутбуков.

Используется ноутбуками 01_source_*, 02_eda_*, 03_comparison_*.
Содержит: палитру для двух приложений, тему matplotlib/seaborn,
helper для сохранения фигур в figures/<notebook_slug>/<name>.pdf.
"""
from __future__ import annotations
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

_THIS_DIR = Path(__file__).resolve().parent      # .../notebooks
_PROJECT_ROOT = _THIS_DIR.parent                  # project root

# Цветовая схема: синий = успех, красный = провал
APP_PALETTE = {
    "com.labpixies.flood": "#3b82f6",
    "com.google.flood2":   "#ef4444",
}

APP_LABEL_MAP = {
    "com.labpixies.flood": "Flood-It! (успех)",
    "com.google.flood2":   "Flood-It! 2 (провал)",
}

FIGSIZE_DEFAULT = (8, 5)
FIGSIZE_WIDE = (12, 5)


def theme_setup() -> None:
    """Применяет глобальные настройки matplotlib + seaborn."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def app_label(app_id: str) -> str:
    """Человеко-читаемое имя приложения."""
    return APP_LABEL_MAP.get(app_id, app_id)


def save_figure(name: str, notebook_slug: str, fig: Figure | None = None) -> Path:
    """Сохраняет PDF фигуру в figures/<notebook_slug>/<name>.pdf.

    Parameters
    ----------
    name : str
        Имя файла без расширения (e.g., "dau_dynamics").
    notebook_slug : str
        Имя ноутбука без расширения (e.g., "01_source_product").
    fig : Figure или None
        Если None — берётся `plt.gcf()`.

    Returns
    -------
    Path
        Путь к сохранённому файлу.
    """
    out_dir = _PROJECT_ROOT / "figures" / notebook_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    out_path = out_dir / f"{name}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def figures_dir(notebook_slug: str) -> Path:
    """Возвращает путь к директории figures/<notebook_slug>/.

    Создаёт директорию если её нет.

    Parameters
    ----------
    notebook_slug : str
        Имя ноутбука без расширения.

    Returns
    -------
    Path
        Абсолютный путь к figures/<notebook_slug>/.
    """
    out_dir = _PROJECT_ROOT / "figures" / notebook_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# === Helpers for source/per-app EDA notebooks ===

def split_by_app(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит датасет на две DataFrame по app_id (Flood-It! и Flood-It! 2).

    Каждая возвращаемая DataFrame отсортирована по date и имеет reset_index.

    Parameters
    ----------
    df : pd.DataFrame
        Полный датасет с колонкой app_id.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_f1, df_f2) — Flood-It! и Flood-It! 2 соответственно.
    """
    f1 = df[df["app_id"] == "com.labpixies.flood"].copy().sort_values("date").reset_index(drop=True)
    f2 = df[df["app_id"] == "com.google.flood2"].copy().sort_values("date").reset_index(drop=True)
    return f1, f2


def top_correlated_pairs(corr: pd.DataFrame, threshold: float = 0.7, n: int = 5) -> pd.Series:
    """Возвращает топ-N пар признаков с |corr| > threshold (без диагонали и дублей).

    Parameters
    ----------
    corr : pd.DataFrame
        Корреляционная матрица.
    threshold : float, optional
        Минимальный модуль корреляции. По умолчанию 0.7.
    n : int, optional
        Сколько пар вернуть. По умолчанию 5.

    Returns
    -------
    pd.Series
        Индекс — MultiIndex (feat1, feat2), значения — corr, отсортировано по |corr| убывающе.
    """
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().sort_values(key=abs, ascending=False)
    return pairs[pairs.abs() > threshold].head(n)


def comparison_table(
    df_f1: pd.DataFrame, df_f2: pd.DataFrame, metrics: list[str],
) -> pd.DataFrame:
    """Сводная таблица median/std для двух приложений по списку метрик.

    Parameters
    ----------
    df_f1, df_f2 : pd.DataFrame
        Данные двух приложений.
    metrics : list[str]
        Колонки для сравнения.

    Returns
    -------
    pd.DataFrame с колонками [metric, f1_median, f2_median, f1_std, f2_std, delta_pct].
    """
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for m in metrics:
            v1, v2 = df_f1[m].median(), df_f2[m].median()
            delta_pct = (v2 - v1) / abs(v1) * 100 if v1 not in (0, np.nan) and not np.isnan(v1) else None
            rows.append({
                "metric": m,
                "f1_median": round(v1, 4) if not np.isnan(v1) else None,
                "f2_median": round(v2, 4) if not np.isnan(v2) else None,
                "f1_std":    round(df_f1[m].std(), 4),
                "f2_std":    round(df_f2[m].std(), 4),
                "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
            })
    return pd.DataFrame(rows)


def add_app_legend(fig, loc: str = "upper center", ncol: int = 2,
                   bbox_to_anchor: tuple = (0.5, 1.02)) -> None:
    """Добавляет общую легенду на figure с человеко-читаемыми именами приложений.

    Удаляет частные легенды на subplot'ах. Использовать ПОСЛЕ построения всех subplot'ов.

    Parameters
    ----------
    fig : matplotlib Figure.
    """
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=APP_PALETTE[k], lw=2, label=APP_LABEL_MAP[k])
               for k in APP_PALETTE]
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles=handles, loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor)


def annotate_sample_size(ax, df_f1: pd.DataFrame, df_f2: pd.DataFrame,
                          col: str, base_title: str | None = None) -> None:
    """Устанавливает заголовок subplot'а с n₁/n₂ для disclosing sparse data.

    Если base_title не передан — используется col.
    """
    n1 = df_f1[col].notna().sum()
    n2 = df_f2[col].notna().sum()
    title = base_title if base_title else col
    ax.set_title(f"{title} (n₁={n1}, n₂={n2})", fontsize=10)


def check_inf_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Возвращает список колонок, содержащих inf-значения.

    Печатает предупреждение, если такие найдены.
    """
    inf_cols = [c for c in cols if np.isinf(df[c].dropna()).any()]
    if inf_cols:
        print(f"⚠ inf-значения найдены в: {inf_cols}")
        print("  → будут трактоваться как NaN при описательной статистике и моделировании.")
    return inf_cols


def assert_shared_columns(
    df_f1: pd.DataFrame, df_f2: pd.DataFrame, cols: list[str],
    sentinel: float = -9999.0,
) -> None:
    """Падает с AssertionError если хоть одна колонка различается между f1 и f2.

    Используется для подтверждения, что Market/External метрики идентичны
    для обоих приложений (общая категория или общий рыночный фон).

    Parameters
    ----------
    df_f1, df_f2 : pd.DataFrame
        Данные двух приложений (отсортированные по date, reset_index).
    cols : list[str]
        Колонки для сравнения.
    sentinel : float, optional
        Значение для замены NaN перед сравнением. Должно быть вне реального диапазона.
        По умолчанию -9999. Для Trends-колонок (interest 0-100) безопасно.

    Raises
    ------
    AssertionError
        С указанием первой различающейся колонки.

    Examples
    --------
    >>> assert_shared_columns(df_f1, df_f2, ["competitor_count", "category_rank"])
    """
    for col in cols:
        same = (
            df_f1[col].fillna(sentinel).reset_index(drop=True)
            == df_f2[col].fillna(sentinel).reset_index(drop=True)
        ).all()
        assert same, f"Колонка {col} различается между f1 и f2"


# ── Modeling helpers ──────────────────────────────────────────────────────────
# Добавлены 2026-04-27 для нового workflow ноутбуков 04_modeling_*.ipynb.
# 2026-05-04: добавлен DEFAULT_OPERATIONAL_WHITELIST для гарантии управляемых
# признаков в финальном наборе (Variant Z, см. spec 2026-05-04-operational-whitelist).

DEFAULT_OPERATIONAL_WHITELIST: list[str] = [
    # Технические рычаги (баг-фиксы, оптимизация)
    "crash_rate", "crash_rate_ma7",
    # UX-рычаги (онбординг, контент)
    "onboarding_completion_rate",
    "avg_session_duration", "avg_session_duration_ma7", "median_session_duration",
    # Маркетинг (привлечение, удержание)
    "daily_installs", "daily_uninstalls",
    "total_sessions",
]


def chronological_split(
    df: pd.DataFrame, test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит DataFrame на (train, test) хронологически.

    Предполагает df отсортирован по дате. Возвращает копии срезов.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм (должен быть отсортирован по date).
    test_frac : float, optional
        Доля тестовой выборки в (0, 1). По умолчанию 0.2.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test) — копии без пересечения индексов.

    Raises
    ------
    ValueError
        Если test_frac не в (0, 1) или len(df) < 5.
    """
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac должен быть в (0, 1), получено {test_frac}")
    n = len(df)
    if n < 5:
        raise ValueError(f"DataFrame слишком маленький (n={n}); минимум 5 строк")
    if "date" in df.columns and not df["date"].is_monotonic_increasing:
        raise ValueError("df должен быть отсортирован по date перед chronological_split")
    sp = int(n * (1 - test_frac))
    return df.iloc[:sp].copy(), df.iloc[sp:].copy()


def select_features_clean(
    df_train: pd.DataFrame,
    target: str,
    max_features: int = 15,
    vif_threshold: float = 10.0,
    random_state: int = 42,
    operational_whitelist: Optional[list[str]] = None,
) -> list[str]:
    """Честный отбор признаков ТОЛЬКО на df_train (без leakage).

    Этапы:
        1. Исключить служебные колонки и все возможные таргеты.
        2. Отбросить колонки с константной/all-NaN дисперсией.
        3. Заполнить NaN медианой train.
        4. VIF-фильтрация с порогом vif_threshold.
        5. Внутренний 80/20 split в df_train для permutation importance с Ridge.
        6. Top-N признаков по permutation importance.

    Parameters
    ----------
    df_train : pd.DataFrame
        Только обучающая выборка. Тест НЕ передавать.
    target : str
        Целевая колонка.
    max_features : int, optional
        Максимальное число признаков на выходе. По умолчанию 15.
    vif_threshold : float, optional
        Порог VIF. По умолчанию 10.0.
    random_state : int, optional
        Seed для Ridge и permutation_importance. По умолчанию 42.
    operational_whitelist : list[str] | None, optional
        Список «управляемых» признаков, которые гарантированно попадают
        в результат, если прошли VIF. Если None — используется
        DEFAULT_OPERATIONAL_WHITELIST. Если [] — расширение отключено
        (legacy behaviour).

    Returns
    -------
    list[str]
        Имена отобранных признаков (top-N по permutation importance,
        расширенные whitelist-признаками). Клиент сам собирает X через build_xy().

    Raises
    ------
    ValueError
        Если len(df_train с не-NaN target) < 10.
    """
    from sklearn.linear_model import Ridge
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler
    from src.feature_engineering import compute_vif, _META_COLUMNS, _ALL_TARGETS

    df_work = df_train.copy()
    df_work = df_work[df_work[target].notna()].copy()

    if len(df_work) < 10:
        raise ValueError(
            f"Слишком мало строк ({len(df_work)}) для target='{target}'"
        )

    y = df_work[target].values.astype(float)

    exclude = _META_COLUMNS | _ALL_TARGETS
    candidate_cols = [
        c for c in df_work.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df_work[c])
        and df_work[c].std() > 1e-10  # NaN > 1e-10 = False → отбрасывается
    ]

    if len(candidate_cols) == 0:
        raise ValueError("Нет кандидатов для признаков после фильтрации")

    X_df = df_work[candidate_cols].copy()
    for col in candidate_cols:
        X_df[col] = X_df[col].fillna(X_df[col].median())

    vif_result = compute_vif(X_df, threshold=vif_threshold)
    low_vif_cols = vif_result.loc[vif_result["keep"], "feature"].tolist()
    if len(low_vif_cols) == 0:
        warnings.warn("VIF удалил всё; используем top-20 по VIF", stacklevel=2)
        low_vif_cols = vif_result.head(20)["feature"].tolist()

    X_vif = X_df[low_vif_cols].values

    n = len(X_vif)
    split_idx = int(n * 0.8)
    X_tr, X_val = X_vif[:split_idx], X_vif[split_idx:]
    y_tr, y_val = y[:split_idx], y[split_idx:]

    if len(X_val) < 5:
        X_tr, y_tr = X_vif, y
        X_val, y_val = X_vif, y

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(X_tr_sc, y_tr)

    perm = permutation_importance(
        ridge, X_val_sc, y_val,
        n_repeats=10,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )

    sorted_idx = np.argsort(perm.importances_mean)[::-1]
    top_idx = sorted_idx[:max_features]
    base_features = [low_vif_cols[i] for i in top_idx]

    # Whitelist-расширение (Variant Z): добавляем управляемые признаки,
    # прошедшие VIF, но не попавшие в base_features по permutation importance
    whitelist = (DEFAULT_OPERATIONAL_WHITELIST
                 if operational_whitelist is None
                 else operational_whitelist)
    whitelist_extras = [
        c for c in whitelist
        if c in low_vif_cols and c not in base_features
    ]
    return base_features + whitelist_extras


def build_xy(
    df: pd.DataFrame,
    target: str,
    feature_names: list[str],
    train_medians: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Собирает (X, y, medians) для train либо test.

    Если train_medians задан → используется для fillna (тестовая выборка).
    Если None → считаются из df и возвращаются (обучающая выборка).

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    feature_names : list[str]
    train_medians : dict | None

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, float]]
        (X, y, medians) — X shape (n_valid, len(feature_names))
    """
    df_work = df[df[target].notna()].copy()
    y = df_work[target].values.astype(float)

    if train_medians is None:
        medians = {col: float(df_work[col].median()) for col in feature_names}
    else:
        medians = dict(train_medians)

    X_df = df_work[feature_names].copy()
    for col in feature_names:
        X_df[col] = X_df[col].fillna(medians[col])

    return X_df.values.astype(float), y, medians


def fit_predict_evaluate(
    model_factory: Callable[[], object],
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    target_name: str,
    cv,
) -> dict:
    """CV на train + финальный фит на полном train + предсказание на test.

    Parameters
    ----------
    model_factory : callable () -> sklearn-compatible estimator
        Функция, возвращающая свежий несобранный оценщик.
    X_train, y_train : np.ndarray
        Обучающие данные.
    X_test, y_test : np.ndarray
        Тестовые данные (для финальной оценки).
    target_name : str
        Имя таргета (для compute_regression_metrics).
    cv : sklearn cross-validator
        Стратегия CV (KFold, TimeSeriesSplit и т.п.).

    Returns
    -------
    dict
        {model, cv_rmse_scores, cv_rmse_mean, cv_rmse_std,
         test_metrics, y_pred, residuals}
    """
    from sklearn.metrics import mean_squared_error
    from src.evaluation import compute_regression_metrics

    cv_rmse_scores: list[float] = []
    for tr_idx, va_idx in cv.split(X_train):
        m = model_factory()
        m.fit(X_train[tr_idx], y_train[tr_idx])
        pred = m.predict(X_train[va_idx])
        cv_rmse_scores.append(
            float(np.sqrt(mean_squared_error(y_train[va_idx], pred)))
        )

    final = model_factory()
    final.fit(X_train, y_train)
    y_pred = np.asarray(final.predict(X_test), dtype=float)

    test_metrics = compute_regression_metrics(
        y_pred=y_pred, y_true=y_test,
        target_name=target_name, y_train=y_train,
    )

    return {
        "model": final,
        "cv_rmse_scores": cv_rmse_scores,
        "cv_rmse_mean": float(np.mean(cv_rmse_scores)),
        "cv_rmse_std": float(np.std(cv_rmse_scores)),
        "test_metrics": test_metrics,
        "y_pred": y_pred.tolist(),
        "residuals": (y_test - y_pred).tolist(),
    }


def update_final_config(
    app_key: str,
    target: str,
    decision: dict,
    path: Optional[Union[str, Path]] = None,
) -> None:
    """Read-modify-write JSON артефакт с решениями моделирования.

    Parameters
    ----------
    app_key : str
        Ключ приложения: 'f1' или 'f2'.
    target : str
        Целевая метрика: 'stickiness', 'dau', 'retention_d7'.
    decision : dict
        Полное решение по таргету (см. spec раздел 7).
    path : str | Path | None, optional
        Путь к JSON-файлу. По умолчанию — results/final_config.json
        относительно корня проекта (родитель notebooks/).
    """
    if path is None:
        project_root = Path(__file__).parent.parent
        path = project_root / "results" / "final_config.json"
    path = Path(path)

    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        config = {
            "schema_version": "1.0",
            "generated_by": "notebooks/04_modeling_*.ipynb",
            "f1": {},
            "f2": {},
        }

    config["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config.setdefault(app_key, {})[target] = decision

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
