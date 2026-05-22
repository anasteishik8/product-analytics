"""Pure helper — выбор winner-модели для пары (app, target).

Этот модуль — единственный разрешённый путь выбора winner-модели в коде демо.
Используется exporter (scripts/export_demo_artifacts.py), WhatIfEngine
(app/streamlit/lib/what_if_engine.py) и тестами consistency.

Логика идентична notebook 07: предпочесть строки с beats_naive=True и
warning=="OK", среди них — минимальный RMSE; иначе — min RMSE среди
beats_naive=True; иначе — WhatIfUnavailable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class WhatIfUnavailable(Exception):
    """Поднимается, когда для пары (app, target) нет ни одной приемлемой модели."""

    def __init__(self, app: str, target: str) -> None:
        super().__init__(
            f"No winner model available for (app={app}, target={target}). "
            "Check forecast_validation_recursive_summary.csv."
        )
        self.app = app
        self.target = target


def resolve_winner_model(app: str, target: str, summary_path: Path) -> str:
    """Возвращает имя winner-модели для пары (app, target).

    Parameters
    ----------
    app : {"f1", "f2"}
    target : {"stickiness", "dau", "retention_d7"}
    summary_path : Path
        Путь к results/forecast_validation_recursive_summary.csv.

    Returns
    -------
    str
        Имя модели — одно из: "Ridge", "RandomForest", "NadarayaWatson",
        "LocalLinear", "XGBoost", "kNN".

    Raises
    ------
    WhatIfUnavailable
        Если ни одна строка не подходит даже под мягкий критерий.
    """
    df = pd.read_csv(summary_path)
    sub = df[(df["app"] == app) & (df["target"] == target)]
    sub_ok = sub[(sub["beats_naive"] == True) & (sub["warning"] == "OK")]  # noqa: E712
    if len(sub_ok) > 0:
        return str(sub_ok.sort_values("RMSE").iloc[0]["model"])
    sub_beats = sub[sub["beats_naive"] == True]  # noqa: E712
    if len(sub_beats) > 0:
        return str(sub_beats.sort_values("RMSE").iloc[0]["model"])
    raise WhatIfUnavailable(app, target)
