"""
make_ch3_tables.py — генерация LaTeX-таблиц tab3_1 и tab3_2 для главы 3 ВКР.

tab3_1 — Методы и их роль в системе (9 строк × 3 колонки).
tab3_2 — Метрики качества прогноза и критерии интерпретации (5 строк × 3 колонки).

Usage:
    python scripts/make_ch3_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from csv_to_latex_table import df_to_latex_booktabs  # noqa: E402


OUT_DIR = ROOT / "vkr" / "v2" / "artifacts" / "tables"


def build_tab3_1() -> Path:
    methods = pd.DataFrame({
        "method": [
            "Ridge",
            "Nadaraya–Watson",
            "LocalLinear",
            "RandomForest",
            "XGBoost / kNN",
            "Recursive forecast",
            "Bootstrap / Monte Carlo",
            "SHAP",
            "Mahalanobis + bad_share",
        ],
        "type": [
            "Параметрическая модель-кандидат",
            "Непараметрическая модель-кандидат",
            "Непараметрическая модель-кандидат",
            "Ансамблевая модель-кандидат",
            "Дополнительные кандидаты, опорное сравнение",
            "Шаг расчёта",
            "Шаг расчёта",
            "Шаг расчёта",
            "Шаг расчёта",
        ],
        "role": [
            "Регуляризованная линейная модель прогноза",
            "Локальная нелинейная модель прогноза",
            "Уточняет NW поправкой на локальный тренд",
            "Нелинейная модель, устойчивая к шуму",
            "Опорное сравнение с градиентным бустингом и методом ближайших соседей",
            "Прогноз на 14 дней без новых наблюдений факта",
            "Доверительные интервалы прогноза и распределения сценарных эффектов",
            "Объяснение модели через вклад признаков; вход для сценарного анализа",
            "Итоговый вердикт DEVELOP / MONITOR / DISCONTINUE по аномальности дней",
        ],
    })

    out_path = OUT_DIR / "tab3_1.tex"
    caption = (
        "Методы, применяемые в системе. Первые шесть строк — модели-кандидаты, "
        "среди которых рабочая модель выбирается эмпирически для каждой пары "
        "продукт-метрика. Остальные строки — шаги расчёта от прогноза до вердикта."
    )
    df_to_latex_booktabs(
        methods,
        out_path,
        columns={
            "method": "Метод",
            "type": "Тип",
            "role": "Роль в системе",
        },
        caption=caption,
        label="tab:ch3_methods",
        alignment="p{3.2cm}p{4.5cm}p{6.5cm}",
    )
    return out_path


def build_tab3_2() -> Path:
    metrics = pd.DataFrame({
        "metric": [
            "RMSE",
            "MAPE",
            "MASE",
            "Direction Accuracy",
            "CI95 coverage",
        ],
        "what": [
            "Среднеквадратическая ошибка прогноза в исходной шкале",
            "Средняя абсолютная относительная ошибка",
            "Отклонение прогноза, нормированное на наивный прогноз («завтра как сегодня»)",
            "Доля шагов, на которых направление изменения предсказано верно",
            "Доля факта, попавшая в построенный 95% доверительный интервал",
        ],
        "when": [
            "Базовая метрика при ненулевых значениях факта и прогноза",
            "Сравнение продуктов с разными шкалами целевой величины",
            "Сравнение модели с наивным прогнозом",
            "Качество прогноза тренда, а не уровня",
            "Калибровка доверительных интервалов",
        ],
    })

    out_path = OUT_DIR / "tab3_2.tex"
    caption = (
        "Метрики качества прогноза, по которым сравниваются модели-кандидаты. "
        "Финальный выбор рабочей модели для каждой пары продукт-метрика делается "
        "по recursive forecast validation на горизонте 14 дней. "
        "Полные формулы — приложение А."
    )
    df_to_latex_booktabs(
        metrics,
        out_path,
        columns={
            "metric": "Метрика",
            "what": "Что считает",
            "when": "Когда применять",
        },
        caption=caption,
        label="tab:ch3_metrics",
        alignment="p{3cm}p{6cm}p{5cm}",
    )
    return out_path


def main() -> None:
    p1 = build_tab3_1()
    p2 = build_tab3_2()
    print(f"OK tab3_1 -> {p1}")
    print(f"OK tab3_2 -> {p2}")


if __name__ == "__main__":
    main()
