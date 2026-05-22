"""make_ch4_tables.py — генерирует tab4_1, tab4_2, tab4_3, tab4_4 (.tex) для главы 4 ВКР.

tab4_1 — технологический стек (requirements.txt)
tab4_2 — назначение модулей src/
tab4_3 — состав results/ (16 выходных файлов, 5 фаз)
tab4_4 — тестирование и воспроизводимость (189 тестов)
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from csv_to_latex_table import df_to_latex_booktabs

TABLES_DIR = ROOT / "vkr" / "v2" / "artifacts" / "tables"


# === Общие плейсхолдеры для math/underscore-защиты ===
_PH_USC_ESC = "\x00UE\x00"   # уже эскейпленный underscore (\_) вне math
_PH_MATH = "\x00MA\x00"
_PH_BSL = "\x00BS\x00"
_PH_LBR = "\x00LB\x00"
_PH_RBR = "\x00RB\x00"
_PH_USC = "\x00US\x00"        # обычный _ внутри math
_PH_CRT = "\x00CR\x00"        # ^ внутри math


def _encode_protect(s: str) -> str:
    """Защищает math ($...$), \\macro{...} и уже-эскейпленные \\_ от _escape_latex."""
    import re

    def repl(m):
        inner = m.group(0)
        inner = inner.replace("\\", _PH_BSL)
        inner = inner.replace("{", _PH_LBR).replace("}", _PH_RBR)
        inner = inner.replace("_", _PH_USC)
        inner = inner.replace("^", _PH_CRT)
        inner = inner.replace("$", _PH_MATH)
        return inner

    # 1) math: $...$
    s = re.sub(r"\$[^$]+\$", repl, s)
    # 2) \macro{...} (latex command с аргументом). Допускается до 80 символов внутри {}.
    s = re.sub(r"\\[a-zA-Z]+\{[^{}]{0,80}\}", repl, s)
    # 3) уже-эскейпленные \_  → плейсхолдер
    s = s.replace("\\_", _PH_USC_ESC)
    return s


def _decode_protect(text: str) -> str:
    text = text.replace(_PH_USC_ESC, "\\_")
    text = text.replace(_PH_MATH, "$")
    text = text.replace(_PH_BSL, "\\")
    text = text.replace(_PH_LBR, "{")
    text = text.replace(_PH_RBR, "}")
    text = text.replace(_PH_USC, "_")
    text = text.replace(_PH_CRT, "^")
    return text


def _apply_protect(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].map(_encode_protect)
    return df


def build_tab4_1() -> Path:
    """Технологический стек: 5 категорий × библиотеки × назначение."""
    stack = pd.DataFrame({
        "Категория": [
            "Обработка данных",
            "Машинное обучение",
            "Сбор данных",
            "Web/UI",
            "Тесты",
        ],
        "Библиотеки": [
            "pandas >=2.0, numpy >=1.24, pyarrow >=14.0 (parquet), scipy >=1.11",
            "scikit-learn >=1.3, xgboost >=2.0, shap >=0.44, joblib >=1.3",
            "google-play-scraper >=1.2, pytrends >=4.9, requests >=2.31 (Wikipedia)",
            "streamlit >=1.30, plotly >=5.18, matplotlib >=3.8, seaborn >=0.13",
            "pytest >=8.0",
        ],
        "Назначение": [
            "Загрузка parquet, аналитика, агрегации",
            "Регрессионные модели, бустинг, объяснимость, сериализация",
            "Парсинг Google Play, Trends API, Wikipedia Pageviews",
            "Демо-интерфейс, интерактивные и статические графики",
            "189 unit/integration тестов",
        ],
    })
    columns = {
        "Категория": "Категория",
        "Библиотеки": "Библиотеки",
        "Назначение": "Назначение",
    }
    out = TABLES_DIR / "tab4_1.tex"
    df_to_latex_booktabs(
        stack,
        out_path=out,
        columns=columns,
        caption=(
            "Технологический стек: Python 3.12 + перечисленные библиотеки. "
            "Минимальные версии зафиксированы в \\texttt{requirements.txt}; "
            "точные версии — в \\texttt{requirements.lock}."
        ),
        label="tab:ch4_stack",
        alignment="p{0.20\\linewidth}p{0.40\\linewidth}p{0.32\\linewidth}",
    )
    return out


def build_tab4_2() -> Path:
    """Назначение модулей src/ — 4 строки."""
    modules = pd.DataFrame({
        "Модуль": [
            "kernel_regression.py",
            "feature_engineering.py",
            "evaluation.py",
            "scenario_analysis.py",
        ],
        "Ключевые сущности": [
            "NadarayaWatsonRegressor, LocalLinearRegressor, "
            "gaussian/epanechnikov/triangular_kernel",
            "load_dataset, compute_vif, select_features, "
            "normalize_features, get_cv_strategy",
            "compute_regression_metrics, diebold_mariano_test, compare_models",
            "ScenarioAnalyzer (run_scenario, bca_bootstrap_ci), "
            "define_scenarios_floodit",
        ],
        "Назначение": [
            "Ядерная регрессия (NW + LL) и базовые ядра; "
            "ширина окна — KFold CV по log-сетке",
            "Загрузка \\texttt{floodit\\_final.parquet}, VIF-фильтрация, "
            "отбор признаков, CV-стратегии",
            "Метрики качества (RMSE/MAE/MAPE/sMAPE/$R^2$/MASE), "
            "тест Дибольда--Мариано",
            "Monte Carlo сценарный анализ ($N=1000$), "
            "BCa bootstrap CI (Efron--Tibshirani)",
        ],
    })
    columns = {
        "Модуль": "Модуль",
        "Ключевые сущности": "Ключевые сущности",
        "Назначение": "Назначение",
    }
    modules = _apply_protect(modules, ["Назначение"])
    out = TABLES_DIR / "tab4_2.tex"
    df_to_latex_booktabs(
        modules,
        out_path=out,
        columns=columns,
        caption=(
            "Назначение модулей \\texttt{src/} --- математическое ядро системы. "
            "Все 4 модуля покрыты unit-тестами, импортируются ноутбуками и "
            "Streamlit-демо."
        ),
        label="tab:ch4_modules",
        alignment="p{0.20\\linewidth}p{0.38\\linewidth}p{0.34\\linewidth}",
    )
    out.write_text(_decode_protect(out.read_text(encoding="utf-8")), encoding="utf-8")
    return out


def build_tab4_3() -> Path:
    """Состав results/ — 5 фаз вычислительного пайплайна."""
    artifacts = pd.DataFrame({
        "Фаза": [
            "04. Моделирование",
            "05. Горизонт прогноза",
            "06b. Прогноз",
            "07. Сценарии",
            "08. Жизнеспособность",
        ],
        "Файлы": [
            "final\\_config.json, forecast\\_validation\\_summary.csv, "
            "forecast\\_validation\\_comparison.csv",
            "horizon\\_config.json, horizon\\_safe.csv, "
            "horizon\\_curve\\_f1\\_dau.csv",
            "forecast\\_validation\\_recursive\\_summary.csv, "
            "demo\\_forecast\\_predictions.csv",
            "scenario\\_analysis\\_summary.csv, scenario\\_recommendations.csv, "
            "scenario\\_avoid\\_actions.csv, scenario\\_shap\\_summary.csv, "
            "mc\\_f2\\_dau\\_cross\\_product\\_mimic.csv",
            "viability\\_summary.csv, viability\\_scores.csv, "
            "viability\\_decomposition.csv",
        ],
        "Назначение": [
            "Лучшая модель для пары (приложение, целевая метрика) + метрики кросс-валидации",
            "Безопасный горизонт прогноза $h_{\\text{safe}}$ + "
            "детальная кривая для F1/DAU",
            "Рекурсивный 14-дневный прогноз + предсказания по дням для демо",
            "Сценарии Монте-Карло ($N=1000$), SHAP-объяснения, "
            "рекомендации и анти-рекомендации",
            "Декомпозиция M1/M2/M3, $D^2$ по дням, итоговые карточки вердикта",
        ],
    })
    columns = {
        "Фаза": "Фаза",
        "Файлы": "Файлы",
        "Назначение": "Назначение",
    }
    artifacts = _apply_protect(artifacts, ["Файлы", "Назначение"])
    out = TABLES_DIR / "tab4_3.tex"
    df_to_latex_booktabs(
        artifacts,
        out_path=out,
        columns=columns,
        caption=(
            "Состав \\texttt{results/} --- 16 выходных файлов вычислительного "
            "пайплайна, единый источник данных для всех таблиц и рисунков "
            "диплома. Каждый файл создаётся одним ноутбуком (фазы 04..08) "
            "и не модифицируется вручную."
        ),
        label="tab:ch4_artifacts",
        alignment="p{0.16\\linewidth}p{0.47\\linewidth}p{0.32\\linewidth}",
    )
    out.write_text(_decode_protect(out.read_text(encoding="utf-8")), encoding="utf-8")
    return out


def build_tab4_4() -> Path:
    """Тестирование и воспроизводимость — 4 категории, 189 тестов."""
    tests = pd.DataFrame({
        "Категория тестов": [
            "Математические примитивы",
            "Pipeline + бизнес-логика",
            "Chapter-pipeline tooling",
            "Export + UI",
        ],
        "Тестовые файлы": [
            "test\\_kernel\\_regression.py, test\\_feature\\_engineering.py, "
            "test\\_evaluation.py, test\\_scenario\\_analysis.py",
            "test\\_modeling\\_helpers.py, test\\_what\\_if\\_engine.py, "
            "test\\_winner\\_registry.py, test\\_streamlit\\_consistency.py",
            "test\\_make\\_registry\\_skeleton.py, test\\_check\\_chapter.py, "
            "test\\_csv\\_to\\_latex\\_table.py, test\\_freeze\\_artifact.py, "
            "test\\_vkr\\_plot\\_style.py",
            "test\\_export\\_demo\\_artifacts.py, test\\_plots.py, test\\_text.py",
        ],
        "Что проверяется": [
            "Корректность NW/LL/ядер, VIF, select\\_features, "
            "метрики, DM-test, BCa CI",
            "End-to-end forecast + scenario + winner-registry invariants + "
            "Streamlit $\\leftrightarrow$ pipeline консистентность",
            "Утилиты сборки диплома (registry, gate-проверки главы, "
            "LaTeX-генерация, единый стиль)",
            "Файлы для демо, plot-функции, текстовые описания",
        ],
    })
    columns = {
        "Категория тестов": "Категория тестов",
        "Тестовые файлы": "Тестовые файлы",
        "Что проверяется": "Что проверяется",
    }
    tests = _apply_protect(tests, ["Тестовые файлы", "Что проверяется"])
    out = TABLES_DIR / "tab4_4.tex"
    df_to_latex_booktabs(
        tests,
        out_path=out,
        columns=columns,
        caption=(
            "Тестирование: 189 unit/integration тестов "
            "(\\texttt{pytest -v}). Воспроизводимость: фиксированный "
            "\\texttt{seed=42}, входные данные неизменны "
            "(\\texttt{data/processed/floodit\\_final.parquet}), "
            "результаты регенерируются командами "
            "\\texttt{python -m nbconvert --execute --inplace}."
        ),
        label="tab:ch4_tests",
        alignment="p{0.22\\linewidth}p{0.42\\linewidth}p{0.30\\linewidth}",
    )
    out.write_text(_decode_protect(out.read_text(encoding="utf-8")), encoding="utf-8")
    return out


def main() -> None:
    t1 = build_tab4_1()
    t2 = build_tab4_2()
    t3 = build_tab4_3()
    t4 = build_tab4_4()
    for p in (t1, t2, t3, t4):
        print(f"OK {p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
