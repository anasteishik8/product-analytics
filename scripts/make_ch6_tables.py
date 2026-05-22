"""make_ch6_tables.py — генерирует tab6_1, tab6_2, tab6_3 (.tex) для главы 6 ВКР."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from csv_to_latex_table import df_to_latex_booktabs

TABLES_DIR = ROOT / "vkr" / "v2" / "artifacts" / "tables"


PRODUCT_LABEL = {
    "com.labpixies.flood": "Flood-It!",
    "com.google.flood2": "Flood-It! 2",
}


def build_tab6_1() -> Path:
    summary = pd.read_csv(ROOT / "results" / "viability_summary.csv")
    df = summary[summary["model"] == "M3"].copy().reset_index(drop=True)
    df["app_id"] = df["app_id"].map(PRODUCT_LABEL).fillna(df["app_id"])
    df["bad_share_pct"] = (df["bad_share_mean"] * 100).round(2)

    columns = {
        "app_id": "Продукт",
        "feature_count": "Признаков",
        "D2_mean": "Среднее D²",
        "bad_share_pct": "Доля плохих дней, %",
        "days_discontinue": "DISCONTINUE, дн.",
        "final_verdict": "Вердикт",
    }

    out = TABLES_DIR / "tab6_1.tex"
    df_to_latex_booktabs(
        df,
        out_path=out,
        columns=columns,
        caption=(
            "Сводка по двум продуктам (полная модель M3): среднее $D^2$, доля "
            "плохих дней, число дней в режиме DISCONTINUE, итоговый вердикт."
        ),
        label="tab:ch6_summary",
        float_fmt="{:.2f}",
        alignment="lrrrrl",
    )
    return out


def build_tab6_2() -> Path:
    contrib = pd.DataFrame({
        "Вклад": [
            "1. Методология scenario-aware h_safe — горизонт прогноза, "
            "определяемый не только метрикой ошибки, но и стабильностью под "
            "counterfactual-возмущениями",
            "2. Расширение системы вердиктов DEVELOP/MONITOR/DISCONTINUE с "
            "Mahalanobis-расстоянием и bad-share как сигналом неблагополучия "
            "по дням",
            "3. Сценарный анализ с BCa-bootstrap CI и SHAP-мостом — связь "
            "между прогнозом, объяснимостью и управляемыми операционными "
            "метриками (crash_rate, onboarding_completion_rate, "
            "median_session_duration)",
        ],
        "Результат": [
            "Эмпирически установлены h_safe от 0 до 24 дней по 6 парам "
            "продукт-метрика; показано что h_safe сильно различается даже "
            "внутри одного продукта",
            "На реальном кейсе F1/F2 показано: M1 (только in-app метрики) "
            "даёт DEVELOP для обоих продуктов; M3 (полная модель со "
            "Store/Market) выявляет F2 как DISCONTINUE — методологически "
            "значимый результат",
            "Сценарий cross_product_mimic для F2/DAU даёт +11.6% "
            "(BCa 95% CI [+10.3; +12.6]); SHAP подтверждает: top-драйверы — "
            "операционные и внешние факторы, не in-app",
        ],
    })

    columns = {
        "Вклад": "Вклад",
        "Результат": "Результат",
    }

    out = TABLES_DIR / "tab6_2.tex"
    df_to_latex_booktabs(
        contrib,
        out_path=out,
        columns=columns,
        caption=(
            "Три авторских вклада работы и подтверждающие результаты на "
            "реальном кейсе Flood-It! / Flood-It! 2."
        ),
        label="tab:ch6_contributions",
        alignment="p{0.42\\linewidth}p{0.50\\linewidth}",
    )
    return out


def build_tab6_3() -> Path:
    limits = pd.DataFrame({
        "Ограничение": [
            "Желательно не менее 100 дней наблюдений на продукт",
            "Доступ к Store/Market блоку признаков (не только in-app)",
            "Recursive forecast горизонт ограничен h_demo=14 дн.",
            "Не применим при структурной перестройке продукта в окне "
            "наблюдения",
            "Чувствительность к выбору operational whitelist",
        ],
        "Условие применимости": [
            "Без 100+ дней h_safe не определяется надёжно по "
            "walk-forward-процедуре",
            "На M1 (только in-app) вердикт DEVELOP для F2 ошибочный — "
            "показано через сравнение с M3",
            "Для среднесрочных горизонтов (месяцы) методика требует "
            "пересмотра baseline-модели",
            "Изменение монетизации, целевой аудитории или платформы "
            "инвалидирует обучающую выборку",
            "Whitelist crash_rate / onboarding_completion_rate / "
            "median_session_duration зафиксирован экспертно",
        ],
    })

    columns = {
        "Ограничение": "Ограничение",
        "Условие применимости": "Условие применимости",
    }

    out = TABLES_DIR / "tab6_3.tex"
    df_to_latex_booktabs(
        limits,
        out_path=out,
        columns=columns,
        caption=(
            "Ограничения предложенного метода и условия его применимости — "
            "пять основных параметров для оценки применимости системы к "
            "другим продуктам."
        ),
        label="tab:ch6_limits",
        alignment="p{0.42\\linewidth}p{0.50\\linewidth}",
    )
    return out


def main() -> None:
    t1 = build_tab6_1()
    t2 = build_tab6_2()
    t3 = build_tab6_3()
    for p in (t1, t2, t3):
        print(f"OK {p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
