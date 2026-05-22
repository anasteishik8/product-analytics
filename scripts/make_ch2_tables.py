"""make_ch2_tables.py — генерирует tab2_1, tab2_2, tab2_3, tab2_4 для главы 2."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from csv_to_latex_table import df_to_latex_booktabs

TABLES_DIR = ROOT / "vkr" / "v2" / "artifacts" / "tables"
PARQUET_PATH = ROOT / "data" / "processed" / "floodit_final.parquet"


# Группировка 60 колонок: ровно 60 в сумме.
GROUPS_60 = {
    "Служебные": ["date", "app_id", "category", "weekday"],
    "Продукт (in-app)": [
        "daily_installs", "daily_uninstalls", "net_installs",
        "dau", "mau", "wau", "stickiness",
        "retention_d1", "retention_d7", "retention_d30",
        "total_sessions", "median_session_duration", "avg_session_duration",
        "crash_rate", "onboarding_completion_rate", "peak_hour",
        "crash_rate_ma7", "avg_session_duration_ma7",
        "dau_growth_rate", "dau_growth_rate_7d", "install_growth_rate",
        "retention_d1_change", "retention_trend", "engagement_score",
    ],
    "Store / рейтинги": [
        "rating", "reviews_count", "positive", "negative", "positive_ratio",
        "sentiment_score", "reviews_30d", "positive_30d", "negative_30d",
        "sentiment_30d", "rating_ma30", "positive_ratio_ma30",
        "rating_volatility_7d", "rating_trend", "negative_change",
        "has_store_data",
    ],
    "Market / конкуренция": [
        "competitor_count", "category_rank", "competitor_count_change",
        "competition_volatility_7d", "category_rank_ma14",
        "category_rank_ma30", "rank_volatility_7d", "rank_improvement",
        "market_trend",
    ],
    "External / медиа": [
        "media_coverage", "market_sentiment", "media_coverage_ma7",
        "media_coverage_change", "seasonality_change",
        "downloads_index", "downloads_index_ma30",
    ],
}


def _verify_grouping(df: pd.DataFrame) -> None:
    flat = [c for cols in GROUPS_60.values() for c in cols]
    if len(flat) != 60:
        raise RuntimeError(f"GROUPS_60 содержит {len(flat)} колонок, ожидалось 60")
    if set(flat) != set(df.columns):
        missing = set(df.columns) - set(flat)
        extra = set(flat) - set(df.columns)
        raise RuntimeError(f"Несовпадение со схемой parquet: missing={missing}, extra={extra}")


def build_tab2_1() -> Path:
    """Источники данных и типы признаков (компактно). Полный перечень — в tab2_2."""
    sources = pd.DataFrame({
        "Источник": [
            "Firebase BigQuery",
            "Google Play Scraper",
            "Kaggle (Google Play dataset)",
            "Google Trends + Wikipedia",
        ],
        "Тип данных": [
            "In-app телеметрия + служебные",
            "Store-метрики и отзывы",
            "Категориальная конкуренция",
            "Внешний интерес",
        ],
        "Примеры признаков": [
            "dau, stickiness, retention_d1, retention_d7, crash_rate",
            "rating, reviews_count, positive_ratio, sentiment_score",
            "competitor_count, category_rank, market_trend",
            "media_coverage, downloads_index, seasonality_change",
        ],
        "Колонок": [
            len(GROUPS_60["Служебные"]) + len(GROUPS_60["Продукт (in-app)"]),
            len(GROUPS_60["Store / рейтинги"]),
            len(GROUPS_60["Market / конкуренция"]),
            len(GROUPS_60["External / медиа"]),
        ],
        "Покрытие": [
            "Полное (111 дн.)",
            "Частичное (флаг has_store_data)",
            "Частичное (зависит от категории)",
            "Полное",
        ],
    })

    columns = {
        "Источник": "Источник",
        "Тип данных": "Тип данных",
        "Примеры признаков": "Примеры признаков",
        "Колонок": "Колонок",
        "Покрытие": "Покрытие",
    }
    out = TABLES_DIR / "tab2_1.tex"
    df_to_latex_booktabs(
        sources,
        out_path=out,
        columns=columns,
        caption=(
            "Источники данных и типы признаков. Полный перечень всех 60 колонок "
            "по смысловым группам — таблица 2.2."
        ),
        label="tab:ch2_sources",
        alignment="p{0.18\\linewidth}p{0.20\\linewidth}p{0.28\\linewidth}rp{0.18\\linewidth}",
    )
    return out


def build_tab2_2() -> Path:
    """Группы признаков итогового датасета 222×60 — полный перечень всех 60 колонок."""
    df = pd.read_parquet(PARQUET_PATH)
    _verify_grouping(df)

    rows = []
    for name, cols in GROUPS_60.items():
        rows.append({
            "Группа": name,
            "Колонок": len(cols),
            "Признаки": ", ".join(cols),
        })
    rows.append({
        "Группа": "Итого",
        "Колонок": sum(len(v) for v in GROUPS_60.values()),
        "Признаки": "—",
    })
    tab = pd.DataFrame(rows)

    columns = {
        "Группа": "Группа",
        "Колонок": "Колонок",
        "Признаки": "Признаки",
    }
    out = TABLES_DIR / "tab2_2.tex"
    df_to_latex_booktabs(
        tab,
        out_path=out,
        columns=columns,
        caption=(
            "Полный перечень 60 колонок итогового датасета по смысловым "
            "группам. Смысловые группы соответствуют ablation-блокам M1/M2/M3 "
            "из главы 6: M1 = «Продукт (in-app)», M2 = M1 + «Market / конкуренция», "
            "M3 = полная модель со всеми блоками."
        ),
        label="tab:ch2_groups",
        # Признаки — широкая колонка с переносом, остальные узкие
        alignment="p{0.22\\linewidth}rp{0.62\\linewidth}",
    )
    return out


def build_tab2_3() -> Path:
    """F1 vs F2 как объекты исследования."""
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    f1 = df[df["app_id"] == "com.labpixies.flood"]
    f2 = df[df["app_id"] == "com.google.flood2"]

    def _fmt_period(d: pd.DataFrame) -> str:
        first = d["date"].min().strftime("%d.%m.%Y")
        last = d["date"].max().strftime("%d.%m.%Y")
        return f"{first} – {last}"

    def _safe_mean(s: pd.Series, fmt: str) -> str:
        if s.notna().any():
            return fmt.format(s.mean())
        return "—"

    def _stats(d: pd.DataFrame) -> dict[str, str]:
        return {
            "Дней наблюдений": str(len(d)),
            "Период": _fmt_period(d),
            "Средний DAU": _safe_mean(d["dau"].astype(float), "{:.0f}"),
            "Средний stickiness": _safe_mean(d["stickiness"], "{:.3f}"),
            "Средний retention_d7": _safe_mean(d["retention_d7"], "{:.3f}"),
            "Средний crash_rate": _safe_mean(d["crash_rate"], "{:.4f}"),
            "Средний рейтинг": _safe_mean(d["rating"], "{:.2f}"),
        }

    s1 = _stats(f1)
    s2 = _stats(f2)

    rows = [{"Показатель": "Статус",
             "Flood-It! (F1)": "4.9M установок (успех)",
             "Flood-It! 2 (F2)": "Снят с публикации"}]
    for key in s1.keys():
        rows.append({
            "Показатель": key,
            "Flood-It! (F1)": s1[key],
            "Flood-It! 2 (F2)": s2[key],
        })
    tab = pd.DataFrame(rows)

    columns = {
        "Показатель": "Показатель",
        "Flood-It! (F1)": "Flood-It! (F1)",
        "Flood-It! 2 (F2)": "Flood-It! 2 (F2)",
    }
    out = TABLES_DIR / "tab2_3.tex"
    df_to_latex_booktabs(
        tab,
        out_path=out,
        columns=columns,
        caption=(
            "Сравнение Flood-It! и Flood-It! 2 как объектов исследования: "
            "исторический контекст и средние значения ключевых метрик за "
            "период наблюдений."
        ),
        label="tab:ch2_products",
        alignment="p{0.32\\linewidth}p{0.30\\linewidth}p{0.30\\linewidth}",
    )
    return out


def build_tab2_4() -> Path:
    """Целевые метрики прогнозирования."""
    targets = pd.DataFrame({
        "Метрика": ["stickiness", "DAU", "retention_d7"],
        "Формула / источник": [
            "DAU / MAU",
            "Daily Active Users (Firebase)",
            "Доля пользователей, вернувшихся на 7-й день",
        ],
        "Диапазон": ["0 – 1", "> 0 (целое)", "0 – 1"],
        "CV-стратегия": [
            "KFold(5, shuffle=True, seed=42)",
            "TimeSeriesSplit(5)",
            "TimeSeriesSplit(5)",
        ],
        "Основная метрика": [
            "MAPE",
            "RMSE / MASE",
            "RMSE / MASE (нули возможны)",
        ],
    })

    columns = {
        "Метрика": "Метрика",
        "Формула / источник": "Формула / источник",
        "Диапазон": "Диапазон",
        "CV-стратегия": "CV-стратегия",
        "Основная метрика": "Основная метрика",
    }
    out = TABLES_DIR / "tab2_4.tex"
    df_to_latex_booktabs(
        targets,
        out_path=out,
        columns=columns,
        caption=(
            "Целевые метрики прогнозирования: семантика, шкала, стратегия "
            "кросс-валидации и основная метрика качества."
        ),
        label="tab:ch2_targets",
        alignment="p{0.16\\linewidth}p{0.28\\linewidth}p{0.13\\linewidth}p{0.22\\linewidth}p{0.18\\linewidth}",
    )
    return out


def main() -> None:
    t1 = build_tab2_1()
    t2 = build_tab2_2()
    t3 = build_tab2_3()
    t4 = build_tab2_4()
    for p in (t1, t2, t3, t4):
        print(f"OK {p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
