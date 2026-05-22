"""Build notebooks/03_comparison_floodit_vs_floodit2.ipynb using nbformat."""
import nbformat
from pathlib import Path

nb = nbformat.v4.new_notebook()
cells = []


def md(src):
    return nbformat.v4.new_markdown_cell(src)


def code(src):
    return nbformat.v4.new_code_cell(src)


# Title
cells.append(md("# 03 — Comparison: Flood-It! vs Flood-It! 2\n\n"
                "Финальный EDA-ноутбук в серии: задаёт нарратив «успех vs провал» на реальных цифрах."))

# Section 0
cells.append(md("""\
## 0. Постановка

**Зачем сравниваем.** Серия EDA началась с разрезов по источникам данных
(`01_source_*`), затем перешла к двум приложениям отдельно (`02_eda_floodit`,
`02_eda_floodit2`). Этот ноутбук — синтез: что именно отличает успешный продукт
(Flood-It!, 4.9M установок, в публикации) от провального (Flood-It! 2, снят с
Google Play)?

**Что хотим показать на цифрах.**

1. По 12 ключевым продуктовым метрикам — насколько медианы f2 отстают от f1.
2. На overlay time-series — где видно «расходящиеся траектории».
3. На первых 30 днях — какие метрики уже различают продукты раньше, чем
   накопилась достаточная история.

**Какие метрики — ранние индикаторы провала.**
Гипотеза, которую проверим в §6:

- **onboarding_completion_rate** — воронка первых сессий (0.46 у f1 vs 0.26 у
  f2 на полном горизонте) различает продукты с первого дня.
- **DAU CV (std/median)** — высокая волатильность аудитории f2 (CV ≈ 0.9 vs
  0.29 у f1) — признак, что аудитория не стабилизируется.
- **stickiness** в первые 30 дней резко проседает у f2.

Эти выводы повлияют на постановку ML-задачи: для бинарной классификации
«успех/провал» нужны ранние Product-метрики, а не накопленные Store/Market.
"""))

# Imports
cells.append(code("""\
import sys
from pathlib import Path

_HERE = Path.cwd()
_ROOT = _HERE.parent if _HERE.name == "notebooks" else _HERE
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from src.feature_engineering import load_dataset
from notebooks._common import (
    APP_PALETTE, APP_LABEL_MAP, FIGSIZE_DEFAULT, FIGSIZE_WIDE,
    save_figure, theme_setup, app_label,
    split_by_app, comparison_table, add_app_legend, annotate_sample_size,
    figures_dir,
)

theme_setup()
np.random.seed(42)

NOTEBOOK_SLUG = "03_comparison"
df = load_dataset()
df_f1, df_f2 = split_by_app(df)

print(f"Полный датасет: {df.shape}")
print(f"  f1 (Flood-It!): {df_f1.shape}")
print(f"  f2 (Flood-It! 2): {df_f2.shape}")
print(f"  Период: {df['date'].min().date()} → {df['date'].max().date()}")
"""))

# Source partition
cells.append(code("""\
# Source partition (повторяет 02_eda_floodit/floodit2 для согласованности)
PRODUCT_COLS = [
    "daily_installs", "daily_uninstalls", "net_installs",
    "dau", "mau", "wau", "stickiness",
    "retention_d1", "retention_d7", "retention_d30",
    "total_sessions", "median_session_duration", "avg_session_duration",
    "crash_rate", "onboarding_completion_rate", "peak_hour",
    "crash_rate_ma7", "avg_session_duration_ma7",
    "dau_growth_rate", "dau_growth_rate_7d", "install_growth_rate",
    "retention_d1_change", "retention_trend",
    "engagement_score",
]

STORE_COLS = [
    "rating", "reviews_count",
    "positive", "negative", "positive_ratio", "sentiment_score",
    "reviews_30d", "positive_30d", "negative_30d", "sentiment_30d",
    "rating_ma30", "positive_ratio_ma30",
    "rating_volatility_7d", "rating_trend", "negative_change",
    "downloads_index", "downloads_index_ma30",
]

MARKET_COLS = [
    "competitor_count", "category_rank",
    "competitor_count_change", "competition_volatility_7d",
    "category_rank_ma14", "category_rank_ma30",
    "rank_volatility_7d", "rank_improvement",
]

EXTERNAL_COLS = [
    "market_trend", "media_coverage",
    "market_sentiment", "media_coverage_ma7", "media_coverage_change",
    "seasonality_change",
]

print(f"Product: {len(PRODUCT_COLS)} cols")
print(f"Store:   {len(STORE_COLS)} cols")
print(f"Market:  {len(MARKET_COLS)} cols")
print(f"External:{len(EXTERNAL_COLS)} cols")
print(f"Total:   {len(PRODUCT_COLS) + len(STORE_COLS) + len(MARKET_COLS) + len(EXTERNAL_COLS)} cols")
"""))

# Section 1
cells.append(md("""\
## 1. Side-by-side описательные статистики (по 4 группам источников)

Раскладываем `describe()` по двум приложениям отдельно для каждой
группы источников. Это даёт первый визуальный slice «где данные есть, где нет».
"""))

# 1.1 Product
cells.append(md("""\
### 1.1. Product (24 колонки)

Telemetry: DAU, MAU, retention, sessions, crashes, engagement.
Источник присутствует у обоих приложений на полную глубину 111 дней.
"""))

cells.append(code("""\
print("=== Flood-It! (f1, успех) ===")
display(df_f1[PRODUCT_COLS].describe().T.round(3))
"""))

cells.append(code("""\
print("=== Flood-It! 2 (f2, провал) ===")
display(df_f2[PRODUCT_COLS].describe().T.round(3))
"""))

# 1.2 Store
cells.append(md("""\
### 1.2. Store (17 колонок)

Google Play Store метрики: rating, reviews, sentiment.
**Особенность:** для f2 store-данные отсутствуют (приложение снято с публикации
до момента сбора снапшотов), поэтому показываем только f1.
"""))

cells.append(code("""\
print("=== Flood-It! (f1) ===")
display(df_f1[STORE_COLS].describe().T.round(3))

f2_store_nonnull = df_f2[STORE_COLS].notna().sum().sum()
print(f"\\n=== Flood-It! 2 (f2) ===")
print(f"Все 17 store-колонок: {f2_store_nonnull} non-null значений из {len(df_f2) * 17}")
print("→ см. notebooks/02_eda_floodit2.ipynb §3 для разбора отсутствия Store-данных у f2")
"""))

# 1.3 Market
cells.append(md("""\
### 1.3. Market (8 колонок, идентичны для обоих приложений)

Колонки описывают категорию Puzzle (общую для обоих продуктов), поэтому
значения идентичны. Одна таблица.
"""))

cells.append(code("""\
# Sanity check: Market-колонки должны совпадать
from notebooks._common import assert_shared_columns
try:
    assert_shared_columns(df_f1, df_f2, MARKET_COLS)
    print("✓ Все 8 Market-колонок идентичны для f1 и f2")
except AssertionError as e:
    print(f"⚠ Расхождение: {e}")

display(df_f1[MARKET_COLS].describe().T.round(3))
"""))

# 1.4 External
cells.append(md("""\
### 1.4. External (6 колонок)

5 колонок (market_trend, media_coverage, market_sentiment,
media_coverage_ma7, media_coverage_change) — общий рыночный фон, идентичны.
`seasonality_change` — app-specific (зависит от sin/cos сезонной декомпозиции
по конкретному ряду DAU).
"""))

cells.append(code("""\
shared_external = ["market_trend", "media_coverage", "market_sentiment",
                   "media_coverage_ma7", "media_coverage_change"]
try:
    assert_shared_columns(df_f1, df_f2, shared_external)
    print(f"✓ {len(shared_external)} External-колонок идентичны для f1 и f2")
except AssertionError as e:
    print(f"⚠ Расхождение: {e}")

print("\\n=== Общие External (одинаковы у обоих) ===")
display(df_f1[shared_external].describe().T.round(3))

print("\\n=== seasonality_change (app-specific) ===")
seas_compare = pd.concat([
    df_f1["seasonality_change"].describe().rename("f1"),
    df_f2["seasonality_change"].describe().rename("f2"),
], axis=1).round(4)
display(seas_compare)
"""))

# Section 2
cells.append(md("""\
## 2. Δ-таблица медиан по 12 ключевым метрикам

Используем helper `comparison_table` для расчёта `f1_median`, `f2_median`,
`delta_pct` (относительная разница f2 vs f1). Сортируем по |Δ%|.

**Интерпретация знака Δ%:**

- `Δ% < 0` — f2 отстаёт от f1 (хуже).
- `Δ% > 0` — f2 опережает f1.

Но «опережение» по retention_d7/d30 у f2 — артефакт малой удерживающейся
аудитории (см. §7); по абсолютным числам f2 проигрывает.
"""))

cells.append(code("""\
KEY_METRICS_FOR_COMPARISON = [
    "dau", "mau", "stickiness",
    "retention_d1", "retention_d7", "retention_d30",
    "crash_rate", "onboarding_completion_rate",
    "total_sessions", "avg_session_duration",
    "engagement_score", "daily_installs",
]

delta_df = comparison_table(df_f1, df_f2, KEY_METRICS_FOR_COMPARISON)
delta_df["direction"] = delta_df["delta_pct"].apply(
    lambda x: "↑ f2 выше" if x is not None and x > 0
    else ("↓ f2 ниже" if x is not None and x < 0 else "—")
)
delta_df_sorted = delta_df.sort_values(
    "delta_pct", key=lambda s: s.abs(), ascending=False
).reset_index(drop=True)
display(delta_df_sorted)
"""))

# Section 3
cells.append(md("""\
## 3. Что общее, что различается

Резюме структурного анализа из §1:

| Группа | f1 | f2 | Идентичность | Источник |
|---|---|---|---|---|
| Product | 24 cols, 100% complete | 24 cols, 100% complete | Различаются | Firebase telemetry, app-specific |
| Store | 17 cols, 100% complete | 17 cols, **0 non-null** | n/a (нет данных у f2) | Google Play Scraper |
| Market | 8 cols, 100% complete | 8 cols, **identical to f1** | Полная (sanity check) | Kaggle Puzzle category |
| External (5 общих) | identical to f2 | identical to f1 | Полная | Google Trends, Wikipedia |
| External seasonality_change | app-specific | app-specific | Различается | sin/cos декомпозиция DAU |

**Выводы:**

1. **Различающие источники:** Product (24 cols) + Store (17 cols, но только у
   f1) + seasonality_change. Итого ≤ 42 cols фактически разделяющих сигнала.
2. **Не-различающие:** Market (8) + 5 общих External = 13 колонок описывают
   рыночный фон, не разделяют продукты.
3. **Структурный сигнал в самом отсутствии Store у f2:** factный признак
   «снят с Google Play» — самый сильный возможный предиктор провала, но он
   измерим только постфактум.
4. **Cross-source counts (см. 02_eda_floodit/floodit2 §6):** ~250 пар
   |corr|>0.5 у f1 против ~75 у f2 — у f1 системнее структура зависимостей.
"""))

# Section 4
cells.append(md("""\
## 4. Overlay time-series ключевых метрик

6 subplots: DAU, MAU, stickiness, retention_d7, crash_rate, engagement_score.
На каждом — две линии (f1 синий = успех, f2 красный = провал) на едином
горизонте 111 дней.
"""))

cells.append(code("""\
overlay_metrics = ["dau", "mau", "stickiness",
                   "retention_d7", "crash_rate", "engagement_score"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, m in enumerate(overlay_metrics):
    ax = axes.flat[i]
    sns.lineplot(data=df, x="date", y=m, hue="app_id",
                 palette=APP_PALETTE, ax=ax, legend=False)
    annotate_sample_size(ax, df_f1, df_f2, m)
    ax.set_xlabel("")
    ax.set_ylabel(m)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
add_app_legend(fig)
plt.tight_layout(rect=(0, 0, 1, 0.96))
save_figure("overlay_timeseries_6", NOTEBOOK_SLUG)
plt.show()
"""))

cells.append(md("""\
**Что видно на overlay:**

- **DAU:** f1 держится в коридоре ~200–400; f2 болтается в широком диапазоне с
  частыми просадками — высокая волатильность.
- **MAU:** f2 формально близок к f1, но с большей дисперсией.
- **stickiness:** на длинной дистанции значения сопоставимы; разница
  проявляется в первые 30 дней (см. §6).
- **retention_d7:** для f2 значения часто выше, но это артефакт малой и
  нерепрезентативной выборки удерживающейся аудитории.
- **crash_rate:** оба продукта существенно выше порога Google Play (2%);
  у f1 даже выше медиана, что делает crash_rate **слабым** разделителем.
- **engagement_score:** медианы близки; различает плохо.
"""))

# Section 5
cells.append(md("""\
## 5. Bar chart Δ-медиан

Топ-12 метрик по абсолютной |Δ%|. Цвет:
- 🔴 красный — Δ% < 0 (медиана f2 ниже f1);
- 🟢 зелёный — Δ% > 0 (медиана f2 выше f1).
"""))

cells.append(code("""\
fig, ax = plt.subplots(figsize=(10, 7))
delta_top = delta_df.dropna(subset=["delta_pct"]).copy()
delta_top["abs_pct"] = delta_top["delta_pct"].abs()
delta_top = delta_top.sort_values("abs_pct", ascending=False).head(12)
delta_top = delta_top.sort_values("delta_pct")  # для красивого barh

colors = ["#ef4444" if d < 0 else "#10b981" for d in delta_top["delta_pct"]]
ax.barh(delta_top["metric"], delta_top["delta_pct"], color=colors)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Δ% медианы (f2 относительно f1)")
ax.set_title("Δ-медианы: f2 vs f1 для 12 ключевых метрик")

# Подписи значений
for i, (m, v) in enumerate(zip(delta_top["metric"], delta_top["delta_pct"])):
    ax.text(v + (1.5 if v >= 0 else -1.5), i, f"{v:+.1f}%",
            va="center", ha="left" if v >= 0 else "right", fontsize=9)

plt.tight_layout()
save_figure("delta_medians_bar", NOTEBOOK_SLUG)
plt.show()
"""))

# Section 6
cells.append(md("""\
## 6. Cross-source: предикторы провала на ранних данных

Подвыборка — первые 30 дней наблюдений. Гипотеза: даже на этом коротком окне
видны различия, которые система прогнозирования сможет уловить.
"""))

cells.append(code("""\
df_early = df[df["date"] <= df["date"].min() + pd.Timedelta(days=30)].copy()
n_f1_early = (df_early["app_id"] == "com.labpixies.flood").sum()
n_f2_early = (df_early["app_id"] == "com.google.flood2").sum()
print(f"Первые 30 дней: {len(df_early)} строк "
      f"({n_f1_early} f1 / {n_f2_early} f2)")

early_metrics = ["dau", "stickiness", "retention_d7", "crash_rate",
                 "onboarding_completion_rate", "engagement_score"]

# Numerical comparison table for early window
e_f1 = df_early[df_early["app_id"] == "com.labpixies.flood"]
e_f2 = df_early[df_early["app_id"] == "com.google.flood2"]
early_compare = comparison_table(e_f1, e_f2, early_metrics)
early_compare["direction"] = early_compare["delta_pct"].apply(
    lambda x: "↑ f2 выше" if x is not None and x > 0
    else ("↓ f2 ниже" if x is not None and x < 0 else "—")
)
display(early_compare.sort_values("delta_pct", key=lambda s: s.abs(),
                                    ascending=False).reset_index(drop=True))
"""))

cells.append(code("""\
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, m in enumerate(early_metrics):
    ax = axes.flat[i]
    sns.violinplot(data=df_early, x="app_id", y=m, hue="app_id",
                   palette=APP_PALETTE, legend=False, ax=ax, inner="quartile")
    ax.set_title(f"{m} (первые 30 дней)")
    ax.set_xticklabels([app_label(t.get_text()) for t in ax.get_xticklabels()],
                        rotation=15)
    ax.set_xlabel("")
plt.tight_layout()
save_figure("early_predictors", NOTEBOOK_SLUG)
plt.show()
"""))

cells.append(md("""\
**Ранние индикаторы (первые 30 дней):**

- `onboarding_completion_rate`: 0.64 (f1) vs 0.11 (f2) — **самый сильный**
  ранний разрыв (Δ ≈ −83%).
- `dau`: 281 (f1) vs 169 (f2) — медиана f2 на 40% ниже уже на старте.
- `stickiness`: 0.17 (f1) vs 0.07 (f2) — разрыв ≈ 2.4× ещё до накопления
  длинной истории.
- `crash_rate`, `engagement_score`: на 30-дневном окне различают слабо.
- `retention_d7`: смещён в сторону f2 (как и на полном горизонте) — артефакт
  малой выборки в первый месяц, не индикатор успеха.
"""))

# Section 7
cells.append(md("""\
## 7. Главный вывод

**Почему Flood-It! живёт, а Flood-It! 2 умер — на 7 пунктах с реальными
числами:**

1. **Onboarding-воронка обвалилась.** Медиана onboarding_completion_rate у f1 =
   **0.46**, у f2 = **0.25** (Δ = −44.8%). На первых 30 днях разрыв ещё резче:
   **0.64 vs 0.11**. Большинство новых пользователей f2 не доходили до первой
   полноценной сессии.

2. **DAU нестабильный.** Coefficient of variation (CV = std/median):
   **0.286** у f1 vs **0.895** у f2 — аудитория f2 болтается в три раза
   шире относительно медианы. Признак, что продукт не нашёл устойчивую базу
   повторных пользователей.

3. **Объёмы установок и сессий ниже.** Daily installs медиана **26 (f1)** vs
   **11 (f2)** (Δ = −57.7%). Total_sessions: **350 vs 277** (Δ = −20.9%).
   Кумулятивно — почти в 2× меньший приток + удержание.

4. **«Высокая» retention_d7 у f2 — ловушка.** Медиана **0.106 (f2)** vs
   **0.069 (f1)**, но это считается на выборке вернувшихся пользователей,
   которая у f2 в разы меньше. Абсолютное число удержанных DAU × retention_d7
   ≈ **19 (f1)** vs **21 (f2)** — почти равно при кратно меньшей базе f2 →
   фактически аудитория не растёт.

5. **Crash-rate разделяет слабо.** Медианы **9.8% (f1)** vs **6.6% (f2)** —
   обе значительно выше порога Google Play 2%, но f1 даже выше. Технические
   проблемы — не главный фактор провала.

6. **Структурная разница в Store-данных.** У f1 — 111 дней рейтингов и
   reviews_count; у f2 — **0 non-null** во всех 17 Store-колонках, потому
   что приложение снято с публикации. Сам факт отсутствия Store-сигнала —
   маркер провала, измеримый только постфактум.

7. **Корреляционная структура слабее.** В 02_eda_floodit/floodit2 §6 (cross-
   source) у f1 ≈ **249** пар |corr|>0.5, у f2 ≈ **75** пар. Меньше связности
   → метрики не движутся согласованно → меньше предсказуемости поведения.

**Что увидела бы система прогнозирования на ранних данных.**
В первые 30 дней три Product-метрики уже сигнализируют о высоком риске:
**onboarding_completion_rate** (Δ ≈ −83%), **stickiness** (Δ ≈ −62%),
**dau** (Δ ≈ −40%). Этого достаточно для бинарного классификатора
«успех/провал», обученного только на ранней Product-телеметрии без Store/
Market — что подтверждает гипотезу из §0.

**Самые предсказательные метрики (топ-3):**
1. `onboarding_completion_rate`
2. `stickiness` (особенно ранняя)
3. `dau` + его волатильность (CV)
"""))

# Section 8
cells.append(md("""\
## 8. Сохранение фигур

Все PDF сохранены через `save_figure(..., NOTEBOOK_SLUG)` в
`figures/03_comparison/`. Проверка содержимого ниже.
"""))

cells.append(code("""\
out_dir = figures_dir(NOTEBOOK_SLUG)
saved = sorted(out_dir.glob("*.pdf"))
print(f"figures/{NOTEBOOK_SLUG}/ — {len(saved)} PDF:")
for p in saved:
    print(f"  • {p.name} ({p.stat().st_size // 1024} KB)")
"""))

# Assemble
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

out_path = Path(__file__).parent / "03_comparison_floodit_vs_floodit2.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Wrote {out_path} ({len(cells)} cells)")
