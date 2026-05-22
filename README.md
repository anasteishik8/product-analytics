# product-analytics

Система прогнозирования востребованности IT-продуктов: полный аналитический цикл — диагностика данных, прогноз ключевых метрик, оценка надёжности, сценарный анализ управляемых признаков, объяснение модели, формирование рекомендаций и сводная оценка состояния продукта — реализован в единой воспроизводимой процедуре.

Кейс-валидация выполнена на двух мобильных приложениях Google категории Puzzle: «Flood-It!» (`com.labpixies.flood`) и «Flood-It! 2» (`com.google.flood2`).

## Структура репозитория

| Каталог | Содержание |
|---|---|
| `src/` | Математическое ядро: ядерная регрессия Надарая–Ватсона, локально-линейная регрессия, VIF-фильтрация, регрессионные метрики, тест Дибольда–Мариано, сценарный анализ Монте-Карло, BCa-бутстреп. |
| `notebooks/` | Расчётный пайплайн: фазы 01 → 08 (обзор источников данных, EDA по продуктам, сравнение, моделирование, безопасный горизонт, валидация прогноза, сценарный анализ, сводная оценка). |
| `app/streamlit/` | Веб-интерфейс из четырёх страниц: Данные, Прогноз, Сценарии, Вердикт. |
| `tests/` | Около 95 unit/integration тестов на pytest, покрывающих все 4 модуля ядра. |
| `scripts/` | Утилиты сборки рисунков и таблиц, проверка реестра контрольных сумм. |
| `data/processed/` | Подготовленный датасет `floodit_final.parquet` (222 строки × 60 колонок). |
| `results/` | 16 файлов результатов вычислительного пайплайна (CSV/JSON). |
| `figures/` | Графики результатов в формате PDF. |
| `vkr/v2/artifacts/` | Реестр контрольных сумм `registry.csv` (SHA-256 для всех файлов результатов). |

## Воспроизведение

```bash
# 1. Подготовка окружения (Python 3.11)
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.lock

# 2. Автоматические тесты (2–3 минуты, около 95 тестов)
pytest tests/ -v

# 3. Smoke-тест импортов математического ядра
python -c "from src import kernel_regression, feature_engineering, evaluation, scenario_analysis; print('all imports OK')"

# 4. Пакетный запуск расчётных ноутбуков
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/06b_forecast_validation_recursive.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/07_scenario_analysis.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/08_product_viability_verdict.ipynb

# 5. Веб-демо
streamlit run app/streamlit/Home.py

# 6. Проверка целостности результатов по реестру SHA-256
python scripts/check_registry.py
```

Полный прогон пайплайна от исходного датасета до сводной оценки занимает около 30 минут на стандартном ноутбуке (Python 3.11, ~8 ГБ ОЗУ, без GPU).

## Технологии

Python 3.11 · Apache Parquet · pandas · NumPy · SciPy · scikit-learn · SHAP · Streamlit · pytest.

Воспроизводимость обеспечивается тремя инвариантами:

1. Фиксированное зерно генератора случайных чисел `seed = 42` во всех модулях, использующих случайные процедуры.
2. Неизменность входного датасета `data/processed/floodit_final.parquet`.
3. Реестр SHA-256 для всех файлов результатов в `vkr/v2/artifacts/registry.csv`.

Полный список зависимостей зафиксирован в `requirements.lock` с явным указанием версий.

## Результаты на тестовом кейсе

| Продукт | Сводная оценка | Доля «плохих» дней | Среднее d² Махаланобиса |
|---|---|---|---|
| «Flood-It!»   | **развивать** | 21,2% | ≈ 362  |
| «Flood-It! 2» | **закрывать** | 32,8% | ≈ 2522 |

Формальные сводные оценки получены без обращения к финалу истории продуктов и совпали с реальной судьбой: «Flood-It!» продолжает поддерживаться разработчиком, «Flood-It! 2» снят с публикации в Google Play.

## Лицензия

MIT — см. `LICENSE`.

## Автор

Лукина А.А., группа 4236, ГУАП, 2026.
