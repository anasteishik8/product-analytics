"""
Пакет src — математические примитивы системы прогнозирования
востребованности IT-продуктов.

Модули реализуют основные алгоритмы, используемые в Jupyter-ноутбуках
(notebooks/04–08):

    kernel_regression    — ядерная регрессия Надарая–Ватсона и
                           локально-линейная регрессия (Härdle 1990,
                           Fan & Gijbels 1996)
    feature_engineering  — загрузка датасета, VIF-анализ (Belsley 1980),
                           отбор признаков (используется
                           notebooks/04_modeling_* в режиме clean
                           через notebooks/_common.py::select_features_clean)
    evaluation           — метрики качества прогноза (RMSE, MAE, MAPE,
                           sMAPE, R², MASE), тест Дибольда–Мариано
    scenario_analysis    — Monte Carlo сценарный анализ + BCa bootstrap
                           (Efron & Tibshirani 1993, Hyndman 2006).
                           ScenarioAnalyzer.bca_bootstrap_ci используется
                           в notebooks/07_scenario_analysis для расчёта
                           доверительных интервалов сценарных прогнозов

Авторская методология (recursive multi-step forecast, counterfactual
scenarios, Mahalanobis viability classifier) реализована inline внутри
ноутбуков 06b, 07, 08 — см. docs/PROJECT_HANDBOOK.md.
"""
