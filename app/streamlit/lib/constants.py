"""Константы Streamlit-приложения: лейблы, маппинги, диапазоны слайдеров.

Единственное место для русификации UI: чтобы перевести на другой язык —
редактируется только этот файл.
"""
from __future__ import annotations

from typing import Final

# Метаданные ВКР (UI шапка)
THESIS_TITLE: Final = "Разработка системы прогнозирования востребованности IT-продуктов"
AUTHOR_NAME: Final = "Лукина Анастасия Андреевна"
SUPERVISOR_NAME: Final = "Колесникова Светлана Ивановна"
DEFENSE_DATE: Final = "2026-06-19"

# Apps
APP_KEYS: Final = ("f1", "f2")
APP_LABELS: Final = {
    "f1": "Flood-It!",
    "f2": "Flood-It! 2",
}
APP_PACKAGE_IDS: Final = {
    "f1": "com.labpixies.flood",
    "f2": "com.google.flood2",
}
APP_DESCRIPTIONS: Final = {
    "f1": "успешный продукт, ~4.9 млн установок",
    "f2": "снят с публикации",
}
APP_COLORS: Final = {
    "f1": "#3b82f6",  # blue
    "f2": "#ef4444",  # red
}

# Targets
TARGET_KEYS: Final = ("stickiness", "dau", "retention_d7")
TARGET_LABELS: Final = {
    "stickiness": "stickiness (DAU/MAU)",
    "dau": "DAU",
    "retention_d7": "retention_d7",
}

# Operational features (управляемые менеджером — единственные, что показаны в What-if)
OPERATIONAL_FEATURES: Final = ("crash_rate", "onboarding_completion_rate", "median_session_duration")
OPERATIONAL_LABELS: Final = {
    "crash_rate": "crash_rate",
    "onboarding_completion_rate": "onboarding_completion_rate",
    "median_session_duration": "median_session_duration",
}

# Слайдеры: min, max, step, preset (recommended target)
SLIDER_CONFIG: Final = {
    "crash_rate": {"min": 0.0, "max": 0.20, "step": 0.005, "preset": 0.02},
    "onboarding_completion_rate": {"min": 0.0, "max": 1.0, "step": 0.01, "preset": 0.80},
    "median_session_duration": {"min": 0.02, "max": 0.20, "step": 0.001, "preset": None},
    # preset для median_session_duration берётся динамически из scenario_analysis_summary
}

# Verdict colors
VERDICT_COLORS: Final = {
    "DEVELOP": "#22c55e",
    "MONITOR": "#eab308",
    "DISCONTINUE": "#ef4444",
}

# Horizon
H_DEMO_DAYS: Final = 14
