"""Генератор русских текстовых интерпретаций для What-if результатов."""
from __future__ import annotations


def interpret_scenario_result(
    target: str,
    feature_changes: dict[str, float],
    delta_pct: float,
    p_success: float,
    extrapolation: bool,
) -> str:
    """Возвращает 1–2 предложения, описывающих результат сценария."""
    direction = "рост" if delta_pct > 0 else ("снижение" if delta_pct < 0 else "отсутствие эффекта")
    changes_str = ", ".join(f"{k} → {v}" for k, v in feature_changes.items()) or "без изменений"
    confidence = "высокая" if p_success >= 0.75 else ("средняя" if p_success >= 0.5 else "низкая")

    lines = [
        f"Сценарий: {changes_str}. "
        f"Ожидаемое {direction} {target} на {delta_pct:.2f}% (p_success = {p_success:.3f}, уверенность — {confidence})."
    ]
    if delta_pct < 0:
        lines.append("Результат указывает на потенциальное ухудшение метрики; интерпретируйте осторожно.")
    if extrapolation:
        lines.append(
            "Величина эффекта аномально большая; вероятно, признак выходит за границы исторически "
            "наблюдённых значений — это экстраполяция, не каузальный прогноз."
        )
    return " ".join(lines)


# Маппинг ops-feature → человеко-читаемая гипотеза.
# Используется в Рекомендациях на странице Scenarios.
_HYPOTHESIS_TEMPLATES = {
    "crash_rate": (
        "Снизить crash_rate до {value:.1%} (примерно {percent_label}) через тестирование "
        "на нестабильных устройствах, профилирование памяти и hotfix-ы для топовых крашей."
    ),
    "onboarding_completion_rate": (
        "Поднять долю завершивших onboarding до {value:.1%} за счёт упрощения tutorial, "
        "сокращения first-time-user-experience и точечной локализации сложных шагов."
    ),
    "median_session_duration": (
        "Увеличить медианную длительность сессии до {value:.4f} условных единиц "
        "(приблизительно {minutes:.1f} мин при интерпретации признака в часах) "
        "через улучшение core gameplay loop, retention-механики и баланса сложности."
    ),
}


def forecast_verdict(
    target: str,
    h_safe_A: int,
    h_demo_used: bool,
    trend: str,
) -> str:
    """Возвращает «Вывод прогноза» для страницы Forecast — 1–3 предложения,
    объясняющих куда движется метрика и насколько этому можно доверять.

    Parameters
    ----------
    target : str
        Имя таргета: "stickiness" / "dau" / "retention_d7".
    h_safe_A : int
        Безопасный горизонт прогноза. 0 = надёжный прогноз отсутствует.
    h_demo_used : bool
        True, если на странице сейчас показан демонстрационный горизонт > h_safe_A.
    trend : str
        Из forecast_validation_recursive_summary.csv колонка `trend`:
        "rising" / "falling" / "flat".

    Returns
    -------
    str
        Готовый русский вывод для st.markdown.
    """
    parts: list[str] = []

    trend_phrase = {
        "rising": "растёт",
        "falling": "снижается",
        "flat": "стабильна",
    }.get(str(trend), "имеет неопределённое поведение")

    if h_safe_A == 0:
        parts.append(
            "Безопасный горизонт для этой пары равен 0 дней — надёжный прогноз отсутствует."
        )
    else:
        parts.append(f"Метрика {trend_phrase} на безопасном горизонте {h_safe_A} дн.")

    if h_demo_used:
        if h_safe_A == 0:
            parts.append("Весь показанный график — демонстрационный, для иллюстрации поведения модели.")
        else:
            parts.append("Участок после h_safe_A показан только для демонстрации.")

    if target == "retention_d7":
        parts.append(
            "retention_d7 — шумная и разреженная метрика; для неё безопасный горизонт "
            "мал или отсутствует, поэтому прогноз используется как диагностический "
            "сигнал, а не как основание для продуктового решения."
        )

    return " ".join(parts)


def hypothesis_for_action(action: str, target_value: float) -> str:
    """Превращает машинное действие вроде 'crash_rate -> 0.020' в человеческую гипотезу.

    Parameters
    ----------
    action : str
        Строка вида "feature_name -> value" из scenario_recommendations.csv.
    target_value : float
        Целевое значение фичи (target_value колонка).

    Returns
    -------
    str
        Текст гипотезы для UI. Если фича неизвестна — fallback к raw-описанию.
    """
    # action имеет вид "feature_name -> value"
    feature_name = action.split("->")[0].strip() if "->" in action else action.strip()

    template = _HYPOTHESIS_TEMPLATES.get(feature_name)
    if template is None:
        return f"Действие: {action}"

    percent_label = f"{target_value:.0%}" if 0 <= target_value <= 1 else f"{target_value:.2f}"
    # median_session_duration в данных хранится в часах (грубая конверсия в минуты — value * 60).
    # Формулировка осторожная: "условные единицы (~X мин при интерпретации в часах)",
    # т.к. единица не прописана явно в главе 2 ВКР.
    minutes = target_value * 60 if feature_name == "median_session_duration" else 0.0
    return template.format(value=target_value, percent_label=percent_label, minutes=minutes)
