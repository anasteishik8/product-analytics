from app.streamlit.lib.text import interpret_scenario_result, hypothesis_for_action


def test_interpret_positive_high_confidence():
    text = interpret_scenario_result(
        target="stickiness",
        feature_changes={"crash_rate": 0.02},
        delta_pct=7.65,
        p_success=0.658,
        extrapolation=False,
    )
    assert "crash_rate" in text
    assert "7.65" in text or "7.7" in text
    assert "stickiness" in text


def test_interpret_negative_warns():
    text = interpret_scenario_result(
        target="retention_d7",
        feature_changes={"crash_rate": 0.02},
        delta_pct=-191.6,
        p_success=0.074,
        extrapolation=False,
    )
    assert "negative" in text.lower() or "снижени" in text.lower() or "ухудшени" in text.lower() or "ниже" in text.lower()


def test_interpret_extrapolation_warning():
    text = interpret_scenario_result(
        target="retention_d7",
        feature_changes={"onboarding_completion_rate": 0.8},
        delta_pct=426.4,
        p_success=0.999,
        extrapolation=True,
    )
    assert "экстраполяц" in text.lower()


def test_hypothesis_crash_rate_contains_percent():
    h = hypothesis_for_action("crash_rate -> 0.020", 0.020)
    assert "crash_rate" in h.lower() or "сбое" in h.lower() or "крашей" in h.lower()
    assert "2" in h  # 2% или 0.020 — какое-то упоминание значения


def test_hypothesis_onboarding_contains_explanation():
    h = hypothesis_for_action("onboarding_completion_rate -> 0.800", 0.800)
    assert "onboarding" in h.lower() or "tutorial" in h.lower() or "обуча" in h.lower()


def test_hypothesis_session_duration_mentions_gameplay():
    h = hypothesis_for_action("median_session_duration -> 0.082", 0.082)
    assert "сессии" in h.lower() or "gameplay" in h.lower()


def test_hypothesis_unknown_feature_fallback():
    h = hypothesis_for_action("foo_bar -> 0.5", 0.5)
    assert "foo_bar" in h or "Действие" in h


from app.streamlit.lib.text import forecast_verdict


def test_forecast_verdict_rising_normal():
    v = forecast_verdict(target="stickiness", h_safe_A=24, h_demo_used=False, trend="rising")
    assert "растёт" in v
    assert "24" in v
    assert "retention" not in v.lower()


def test_forecast_verdict_falling_with_demo():
    v = forecast_verdict(target="dau", h_safe_A=2, h_demo_used=True, trend="falling")
    assert "снижается" in v
    assert "2" in v
    assert "демонстрац" in v.lower()


def test_forecast_verdict_h_safe_zero():
    v = forecast_verdict(target="retention_d7", h_safe_A=0, h_demo_used=True, trend="falling")
    assert "надёжный прогноз отсутствует" in v
    # для h_safe_A=0 не должно быть фразы про конкретный безопасный горизонт > 0
    assert "стабильна" not in v
    assert "снижается" not in v
    # retention_d7-блок присутствует
    assert "шумная" in v


def test_forecast_verdict_retention_d7_adds_diagnostic_note():
    v = forecast_verdict(target="retention_d7", h_safe_A=3, h_demo_used=False, trend="flat")
    assert "стабильна" in v
    assert "диагностич" in v.lower()


def test_forecast_verdict_flat_trend():
    v = forecast_verdict(target="stickiness", h_safe_A=24, h_demo_used=False, trend="flat")
    assert "стабильна" in v
