"""Streamlit page: Сценарии (live What-if)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.streamlit.lib.constants import (
    APP_KEYS, APP_LABELS, TARGET_KEYS, TARGET_LABELS,
    OPERATIONAL_FEATURES, OPERATIONAL_LABELS, SLIDER_CONFIG,
)
from app.streamlit.lib.data_loaders import (
    load_floodit_df,
    load_final_config,
    load_scenario_analysis_summary,
    load_scenario_shap_summary,
    load_scenario_recommendations,
    load_scenario_avoid_actions,
)
from app.streamlit.lib.what_if_engine import WhatIfEngine, WhatIfRequest
from app.streamlit.lib.plots import plot_delta_histogram
from app.streamlit.lib.text import interpret_scenario_result, hypothesis_for_action
from app.streamlit.lib.winner_registry import resolve_winner_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = PROJECT_ROOT / "results" / "forecast_validation_recursive_summary.csv"

st.set_page_config(page_title="Сценарии", page_icon="🎚", layout="wide")
st.title("Сценарии (What-if)")
st.caption(
    "Counterfactual sandbox: оценка эффекта изменения управляемых operational features. "
    "Модель и обучение не пересчитываются; меняется только baseline state."
)


@st.cache_resource
def get_engine() -> WhatIfEngine:
    df = load_floodit_df()
    final_config = load_final_config()
    return WhatIfEngine(df, SUMMARY_PATH, final_config)


engine = get_engine()
final_config = load_final_config()
scenarios_csv = load_scenario_analysis_summary()

c1, c2 = st.columns(2)
app_label = c1.selectbox("Продукт", [APP_LABELS[k] for k in APP_KEYS], key="scen_app")
app_key = next(k for k, v in APP_LABELS.items() if v == app_label)

target_label = c2.selectbox("Метрика", [TARGET_LABELS[t] for t in TARGET_KEYS], key="scen_target")
target_key = next(k for k, v in TARGET_LABELS.items() if v == target_label)

base_features = final_config[app_key][target_key]["features"]
model_name = resolve_winner_model(app_key, target_key, SUMMARY_PATH)
medians = final_config[app_key][target_key].get("feature_train_medians", {})

st.info(f"Используется модель **{model_name}** — та же, на которой строился прогноз.")

# ── Секция: Диагноз модели ────────────────────────────────────────────────────
st.markdown("### 🩺 Диагностические сигналы")
st.caption(
    f"Что модель **{model_name}** видит как ключевые сигналы для прогноза "
    f"метрики **{target_label}** на продукте {app_label}. Это не рекомендации к действию, "
    "а объяснение, какие признаки больше всего влияют на текущий прогноз."
)
shap_df = load_scenario_shap_summary()
shap_sub = shap_df[(shap_df["app"] == app_key) & (shap_df["target"] == target_key)].sort_values("shap_abs", ascending=False).head(3)

if len(shap_sub) == 0:
    st.info("SHAP-данные для этой пары недоступны.")
else:
    diag_cols = st.columns(len(shap_sub))
    for col, (_, row) in zip(diag_cols, shap_sub.iterrows()):
        with col:
            feature_name = str(row["feature"])
            shap_value = float(row["shap_value"])
            is_op = bool(row["is_operational"])
            # Безопасная формулировка: SHAP value — это вклад в прогноз, без интерпретации "лучше/хуже"
            direction = "вносит положительный вклад в прогноз" if shap_value > 0 else "вносит отрицательный вклад в прогноз"
            # Цвет нейтральный для всех — не путаем зелёным "хорошо" / красным "плохо"
            color = "#3b82f6" if shap_value > 0 else "#6366f1"
            op_badge = "🎚 управляемая характеристика" if is_op else "🔒 историческая динамика, не управляется напрямую"
            st.markdown(
                f"<div style='padding:10px;border-left:4px solid {color};background:#f9fafb;'>"
                f"<div style='font-weight:bold;font-size:14px;'>{feature_name}</div>"
                f"<div style='font-size:12px;color:#6b7280;'>{op_badge}</div>"
                f"<div style='font-size:12px;margin-top:4px;'>"
                f"shap = {shap_value:+.4f} ({direction})"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
st.divider()

# ── Секция: Рекомендации из исследования ──────────────────────────────────────
st.markdown("### 💡 Рекомендации из исследования")
st.caption("Действия, отранжированные по приоритету. Не показаны экстраполяционные сценарии.")
rec_df = load_scenario_recommendations()
rec_sub = rec_df[
    (rec_df["app"] == app_key) &
    (rec_df["target"] == target_key) &
    (rec_df["extrapolation_warning"] == False)  # noqa: E712
].sort_values("priority_rank")

if len(rec_sub) == 0:
    st.info(
        "Для этой пары система не нашла безопасных автоматических рекомендаций. "
        "Доступен только ручной анализ; рискованные сценарии вынесены в анти-рекомендации."
    )
else:
    for _, row in rec_sub.iterrows():
        action = str(row["action"])
        target_value = float(row["target_value"])
        delta_pct = float(row["expected_delta_pct"])
        p_succ = float(row["p_success"])
        ci_lo = float(row["ci95_lo_pct"])
        ci_hi = float(row["ci95_hi_pct"])
        rank = int(row["priority_rank"])
        hypothesis = hypothesis_for_action(action, target_value)
        with st.container(border=True):
            st.markdown(f"**Приоритет {rank}.** {hypothesis}")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Ожидаемый Δ, %", f"{delta_pct:+.2f}")
            cc2.metric("CI95, %", f"[{ci_lo:.2f}; {ci_hi:.2f}]")
            cc3.metric("p_success", f"{p_succ:.3f}")
            st.caption(f"Машинная форма: `{action}`")
st.divider()

st.markdown("### 🎛 Ручной анализ")


def _safe_recommended_target(feature: str) -> float | None:
    """Возвращает recommended target для одиночной фичи из scenario_recommendations.csv.

    Источник — уже отфильтрованные **безопасные** single-action рекомендации
    (extrapolation_warning=False). Этим обходится баг, когда combined_improvement
    в полном scenario_analysis_summary.csv помечен как экстраполяция и ошибочно
    блокирует независимую безопасную рекомендацию для той же фичи
    (например, f2/stickiness/crash_rate=0.02 при экстраполяционном combined).
    """
    sub = rec_df[
        (rec_df["app"] == app_key)
        & (rec_df["target"] == target_key)
        & (rec_df["extrapolation_warning"] == False)  # noqa: E712
    ]
    for _, row in sub.iterrows():
        action_str = str(row["action"])
        action_feat = action_str.split("->")[0].strip() if "->" in action_str else action_str.strip()
        if action_feat == feature:
            return float(row["target_value"])
    return None


# guardrail = у фичи нет безопасной single-action рекомендации для текущей пары.
# Не путать с "признак вообще нельзя крутить" — слайдер всё равно доступен для ручного анализа,
# просто в Recommended target preset мы его не подставляем.
guardrail_disabled: dict[str, bool] = {
    feat: _safe_recommended_target(feat) is None
    for feat in OPERATIONAL_FEATURES
}
any_safe = any(not guardrail_disabled[f] for f in OPERATIONAL_FEATURES)


def _clamp(v: float, cfg: dict) -> float:
    """Огибаем диапазон слайдера, иначе session_state мог бы выйти за [min, max]."""
    return float(min(max(v, float(cfg["min"])), float(cfg["max"])))


def _apply_baseline_preset() -> None:
    """Callback кнопки «Сбросить к baseline»: ставит слайдеры на train medians."""
    for feat in OPERATIONAL_FEATURES:
        if feat in base_features and feat in medians:
            st.session_state[f"sl_{app_key}_{target_key}_{feat}"] = _clamp(
                float(medians[feat]), SLIDER_CONFIG[feat]
            )


def _apply_recommended_preset() -> None:
    """Callback кнопки «Recommended target»: ставит слайдеры на значения из
    scenario_recommendations.csv (только безопасные single-action рекомендации).
    Фичи без безопасной рекомендации остаются на текущем значении слайдера."""
    for feat in OPERATIONAL_FEATURES:
        if guardrail_disabled[feat] or feat not in base_features:
            continue
        rec = _safe_recommended_target(feat)
        if rec is not None:
            st.session_state[f"sl_{app_key}_{target_key}_{feat}"] = _clamp(
                float(rec), SLIDER_CONFIG[feat]
            )


# Слайдеры (callback'и применяются ДО создания виджетов — это обходит StreamlitAPIException).
slider_values: dict[str, float] = {}
slider_cols = st.columns(len(OPERATIONAL_FEATURES))

for col, feat in zip(slider_cols, OPERATIONAL_FEATURES):
    with col:
        cfg = SLIDER_CONFIG[feat]
        default = float(medians.get(feat, (cfg["min"] + cfg["max"]) / 2))
        disabled = feat not in base_features
        help_txt = "не используется моделью для выбранной пары" if disabled else None
        val = st.slider(
            OPERATIONAL_LABELS[feat],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(min(max(default, cfg["min"]), cfg["max"])),
            step=float(cfg["step"]),
            disabled=disabled,
            help=help_txt,
            key=f"sl_{app_key}_{target_key}_{feat}",
        )
        slider_values[feat] = val

# Кнопки-пресеты — две: сброс к baseline + применить research recommendation.
pc1, pc2 = st.columns(2)
pc1.button(
    "🔄 Сбросить к baseline (train median)",
    on_click=_apply_baseline_preset,
    help="Возвращает все слайдеры на медианные значения признаков из обучающей выборки.",
)
pc2.button(
    "💡 Применить Recommended target",
    disabled=not any_safe,
    on_click=_apply_recommended_preset,
    help=(
        "Подставляет значения признаков из топовых рекомендаций исследования "
        "(scenario_recommendations.csv). Экстраполяционные сценарии исключены — "
        "для них используйте ручной анализ слайдером."
    ),
)

if any(guardrail_disabled.values()):
    no_safe_feats = [feat for feat in OPERATIONAL_FEATURES if guardrail_disabled[feat]]
    st.caption(
        f"⚠️ Для этой пары нет безопасных одиночных рекомендаций по признакам: "
        f"{', '.join(no_safe_feats)}. Их слайдеры остаются на текущем значении при "
        "нажатии «Recommended target»; используйте ручной анализ слайдером."
    )

if st.button("Запустить сценарий", type="primary"):
    changes = {
        f: slider_values[f] for f in OPERATIONAL_FEATURES
        if f in base_features and abs(slider_values[f] - float(medians.get(f, slider_values[f]))) > 1e-9
    }
    with st.spinner("MC-симуляция (~1 сек)..."):
        req = WhatIfRequest(app=app_key, target=target_key, changes=changes)
        result = engine.run(req)

    m1, m2, m3 = st.columns(3)
    m1.metric("Δ, %", f"{result.expected_delta_pct:.2f}")
    m2.metric("CI95, %", f"[{result.ci95_lo_pct:.2f}; {result.ci95_hi_pct:.2f}]")
    m3.metric("p_success", f"{result.p_success:.3f}")

    st.write(
        interpret_scenario_result(
            target=target_key,
            feature_changes=changes,
            delta_pct=result.expected_delta_pct,
            p_success=result.p_success,
            extrapolation=result.extrapolation_warning,
        )
    )

    fig = plot_delta_histogram(
        delta_pct_distribution=result.delta_pct_distribution,
        ci_lo=result.ci95_lo_pct,
        ci_hi=result.ci95_hi_pct,
        median=result.expected_delta_pct,
    )
    st.pyplot(fig, use_container_width=False)

    if result.extrapolation_warning:
        st.caption(
            "Большая величина эффекта может объясняться экстраполяцией за пределы "
            "исторически наблюдённых значений признака."
        )

st.divider()
with st.expander("⚠️ Чего лучше не делать (анти-рекомендации)"):
    avoid_df = load_scenario_avoid_actions()
    avoid_sub = avoid_df[(avoid_df["app"] == app_key) & (avoid_df["target"] == target_key)].sort_values("avoid_rank")
    if len(avoid_sub) == 0:
        st.info("Для этой пары нет анти-рекомендаций.")
    else:
        st.markdown("Модель прогнозирует, что эти изменения **ухудшат** метрику или дадут экстраполяционный эффект:")
        display = avoid_sub[["avoid_rank", "action", "expected_delta_pct", "p_success", "extrapolation_warning"]].copy()
        display.columns = ["Ранг", "Действие", "Ожидаемый Δ, %", "p_success", "Экстраполяция?"]
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Отрицательный Δ означает ухудшение метрики. Высокий p_success при отрицательном Δ — "
            "уверенный прогноз ухудшения. Флаг «Экстраполяция?» предупреждает о выходе фичи за "
            "пределы наблюдённых значений."
        )
