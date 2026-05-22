"""Streamlit page: Прогноз."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit.lib.constants import (
    APP_KEYS, APP_LABELS, TARGET_KEYS, TARGET_LABELS, H_DEMO_DAYS,
)
from app.streamlit.lib.data_loaders import (
    load_demo_forecast_predictions,
    load_forecast_validation_summary,
    load_horizon_safe,
)
from app.streamlit.lib.plots import plot_forecast_with_ci
from app.streamlit.lib.text import forecast_verdict

st.set_page_config(page_title="Прогноз", page_icon="📈", layout="wide")
st.title("Прогноз")

predictions = load_demo_forecast_predictions()
horizon_safe = load_horizon_safe()
val_summary = load_forecast_validation_summary()

c1, c2, c3 = st.columns([1, 1, 1])
app_label = c1.selectbox("Продукт", [APP_LABELS[k] for k in APP_KEYS])
app_key = next(k for k, v in APP_LABELS.items() if v == app_label)

target_label = c2.selectbox("Метрика", [TARGET_LABELS[t] for t in TARGET_KEYS])
target_key = next(k for k, v in TARGET_LABELS.items() if v == target_label)

horizon_choice = c3.selectbox("Горизонт", ["Безопасный (h_safe_A)", f"Демонстрационный ({H_DEMO_DAYS} дней)"])

# h_safe_A для пары
hs_row = horizon_safe[(horizon_safe["app_key"] == app_key) & (horizon_safe["target"] == target_key)]
h_safe_A = int(hs_row.iloc[0]["h_safe_A"]) if len(hs_row) else 0

sub = predictions[(predictions["app"] == app_key) & (predictions["target"] == target_key)]
history = sub[sub["segment"] == "history"]
forecast_safe = sub[sub["segment"] == "forecast_safe"]
forecast_demo = sub[sub["segment"] == "forecast_demo"]

if horizon_choice.startswith("Безопасный"):
    forecast = forecast_safe
    h_demo_used = False
else:
    forecast = pd.concat([forecast_safe, forecast_demo], ignore_index=True)
    h_demo_used = True
    if H_DEMO_DAYS > h_safe_A:
        st.warning(
            f"Демонстрационный горизонт {H_DEMO_DAYS} дней превышает безопасный "
            f"{h_safe_A} дней. Прогноз показан для иллюстрации; для решений "
            "используется только безопасный горизонт."
        )

# Вывод прогноза — человеческим языком, ДО графика.
# Trend читается из forecast_validation_recursive_summary.csv (rising / falling / flat).
trend_for_verdict = "flat"
if len(sub) > 0:
    model_name_for_trend = str(sub.iloc[0]["model"])
    vrow_t = val_summary[
        (val_summary["app"] == app_key)
        & (val_summary["target"] == target_key)
        & (val_summary["model"] == model_name_for_trend)
    ]
    if len(vrow_t) > 0 and "trend" in vrow_t.columns:
        trend_for_verdict = str(vrow_t.iloc[0].get("trend", "flat"))

verdict_text = forecast_verdict(
    target=target_key,
    h_safe_A=h_safe_A,
    h_demo_used=h_demo_used,
    trend=trend_for_verdict,
)
st.markdown(f"**Вывод прогноза:** {verdict_text}")

fig = plot_forecast_with_ci(
    history=history, forecast=forecast,
    h_safe_A=h_safe_A, h_demo=H_DEMO_DAYS, target_label=target_label,
)
st.pyplot(fig, use_container_width=True)

# метрики winner-модели
if len(sub) > 0:
    model_name = str(sub.iloc[0]["model"])
    vrow = val_summary[(val_summary["app"] == app_key) & (val_summary["target"] == target_key) & (val_summary["model"] == model_name)]
    if len(vrow) > 0:
        v = vrow.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Модель", model_name)
        m2.metric("RMSE", f"{v['RMSE']:.4g}")
        m3.metric("MAPE, %", f"{v['MAPE_pct']:.1f}")
        m4.metric("MASE", f"{v['MASE']:.2f}")

with st.expander("Методическая валидация (все кандидаты)"):
    sub_v = val_summary[(val_summary["app"] == app_key) & (val_summary["target"] == target_key)]
    if len(sub_v) > 0:
        st.dataframe(
            sub_v[["model", "RMSE", "MAPE_pct", "MASE", "Dir_Acc_pct", "beats_naive", "warning"]],
            use_container_width=True, hide_index=True,
        )

with st.expander("Методика расчёта горизонта"):
    if len(hs_row) > 0:
        st.markdown(
            f"- **h_safe_A** = {int(hs_row.iloc[0]['h_safe_A'])} — основной критерий (используется в UI)\n"
            f"- **h_safe_B** = {int(hs_row.iloc[0]['h_safe_B'])} — менее консервативный\n"
            f"- **h_safe_C** = {int(hs_row.iloc[0]['h_safe_C'])} — оптимистичный"
        )
