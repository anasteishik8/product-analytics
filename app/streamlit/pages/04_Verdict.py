"""Streamlit page: Вердикт."""
from __future__ import annotations

import streamlit as st

from app.streamlit.lib.constants import APP_KEYS, APP_LABELS, APP_PACKAGE_IDS, VERDICT_COLORS
from app.streamlit.lib.data_loaders import load_viability_summary, load_viability_decomposition
from app.streamlit.lib.plots import plot_verdict_decomposition

st.set_page_config(page_title="Вердикт", page_icon="✅", layout="wide")
st.title("Вердикт")

st.markdown("### Итоговый вердикт системы: **M3**")
st.caption("M1 и M2 — ablation для контекста. M3 — полный пайплайн product + market + store_completeness.")

summary = load_viability_summary()
decomp = load_viability_decomposition()


def _style_verdict(val):
    if not isinstance(val, str):
        return ""
    color = VERDICT_COLORS.get(val, "")
    return f"background-color: {color}; color: white;" if color else ""


styled = summary.style.applymap(_style_verdict, subset=["final_verdict"])
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()
app_label = st.selectbox("Карточка продукта", [APP_LABELS[k] for k in APP_KEYS])
app_key = next(k for k, v in APP_LABELS.items() if v == app_label)
app_id = APP_PACKAGE_IDS[app_key]

m3_row = summary[(summary["app_id"] == app_id) & (summary["model"] == "M3")]
if len(m3_row):
    verdict = str(m3_row.iloc[0]["final_verdict"])
    color = VERDICT_COLORS.get(verdict, "#6b7280")
    st.markdown(
        f"<div style='display:inline-block;padding:10px 24px;border-radius:8px;"
        f"background:{color};color:white;font-size:22px;font-weight:bold;'>"
        f"M3 → {verdict}</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T95", f"{m3_row.iloc[0]['T95']}")
    c2.metric("T99", f"{m3_row.iloc[0]['T99']}")
    c3.metric("D² mean", f"{m3_row.iloc[0]['D2_mean']:.2f}")
    c4.metric("bad share", f"{m3_row.iloc[0]['bad_share_mean']:.3f}")

decomp_app = summary[summary["app_id"] == app_id][["model", "days_develop", "days_monitor", "days_discontinue"]].reset_index(drop=True)
if len(decomp_app):
    fig = plot_verdict_decomposition(decomp_app)
    st.pyplot(fig, use_container_width=False)

# простое текстовое заключение
if len(m3_row):
    verdict = str(m3_row.iloc[0]["final_verdict"])
    if verdict == "DEVELOP":
        st.success("Продукт демонстрирует устойчивые показатели; рекомендация — продолжать разработку.")
    elif verdict == "MONITOR":
        st.warning("Показатели смешанные; рекомендация — мониторинг и точечные действия.")
    elif verdict == "DISCONTINUE":
        st.error("Показатели и горизонты прогноза сигнализируют о высоком риске; рекомендация — рассмотреть прекращение.")
