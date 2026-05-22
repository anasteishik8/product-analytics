"""Streamlit Home — обзор проекта.

Запуск: streamlit run app/streamlit/Home.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.streamlit.lib.constants import (
    THESIS_TITLE, AUTHOR_NAME, SUPERVISOR_NAME, DEFENSE_DATE,
    APP_LABELS, APP_PACKAGE_IDS, APP_DESCRIPTIONS, VERDICT_COLORS,
)
from app.streamlit.lib.data_loaders import load_viability_summary

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

st.set_page_config(
    page_title="Обзор",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(THESIS_TITLE)
st.caption(f"Автор: {AUTHOR_NAME} · Научный руководитель: {SUPERVISOR_NAME} · Защита: {DEFENSE_DATE}")
st.divider()


def _m3_verdict_for(app_key: str, summary) -> str:
    sub = summary[(summary["app_id"] == APP_PACKAGE_IDS[app_key]) & (summary["model"] == "M3")]
    if len(sub) == 0:
        return "—"
    return str(sub.iloc[0]["final_verdict"])


summary = load_viability_summary()

col1, col2 = st.columns(2)
for col, app_key in zip([col1, col2], ["f1", "f2"]):
    with col:
        st.subheader(APP_LABELS[app_key])
        card_path = ASSETS_DIR / (f"flood_it_card.svg" if app_key == "f1" else "flood_it_2_card.svg")
        if card_path.exists():
            st.image(str(card_path), use_container_width=True)
        else:
            png_fallback = ASSETS_DIR / (f"flood_it_card.png" if app_key == "f1" else "flood_it_2_card.png")
            if png_fallback.exists():
                st.image(str(png_fallback), use_container_width=True)
        st.markdown(f"**Package:** `{APP_PACKAGE_IDS[app_key]}`")
        st.markdown(f"**Описание:** {APP_DESCRIPTIONS[app_key]}")
        verdict = _m3_verdict_for(app_key, summary)
        color = VERDICT_COLORS.get(verdict, "#6b7280")
        st.markdown(
            f"<div style='display:inline-block;padding:4px 10px;border-radius:6px;"
            f"background:{color};color:white;font-weight:bold;'>Итоговый вердикт (M3): {verdict}</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.info(
    "Метод позволяет различить успешный и снятый с публикации продукт "
    "на основе одних и тех же 60 признаков. Подробнее — на страницах слева."
)
