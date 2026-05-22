"""Streamlit page: Данные."""
from __future__ import annotations

import streamlit as st

from app.streamlit.lib.constants import APP_PACKAGE_IDS
from app.streamlit.lib.data_loaders import load_floodit_df
from app.streamlit.lib.plots import plot_nan_heatmap

st.set_page_config(page_title="Данные", page_icon="🗄", layout="wide")
st.title("Данные")

df = load_floodit_df()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Shape", f"{df.shape[0]}×{df.shape[1]}")
c2.metric("Приложений", df["app_id"].nunique())
c3.metric("Дней", df["date"].nunique())
c4.metric("Синтетика", "0%")

st.subheader("4 группы источников")
sources = [
    ("Google Play Scraper", "store metrics", "daily"),
    ("Firebase BigQuery (Kaggle export)", "engagement / retention", "daily"),
    ("Market trends (custom)", "competitor_count, category_rank", "daily"),
    ("External signals (Google Trends + Wikipedia)", "downloads_index, media_coverage", "daily"),
]
st.dataframe(
    {"source": [s[0] for s in sources],
     "metrics_count": [s[1] for s in sources],
     "frequency": [s[2] for s in sources]},
    use_container_width=True, hide_index=True,
)

st.subheader("Превью данных (head 5)")
preview_cols = ["app_id", "date", "dau", "mau", "stickiness", "retention_d7", "crash_rate"]
existing = [c for c in preview_cols if c in df.columns]
st.dataframe(df.head(5)[existing], use_container_width=True, hide_index=True)

st.subheader("Покрытие данных")
fig = plot_nan_heatmap(df)
st.pyplot(fig, use_container_width=False)

missing = df.isna().mean()
missing_high = missing[missing >= 0.05].sort_values(ascending=False)
if len(missing_high) > 0:
    st.markdown("**Колонки с missingness ≥ 5%:**")
    st.dataframe(
        {"feature": missing_high.index.tolist(),
         "missing_ratio": [f"{v:.1%}" for v in missing_high.values]},
        use_container_width=True, hide_index=True,
    )
else:
    st.success("Все колонки имеют missingness < 5%.")
