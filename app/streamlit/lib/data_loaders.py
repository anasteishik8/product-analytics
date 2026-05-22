"""@st.cache_data обёртки для всех источников Streamlit-приложения.

Все CSV/parquet/JSON читаются здесь и только здесь.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_floodit_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "floodit_final.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_viability_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "viability_summary.csv")


@st.cache_data
def load_viability_decomposition() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "viability_decomposition.csv")


@st.cache_data
def load_demo_forecast_predictions() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "demo_forecast_predictions.csv")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data
def load_horizon_safe() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "horizon_safe.csv")


@st.cache_data
def load_forecast_validation_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "forecast_validation_recursive_summary.csv")


@st.cache_data
def load_scenario_analysis_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "scenario_analysis_summary.csv")


@st.cache_data
def load_final_config() -> dict:
    with open(RESULTS_DIR / "final_config.json") as f:
        return json.load(f)


@st.cache_data
def load_scenario_shap_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "scenario_shap_summary.csv")


@st.cache_data
def load_scenario_recommendations() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "scenario_recommendations.csv")


@st.cache_data
def load_scenario_avoid_actions() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "scenario_avoid_actions.csv")
