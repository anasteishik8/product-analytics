"""Тесты exporter: детерминизм + соответствие схеме."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_demo_artifacts.py"
OUTPUT_CSV = PROJECT_ROOT / "results" / "demo_forecast_predictions.csv"

EXPECTED_COLUMNS = [
    "app", "target", "model", "date", "day_idx", "horizon",
    "segment", "y_true", "y_pred", "ci_lo", "ci_hi",
    "h_safe_A", "h_demo", "warning",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exporter_produces_csv_with_correct_schema():
    """После запуска exporter — CSV существует и имеет правильные колонки."""
    subprocess.run([sys.executable, str(EXPORT_SCRIPT)], check=True, cwd=PROJECT_ROOT)
    assert OUTPUT_CSV.exists()
    df = pd.read_csv(OUTPUT_CSV)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert set(df["app"].unique()) == {"f1", "f2"}
    assert set(df["target"].unique()) == {"stickiness", "dau", "retention_d7"}
    assert set(df["segment"].unique()) <= {"history", "forecast_safe", "forecast_demo"}


def test_exporter_is_deterministic():
    """Два запуска подряд → одинаковый sha256."""
    subprocess.run([sys.executable, str(EXPORT_SCRIPT)], check=True, cwd=PROJECT_ROOT)
    hash1 = _sha256(OUTPUT_CSV)
    subprocess.run([sys.executable, str(EXPORT_SCRIPT)], check=True, cwd=PROJECT_ROOT)
    hash2 = _sha256(OUTPUT_CSV)
    assert hash1 == hash2, f"non-deterministic: {hash1} vs {hash2}"


def test_exporter_history_segment_matches_real_data(real_floodit_df):
    """history segment должен содержать настоящие y_true из parquet (до day 86)."""
    subprocess.run([sys.executable, str(EXPORT_SCRIPT)], check=True, cwd=PROJECT_ROOT)
    df = pd.read_csv(OUTPUT_CSV)
    history_f1 = df[(df["app"] == "f1") & (df["segment"] == "history") & (df["target"] == "stickiness")]
    assert len(history_f1) == 87, f"expected 87 history rows for f1/stickiness, got {len(history_f1)}"
    # точечная проверка: day_idx=0 == первое значение в parquet
    src_app = real_floodit_df[real_floodit_df["app_id"] == "com.labpixies.flood"].sort_values("date").reset_index(drop=True)
    expected_first = float(src_app.iloc[0]["stickiness"])
    actual_first = float(history_f1.sort_values("day_idx").iloc[0]["y_true"])
    if not pd.isna(expected_first):
        assert abs(actual_first - expected_first) < 1e-9
