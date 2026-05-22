"""Тесты для scripts/csv_to_latex_table.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from csv_to_latex_table import df_to_latex_booktabs  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "sample_table_input.csv"


def test_basic_table_has_booktabs_and_caption(tmp_path):
    df = pd.read_csv(FIXTURE)
    out = tmp_path / "tab.tex"
    df_to_latex_booktabs(
        df, out,
        columns={"app": "Продукт", "target": "Метрика", "model": "Модель"},
        caption="Тест",
        label="tab:test",
    )
    text = out.read_text(encoding="utf-8")
    assert r"\toprule" in text
    assert r"\midrule" in text
    assert r"\bottomrule" in text
    assert r"\caption{Тест}" in text
    assert r"\label{tab:test}" in text
    assert "Продукт" in text
    assert "Метрика" in text
    assert "Модель" in text


def test_float_formatting_applied(tmp_path):
    df = pd.read_csv(FIXTURE)
    out = tmp_path / "tab.tex"
    df_to_latex_booktabs(
        df, out,
        columns={"RMSE": "RMSE"},
        caption="X", label="tab:x",
        float_fmt="{:.4f}",
    )
    text = out.read_text(encoding="utf-8")
    assert "0.0268" in text
    assert "0.0373" in text


def test_nan_replaced_with_dash(tmp_path):
    df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
    out = tmp_path / "tab.tex"
    df_to_latex_booktabs(df, out, columns={"a": "A"}, caption="X", label="tab:x")
    text = out.read_text(encoding="utf-8")
    assert "—" in text


def test_column_order_preserved(tmp_path):
    df = pd.read_csv(FIXTURE)
    out = tmp_path / "tab.tex"
    df_to_latex_booktabs(
        df, out,
        columns={"model": "Модель", "app": "Продукт"},  # reversed
        caption="X", label="tab:x",
    )
    text = out.read_text(encoding="utf-8")
    model_idx = text.index("Модель")
    app_idx = text.index("Продукт")
    assert model_idx < app_idx


def test_utf8_encoding(tmp_path):
    df = pd.DataFrame({"a": ["Привет"]})
    out = tmp_path / "tab.tex"
    df_to_latex_booktabs(df, out, columns={"a": "Колонка"}, caption="Кириллица", label="tab:k")
    raw = out.read_bytes()
    assert "Привет".encode("utf-8") in raw
    assert "Кириллица".encode("utf-8") in raw


def test_special_chars_escaped(tmp_path):
    df = pd.DataFrame({"a": ["A&B"], "b": ["50%"]})
    out = tmp_path / "tab.tex"
    df_to_latex_booktabs(df, out, columns={"a": "X", "b": "Y"}, caption="X", label="tab:x")
    text = out.read_text(encoding="utf-8")
    assert r"\&" in text
    assert r"\%" in text
