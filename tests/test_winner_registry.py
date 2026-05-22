"""Тесты winner_registry — единственного источника выбора winner-модели."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from app.streamlit.lib.winner_registry import (
    resolve_winner_model,
    WhatIfUnavailable,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "results" / "forecast_validation_recursive_summary.csv"


def test_resolve_f1_stickiness_returns_ridge():
    assert resolve_winner_model("f1", "stickiness", SUMMARY_PATH) == "Ridge"


def test_resolve_f1_dau_returns_random_forest():
    assert resolve_winner_model("f1", "dau", SUMMARY_PATH) == "RandomForest"


def test_resolve_f1_retention_d7_returns_local_linear():
    assert resolve_winner_model("f1", "retention_d7", SUMMARY_PATH) == "LocalLinear"


def test_resolve_f2_stickiness_returns_ridge():
    assert resolve_winner_model("f2", "stickiness", SUMMARY_PATH) == "Ridge"


def test_resolve_f2_dau_returns_nadaraya_watson():
    assert resolve_winner_model("f2", "dau", SUMMARY_PATH) == "NadarayaWatson"


def test_resolve_f2_retention_d7_returns_nadaraya_watson():
    assert resolve_winner_model("f2", "retention_d7", SUMMARY_PATH) == "NadarayaWatson"


def test_resolve_raises_when_no_match(tmp_path):
    """Если ни одна строка не проходит фильтры — WhatIfUnavailable."""
    fake_csv = tmp_path / "fake.csv"
    pd.DataFrame({
        "app": ["f1"],
        "target": ["stickiness"],
        "model": ["Ridge"],
        "RMSE": [0.1],
        "beats_naive": [False],
        "warning": ["does not beat naive"],
    }).to_csv(fake_csv, index=False)
    with pytest.raises(WhatIfUnavailable):
        resolve_winner_model("f1", "stickiness", fake_csv)


ALLOWLIST = {
    "app/streamlit/lib/winner_registry.py",
    "tests/test_winner_registry.py",
}

PATTERNS = [
    re.compile(r"final_config\s*\[.*?\]\s*\[.*?\]\s*\[(['\"])model\1\]"),
    re.compile(r"\.get\(['\"]model['\"]\)"),
]

SEARCH_ROOTS = ["app", "scripts"]


def test_invariant_1_no_alternate_model_selection():
    """Выбор winner-модели допускается только через winner_registry.

    Запрещены: final_config[...][...]["model"], .get("model"), .get('model').
    Чтение features/feature_train_medians/operational_features из final_config — разрешено.
    """
    project_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []

    for root_name in SEARCH_ROOTS:
        root = project_root / root_name
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            relative = py_file.relative_to(project_root).as_posix()
            if relative in ALLOWLIST:
                continue
            text = py_file.read_text(encoding="utf-8")
            for pat in PATTERNS:
                for match in pat.finditer(text):
                    line_no = text[: match.start()].count("\n") + 1
                    violations.append(f"{relative}:{line_no}: {match.group(0)}")

    assert not violations, (
        "Прямой выбор winner-модели вне winner_registry:\n  "
        + "\n  ".join(violations)
        + "\n\nИспользуй resolve_winner_model() из app/streamlit/lib/winner_registry.py."
    )
