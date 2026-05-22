"""Тесты для scripts/make_registry_skeleton.py."""
import csv
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from make_registry_skeleton import parse_contract, build_registry_rows, merge_with_existing  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "sample_contract.md"


def test_parse_contract_extracts_yaml_frontmatter():
    data = parse_contract(FIXTURE)
    assert data["chapter"] == 99
    assert data["title"] == "Test Chapter"
    assert len(data["figures"]) == 2
    assert len(data["tables"]) == 1


def test_build_registry_rows_produces_one_row_per_figure_and_table():
    data = parse_contract(FIXTURE)
    rows = build_registry_rows(data)
    assert len(rows) == 3  # 2 figures + 1 table

    fig1 = next(r for r in rows if r["id"] == "fig99_1")
    assert fig1["type"] == "figure"
    assert fig1["chapter"] == "99"
    assert fig1["title"] == "Test figure 1"
    assert fig1["source_path"] == "notebooks/test.ipynb"
    assert fig1["status"] == "planned"
    assert fig1["body_or_appendix"] == "body"

    tab1 = next(r for r in rows if r["id"] == "tab99_1")
    assert tab1["type"] == "table"


def test_merge_preserves_existing_status_for_known_ids(tmp_path):
    existing_csv = tmp_path / "registry.csv"
    fields = [
        "id", "type", "chapter", "title", "caption",
        "source_kind", "source_path", "artifact_path",
        "source_hash", "artifact_hash", "status",
        "body_or_appendix", "last_verified", "notes",
    ]
    with existing_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "id": "fig99_1", "type": "figure", "chapter": "99",
            "title": "Old title", "caption": "manually filled",
            "source_kind": "notebook", "source_path": "notebooks/test.ipynb",
            "artifact_path": "vkr/v2/artifacts/figures/fig99_1.pdf",
            "source_hash": "abc", "artifact_hash": "def",
            "status": "frozen", "body_or_appendix": "body",
            "last_verified": "2026-05-10", "notes": "",
        })

    data = parse_contract(FIXTURE)
    new_rows = build_registry_rows(data)
    merged = merge_with_existing(new_rows, existing_csv)

    fig99_1 = next(r for r in merged if r["id"] == "fig99_1")
    assert fig99_1["status"] == "frozen"
    assert fig99_1["caption"] == "manually filled"
    assert fig99_1["last_verified"] == "2026-05-10"

    fig99_2 = next(r for r in merged if r["id"] == "fig99_2")
    assert fig99_2["status"] == "planned"


def test_build_rows_skips_unknown_keys_gracefully():
    data = {
        "chapter": 99,
        "figures": [{"id": "fig_x", "title": "T", "source": "s"}],
        "tables": [],
    }
    rows = build_registry_rows(data)
    assert len(rows) == 1
    assert rows[0]["id"] == "fig_x"
