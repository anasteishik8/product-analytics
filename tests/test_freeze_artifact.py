"""Тесты для scripts/freeze_artifact.py."""
import csv
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from freeze_artifact import update_row, REGISTRY_FIELDS  # noqa: E402


def _write_registry(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in REGISTRY_FIELDS})


def _read_registry(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _seed_row(id_: str) -> dict:
    return {
        "id": id_, "type": "figure", "chapter": "5", "title": "Тест",
        "caption": "", "source_kind": "", "source_path": "",
        "artifact_path": "", "source_hash": "", "artifact_hash": "",
        "status": "planned", "body_or_appendix": "body",
        "last_verified": "", "notes": "",
    }


def test_update_row_modifies_only_target_id(tmp_path):
    reg = tmp_path / "registry.csv"
    _write_registry(reg, [_seed_row("fig5_1"), _seed_row("fig5_2")])
    update_row(reg, "fig5_1", status="frozen", caption="My caption")
    rows = _read_registry(reg)
    by_id = {r["id"]: r for r in rows}
    assert by_id["fig5_1"]["status"] == "frozen"
    assert by_id["fig5_1"]["caption"] == "My caption"
    assert by_id["fig5_2"]["status"] == "planned"
    assert by_id["fig5_2"]["caption"] == ""


def test_update_row_computes_source_hash(tmp_path):
    reg = tmp_path / "registry.csv"
    _write_registry(reg, [_seed_row("fig5_1")])
    src = tmp_path / "data.csv"
    src.write_text("hello", encoding="utf-8")
    update_row(reg, "fig5_1", source_path=str(src))
    rows = _read_registry(reg)
    assert rows[0]["source_hash"] != ""
    assert len(rows[0]["source_hash"]) == 16


def test_update_row_computes_artifact_hash(tmp_path):
    reg = tmp_path / "registry.csv"
    _write_registry(reg, [_seed_row("fig5_1")])
    art = tmp_path / "art.pdf"
    art.write_bytes(b"PDF stub")
    update_row(reg, "fig5_1", artifact_path=str(art))
    rows = _read_registry(reg)
    assert rows[0]["artifact_hash"] != ""
    assert len(rows[0]["artifact_hash"]) == 16


def test_update_row_sets_last_verified_when_frozen(tmp_path):
    reg = tmp_path / "registry.csv"
    _write_registry(reg, [_seed_row("fig5_1")])
    update_row(reg, "fig5_1", status="frozen")
    rows = _read_registry(reg)
    assert len(rows[0]["last_verified"]) == 10
    assert rows[0]["last_verified"][4] == "-"


def test_update_row_raises_for_unknown_id(tmp_path):
    reg = tmp_path / "registry.csv"
    _write_registry(reg, [_seed_row("fig5_1")])
    with pytest.raises(KeyError, match="not found"):
        update_row(reg, "fig5_99", status="frozen")


def test_update_row_preserves_row_order(tmp_path):
    reg = tmp_path / "registry.csv"
    _write_registry(reg, [_seed_row("fig5_3"), _seed_row("fig5_1"), _seed_row("fig5_2")])
    update_row(reg, "fig5_1", status="frozen")
    rows = _read_registry(reg)
    assert [r["id"] for r in rows] == ["fig5_3", "fig5_1", "fig5_2"]
