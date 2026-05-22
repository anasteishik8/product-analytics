"""Тесты для scripts/check_chapter.py."""
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from check_chapter import (  # noqa: E402
    check_page_count,
    check_artifacts_inserted,
    check_no_extra_artifacts,
    check_artifacts_frozen,
    check_conclusions_block,
    check_forbidden_phrases,
    check_citations_resolve,
    extract_numbers,
)

FIXTURE = Path(__file__).parent / "fixtures"


def test_check_page_count_within_budget():
    text = "слово " * 250 * 2  # ≈ 2 страницы
    ok, msg = check_page_count(text, page_budget=[1, 3])
    assert ok, msg


def test_check_page_count_over_budget():
    text = "слово " * 250 * 5
    ok, msg = check_page_count(text, page_budget=[1, 3])
    assert not ok
    assert "5" in msg or "превышен" in msg.lower()


def test_check_artifacts_inserted_all_present():
    text = "tab99_1 fig99_1 fig99_2"
    contract = {
        "figures": [{"id": "fig99_1"}, {"id": "fig99_2"}],
        "tables": [{"id": "tab99_1"}],
    }
    ok, missing = check_artifacts_inserted(text, contract)
    assert ok
    assert missing == []


def test_check_artifacts_inserted_missing():
    text = "только fig99_1"
    contract = {
        "figures": [{"id": "fig99_1"}, {"id": "fig99_2"}],
        "tables": [{"id": "tab99_1"}],
    }
    ok, missing = check_artifacts_inserted(text, contract)
    assert not ok
    assert "fig99_2" in missing
    assert "tab99_1" in missing


def test_check_no_extra_artifacts_finds_orphans():
    text = "fig99_1 fig99_2 fig99_99_OOPS"
    contract = {
        "figures": [{"id": "fig99_1"}, {"id": "fig99_2"}],
        "tables": [],
    }
    ok, extras = check_no_extra_artifacts(text, contract)
    assert not ok
    assert "fig99_99_OOPS" in extras


def test_check_artifacts_frozen_pass():
    contract = {
        "figures": [{"id": "fig99_1"}, {"id": "fig99_2"}],
        "tables": [{"id": "tab99_1"}],
    }
    ok, not_frozen = check_artifacts_frozen(contract, FIXTURE / "sample_registry.csv")
    assert ok
    assert not_frozen == []


def test_check_artifacts_frozen_detects_planned(tmp_path):
    csv_path = tmp_path / "registry.csv"
    csv_path.write_text(
        "id,type,chapter,title,caption,source_kind,source_path,artifact_path,source_hash,artifact_hash,status,body_or_appendix,last_verified,notes\n"
        "fig99_1,figure,99,T,,,,,,,planned,body,,\n",
        encoding="utf-8",
    )
    contract = {"figures": [{"id": "fig99_1"}], "tables": []}
    ok, not_frozen = check_artifacts_frozen(contract, csv_path)
    assert not ok
    assert "fig99_1" in not_frozen


def test_check_conclusions_block_present_and_long_enough():
    text = "## Выводы\n- A\n- B\n- C\n- D\n"
    ok, n = check_conclusions_block(text, required=3)
    assert ok
    assert n == 4


def test_check_conclusions_block_missing():
    text = "просто текст без блока выводов"
    ok, n = check_conclusions_block(text, required=3)
    assert not ok


def test_check_forbidden_phrases_finds_VIF():
    text = "Подробная VIF-таблица показывает..."
    forbidden = ["VIF-таблиц"]
    ok, hits = check_forbidden_phrases(text, forbidden)
    assert not ok
    assert any("VIF" in h for h in hits)


def test_extract_numbers_finds_decimals_and_percents():
    text = "значение 17.4%, что соответствует +11.60% эффекту и горизонту 2 дня"
    nums = extract_numbers(text)
    nums_str = [n[0] for n in nums]
    assert "17.4" in nums_str
    assert "11.60" in nums_str
    assert "2" in nums_str


def test_check_citations_resolve(tmp_path):
    bib = tmp_path / "bibliography.bib"
    bib.write_text("@book{efron1993bootstrap,\n  title={X},\n  author={E},\n  year={1993}\n}\n", encoding="utf-8")
    text = "См. [@efron1993bootstrap] для деталей."
    ok, unresolved = check_citations_resolve(text, bib)
    assert ok
    assert unresolved == []


def test_check_citations_unresolved(tmp_path):
    bib = tmp_path / "bibliography.bib"
    bib.write_text("", encoding="utf-8")
    text = "См. [@efron1993bootstrap]."
    ok, unresolved = check_citations_resolve(text, bib)
    assert not ok
    assert "efron1993bootstrap" in unresolved
