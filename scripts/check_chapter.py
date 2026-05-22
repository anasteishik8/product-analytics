"""
check_chapter.py — технический gate готовности главы ВКР.

Usage:
    python scripts/check_chapter.py 5

Прогоняет 9 технических проверок готовности главы. Печатает ✓/✗/⚠.
Возвращает exit 0 если все блокирующие проверки ✓.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Windows cmd defaults to cp1251 — force UTF-8 so ✓/✗/⚠ and Cyrillic print cleanly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


WORDS_PER_PAGE = 250
AI_MARKERS = [
    "таким образом",
    "стоит отметить",
    "в данной работе рассматривается",
    "необходимо отметить",
    "следует подчеркнуть",
]
AI_MARKER_THRESHOLD = 0.01  # 1% от количества слов


def parse_contract(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: незакрытый YAML frontmatter")
    return yaml.safe_load(parts[1])


def find_contract(chapter_num: int, contracts_dir: Path) -> Path:
    matches = list(contracts_dir.glob(f"{chapter_num:02d}_*.md"))
    if not matches:
        raise FileNotFoundError(f"Нет контракта для главы {chapter_num} в {contracts_dir}")
    return matches[0]


def find_chapter_text(chapter_num: int, chapters_dir: Path) -> Path:
    matches = list(chapters_dir.glob(f"{chapter_num:02d}_*.md"))
    if not matches:
        raise FileNotFoundError(f"Нет текста главы {chapter_num} в {chapters_dir}")
    return matches[0]


# ----- Individual checks -----

def check_page_count(text: str, page_budget: list[int]) -> tuple[bool, str]:
    words = len(text.split())
    pages = words / WORDS_PER_PAGE
    lo, hi = page_budget
    ok = lo <= pages <= hi
    msg = f"{pages:.1f} pp (budget {lo}–{hi}, {words} слов)"
    return ok, msg


def check_artifacts_inserted(text: str, contract: dict[str, Any]) -> tuple[bool, list[str]]:
    ids = [item["id"] for item in (contract.get("figures") or []) + (contract.get("tables") or [])]
    missing = [i for i in ids if not re.search(rf"\b{re.escape(i)}\b", text)]
    return len(missing) == 0, missing


def check_no_extra_artifacts(text: str, contract: dict[str, Any]) -> tuple[bool, list[str]]:
    allowed = {item["id"] for item in (contract.get("figures") or []) + (contract.get("tables") or [])}
    # Строгий паттерн: fig/tab + цифры + подчёркивание + ID — не матчит слова "figures"/"tables"
    found = set(re.findall(r"\b(?:fig|tab)\d+_\w+\b", text))
    extras = sorted(found - allowed)
    return len(extras) == 0, extras


def check_artifacts_frozen(contract: dict[str, Any], registry_csv: Path) -> tuple[bool, list[str]]:
    ids = {item["id"] for item in (contract.get("figures") or []) + (contract.get("tables") or [])}
    not_frozen: list[str] = []
    if not registry_csv.exists():
        return False, list(ids)
    with registry_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["id"] in ids and row["status"] != "frozen":
                not_frozen.append(row["id"])
    return len(not_frozen) == 0, not_frozen


def check_conclusions_block(text: str, required: int) -> tuple[bool, int]:
    m = re.search(r"^##\s*Выводы\s*$(.*?)(?=^##|\Z)", text, re.MULTILINE | re.DOTALL)
    if not m:
        return False, 0
    body = m.group(1)
    bullets = re.findall(r"^\s*[-*]\s+\S+", body, re.MULTILINE)
    return len(bullets) >= required, len(bullets)


def check_forbidden_phrases(text: str, forbidden: list[str]) -> tuple[bool, list[str]]:
    hits: list[str] = []
    lower = text.lower()
    for phrase in forbidden:
        if phrase.lower() in lower:
            hits.append(phrase)
    return len(hits) == 0, hits


def check_citations_resolve(text: str, bib_path: Path) -> tuple[bool, list[str]]:
    cited = set(re.findall(r"\[@([\w\d_-]+)\]|(?<![\w])@([\w\d_-]+)", text))
    cited_keys = {a or b for a, b in cited if a or b}
    if not bib_path.exists():
        return len(cited_keys) == 0, sorted(cited_keys)
    bib_text = bib_path.read_text(encoding="utf-8")
    known_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    unresolved = sorted(cited_keys - known_keys)
    return len(unresolved) == 0, unresolved


def check_ai_markers(text: str) -> tuple[bool, float, list[str]]:
    words = len(text.split())
    if words == 0:
        return True, 0.0, []
    hits = [m for m in AI_MARKERS if m in text.lower()]
    concentration = len(hits) / max(words, 1)
    return concentration <= AI_MARKER_THRESHOLD, concentration, hits


def extract_numbers(text: str) -> list[tuple[str, int]]:
    matches: list[tuple[str, int]] = []
    for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\b", text):
        matches.append((m.group(1), m.start()))
    return matches


# ----- Main reporter -----

def run_checks(chapter_num: int, base: Path = Path(".")) -> int:
    contracts_dir = base / "vkr/v2/contracts"
    chapters_dir = base / "vkr/v2/chapters"
    registry_csv = base / "vkr/v2/artifacts/registry.csv"
    bib_path = base / "vkr/v2/bibliography.bib"

    contract_path = find_contract(chapter_num, contracts_dir)
    chapter_path = find_chapter_text(chapter_num, chapters_dir)

    contract = parse_contract(contract_path)
    chapter_text = chapter_path.read_text(encoding="utf-8")

    title = contract.get("title", f"chapter {chapter_num}")
    print(f"[chapter {chapter_num} — {title}]\n")

    failures = 0

    ok, msg = check_page_count(chapter_text, contract.get("page_budget", [0, 999]))
    print(("✓" if ok else "✗") + f" 1. Page count: {msg}")
    failures += 0 if ok else 1

    ok, missing = check_artifacts_inserted(chapter_text, contract)
    print(("✓" if ok else "✗") + f" 2. Artifacts inserted" + (f" — missing: {missing}" if not ok else ""))
    failures += 0 if ok else 1

    ok, extras = check_no_extra_artifacts(chapter_text, contract)
    print(("✓" if ok else "✗") + f" 3. No extra artifacts" + (f" — extras: {extras}" if not ok else ""))
    failures += 0 if ok else 1

    ok, not_frozen = check_artifacts_frozen(contract, registry_csv)
    print(("✓" if ok else "✗") + f" 4. Artifacts frozen in registry" + (f" — not frozen: {not_frozen}" if not ok else ""))
    failures += 0 if ok else 1

    required = len(contract.get("key_conclusions") or [])
    ok, n = check_conclusions_block(chapter_text, required=max(required, 1))
    print(("✓" if ok else "✗") + f" 5. Conclusions block: {n} items (need ≥{max(required, 1)})")
    failures += 0 if ok else 1

    ok, hits = check_forbidden_phrases(chapter_text, contract.get("must_not_include") or [])
    print(("✓" if ok else "✗") + f" 6. No forbidden phrases" + (f" — hits: {hits}" if not ok else ""))
    failures += 0 if ok else 1

    ok, unresolved = check_citations_resolve(chapter_text, bib_path)
    print(("✓" if ok else "✗") + f" 7. Citations resolve in bibliography.bib" + (f" — unresolved: {unresolved}" if not ok else ""))
    failures += 0 if ok else 1

    nums = extract_numbers(chapter_text)
    print(f"⚠ 8. Numeric values detected: {len(nums)} (verify sources manually)")

    ok_ai, conc, hits = check_ai_markers(chapter_text)
    print(("✓" if ok_ai else "⚠") + f" 9. AI-marker concentration: {conc*100:.2f}% (threshold {AI_MARKER_THRESHOLD*100:.1f}%)")

    print()
    if failures == 0:
        print(f"CHAPTER {chapter_num} READY ✓")
        return 0
    print(f"CHAPTER {chapter_num} NOT READY: {failures} blocking failure(s)")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Технический gate готовности главы ВКР.")
    parser.add_argument("chapter", type=int, help="Номер главы (0=intro, 1-6, 7=conclusion)")
    parser.add_argument("--base", type=Path, default=Path("."), help="Корень проекта")
    args = parser.parse_args(argv)
    return run_checks(args.chapter, args.base)


if __name__ == "__main__":
    sys.exit(main())
