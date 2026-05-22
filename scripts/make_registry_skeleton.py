"""
make_registry_skeleton.py — собирает vkr/v2/artifacts/registry.csv из контрактов.

Usage:
    python scripts/make_registry_skeleton.py

Читает все vkr/v2/contracts/*.md, парсит YAML frontmatter, извлекает
figures/tables, формирует registry.csv. Существующие строки не перетирает —
только добавляет новые id со статусом 'planned'.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml


REGISTRY_FIELDS = [
    "id",
    "type",
    "chapter",
    "title",
    "caption",
    "source_kind",
    "source_path",
    "artifact_path",
    "source_hash",
    "artifact_hash",
    "status",
    "body_or_appendix",
    "last_verified",
    "notes",
]


def parse_contract(path: Path) -> dict[str, Any]:
    """Парсит YAML frontmatter из markdown-файла контракта."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: нет YAML frontmatter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: незакрытый YAML frontmatter")
    return yaml.safe_load(parts[1])


def build_registry_rows(contract: dict[str, Any]) -> list[dict[str, str]]:
    """Строит список row-dict для одного контракта."""
    chapter = str(contract.get("chapter", ""))
    rows: list[dict[str, str]] = []

    for fig in contract.get("figures") or []:
        rows.append(_row(fig, "figure", chapter))
    for tab in contract.get("tables") or []:
        rows.append(_row(tab, "table", chapter))

    return rows


def _row(item: dict[str, str], kind: str, chapter: str) -> dict[str, str]:
    return {
        "id": item["id"],
        "type": kind,
        "chapter": chapter,
        "title": item.get("title", ""),
        "caption": "",
        "source_kind": "",
        "source_path": item.get("source", ""),
        "artifact_path": "",
        "source_hash": "",
        "artifact_hash": "",
        "status": "planned",
        "body_or_appendix": "body",
        "last_verified": "",
        "notes": "",
    }


def merge_with_existing(new_rows: list[dict[str, str]], existing_csv: Path) -> list[dict[str, str]]:
    """Объединяет new_rows со существующим registry.csv, не перетирая
    статусы и заполнённые поля для известных id."""
    if not existing_csv.exists():
        return new_rows

    by_id: dict[str, dict[str, str]] = {}
    with existing_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_id[row["id"]] = row

    merged: list[dict[str, str]] = []
    for new in new_rows:
        if new["id"] in by_id:
            existing = by_id[new["id"]]
            # title из контракта побеждает; всё остальное — из existing если непусто
            existing["title"] = new["title"]
            existing["chapter"] = new["chapter"]
            existing["type"] = new["type"]
            # source_path обновляем из контракта только если в existing пусто
            if not existing.get("source_path"):
                existing["source_path"] = new["source_path"]
            merged.append(existing)
        else:
            merged.append(new)
    return merged


def write_registry(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r["chapter"], r["type"], r["id"])):
            w.writerow({k: row.get(k, "") for k in REGISTRY_FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build registry skeleton from VKR contracts.")
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=Path("vkr/v2/contracts"),
        help="Директория с контрактами глав",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("vkr/v2/artifacts/registry.csv"),
        help="Куда писать registry.csv",
    )
    args = parser.parse_args(argv)

    contract_paths = sorted(args.contracts_dir.glob("*.md"))
    if not contract_paths:
        print(f"Нет контрактов в {args.contracts_dir}", file=sys.stderr)
        return 1

    all_rows: list[dict[str, str]] = []
    for p in contract_paths:
        contract = parse_contract(p)
        all_rows.extend(build_registry_rows(contract))

    merged = merge_with_existing(all_rows, args.out)
    write_registry(merged, args.out)
    print(f"Записан {args.out}: {len(merged)} строк из {len(contract_paths)} контрактов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
