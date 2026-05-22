"""
freeze_artifact.py — обновляет 1 строку в vkr/v2/artifacts/registry.csv.

Usage as CLI:
    python scripts/freeze_artifact.py fig5_1 \
        --source-kind results \
        --source results/horizon_safe.csv \
        --artifact vkr/v2/artifacts/figures/fig5_1.pdf \
        --caption "..." \
        --status frozen

Usage as library:
    from freeze_artifact import update_row
    update_row(registry_csv, "fig5_1", status="frozen", ...)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


REGISTRY_FIELDS = [
    "id", "type", "chapter", "title", "caption", "source_kind",
    "source_path", "artifact_path", "source_hash", "artifact_hash",
    "status", "body_or_appendix", "last_verified", "notes",
]


def _sha256_short(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def update_row(registry_csv: Path, id_: str, **fields: str) -> None:
    """Обновляет одну строку CSV. Считает хэши если переданы source_path/artifact_path.
    Ставит last_verified=today если status переключён на 'frozen' или 'verified'.
    """
    registry_csv = Path(registry_csv)
    with registry_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    target = next((r for r in rows if r["id"] == id_), None)
    if target is None:
        raise KeyError(f"id {id_} not found in {registry_csv}")

    for k, v in fields.items():
        if k not in REGISTRY_FIELDS:
            raise ValueError(f"unknown field: {k}")
        target[k] = str(v) if v is not None else ""

    if "source_path" in fields and fields["source_path"]:
        sp = Path(fields["source_path"])
        if sp.exists() and sp.is_file():
            target["source_hash"] = _sha256_short(sp)

    if "artifact_path" in fields and fields["artifact_path"]:
        ap = Path(fields["artifact_path"])
        if ap.exists() and ap.is_file():
            target["artifact_hash"] = _sha256_short(ap)

    if fields.get("status") in ("verified", "frozen"):
        target["last_verified"] = dt.date.today().isoformat()

    with registry_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REGISTRY_FIELDS})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Update one row in vkr/v2/artifacts/registry.csv.")
    p.add_argument("id", help="Artifact id (e.g. fig5_1)")
    p.add_argument("--registry", type=Path, default=Path("vkr/v2/artifacts/registry.csv"))
    p.add_argument("--source-kind")
    p.add_argument("--source", dest="source_path")
    p.add_argument("--artifact", dest="artifact_path")
    p.add_argument("--caption")
    p.add_argument("--status", choices=["planned", "generated", "verified", "frozen", "stale"])
    p.add_argument("--body-or-appendix", dest="body_or_appendix",
                   choices=["body", "appA", "appB", "appC", "appD", "appE"])
    p.add_argument("--notes")
    args = p.parse_args(argv)

    fields = {k: v for k, v in vars(args).items()
              if k not in ("id", "registry") and v is not None}
    update_row(args.registry, args.id, **fields)
    print(f"OK updated {args.id} in {args.registry}: {list(fields.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
