"""check_registry.py — проверка целостности артефактов по реестру SHA-256.

Читает vkr/v2/artifacts/registry.csv, вычисляет SHA-256 каждого файла из
колонки artifact_path и сверяет первые 16 hex-символов с зафиксированным
значением artifact_hash. Возвращает 0, если все хэши совпадают, и 1 при
расхождениях или отсутствии файлов.

Использование:
    python scripts/check_registry.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "vkr" / "v2" / "artifacts" / "registry.csv"

HASH_LEN = 16  # первые 16 hex-символов SHA-256


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:HASH_LEN].lower()


def main() -> int:
    if not REGISTRY.exists():
        print(f"ERROR: registry not found: {REGISTRY}", file=sys.stderr)
        return 2

    ok = 0
    mismatched: list[tuple[str, str, str]] = []
    missing: list[str] = []

    with REGISTRY.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"artifact_path", "artifact_hash"}
        if not required.issubset(set(reader.fieldnames or [])):
            print(
                f"ERROR: registry must contain columns {sorted(required)}",
                file=sys.stderr,
            )
            return 2

        for row in reader:
            rel = (row.get("artifact_path") or "").strip()
            expected = (row.get("artifact_hash") or "").strip().lower()
            if not rel or not expected:
                continue

            target = (ROOT / rel).resolve()
            if not target.exists():
                missing.append(rel)
                continue

            actual = sha256_short(target)
            if actual == expected:
                ok += 1
            else:
                mismatched.append((rel, expected, actual))

    print(f"OK:       {ok}")
    print(f"MISSING:  {len(missing)}")
    for m in missing:
        print(f"  - {m}")
    print(f"MISMATCH: {len(mismatched)}")
    for path, exp, act in mismatched:
        print(f"  - {path}")
        print(f"      expected: {exp}")
        print(f"      actual:   {act}")

    return 0 if (not missing and not mismatched) else 1


if __name__ == "__main__":
    sys.exit(main())
