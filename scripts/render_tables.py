"""render_tables.py — пакетная пересборка LaTeX-таблиц из CSV-файлов результатов.

Запускает по очереди модули make_chN_tables, каждый из которых читает
соответствующие файлы из results/ и записывает .tex в vkr/v2/artifacts/tables/.

Использование:
    python scripts/render_tables.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "make_ch1_tables.py",
    "make_ch2_tables.py",
    "make_ch3_tables.py",
    "make_ch4_tables.py",
    "make_ch6_tables.py",
]


def main() -> int:
    failed = []
    for name in SCRIPTS:
        path = ROOT / "scripts" / name
        if not path.exists():
            print(f"skip: {name} not found")
            continue
        print(f"=== {name} ===")
        result = subprocess.run([sys.executable, str(path)])
        if result.returncode != 0:
            failed.append(name)
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
