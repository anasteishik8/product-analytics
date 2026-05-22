"""
build_diploma.py — собирает vkr/v2/build/diploma.pdf через pandoc + XeLaTeX.

Usage:
    python scripts/build_diploma.py
    python scripts/build_diploma.py --check-only       # только проверка структуры

Кросплатформенный. Ищет в системе pandoc и xelatex; падает с понятной
ошибкой если их нет.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


CHAPTER_ORDER = [
    "00_intro.md",
    "01_problem.md",
    "02_data.md",
    "03_methods.md",
    "04_architecture.md",
    "05_experiments.md",
    "06_results.md",
    "07_conclusion.md",
]

APPENDIX_ORDER = [
    "appA_features.md",
    "appB_eda_extras.md",
    "appC_modeling_full.md",
    "appD_scenarios_verdict.md",
    "appE_code_run.md",
]


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.stderr.write(
            f"Не найден {name} в PATH.\n"
            f"Windows: установить через winget или скачать с https://pandoc.org\n"
            f"Linux: apt install {name}\n"
        )
        sys.exit(2)
    return path


def collect_inputs(v2: Path) -> list[Path]:
    inputs: list[Path] = []
    chapters_dir = v2 / "chapters"
    appendices_dir = v2 / "appendices"

    for name in CHAPTER_ORDER:
        p = chapters_dir / name
        if p.exists():
            inputs.append(p)
        else:
            sys.stderr.write(f"WARN: нет {p}, пропускаю\n")

    for name in APPENDIX_ORDER:
        p = appendices_dir / name
        if p.exists():
            inputs.append(p)

    if not inputs:
        sys.stderr.write("Нет ни одной главы в vkr/v2/chapters/ — нечего собирать\n")
        sys.exit(3)
    return inputs


def check_only(v2: Path) -> int:
    inputs = collect_inputs(v2)
    print(f"Соберётся PDF из {len(inputs)} файлов:")
    for p in inputs:
        print(f"  - {p.relative_to(v2.parent.parent) if v2.parent.parent in p.parents else p}")
    return 0


def build(v2: Path, base: Path) -> int:
    pandoc = find_tool("pandoc")
    find_tool("xelatex")  # просто проверяем что есть

    inputs = collect_inputs(v2)
    out_pdf = v2 / "build" / "diploma.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    bib = v2 / "bibliography.bib"
    preamble = base / "vkr" / "preamble_gost.tex"

    cmd: list[str] = [
        pandoc,
        *[str(p) for p in inputs],
        "-o", str(out_pdf),
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=2",
        "-V", "lang=ru",
        "-V", "mainfont=Times New Roman",
        "-V", "fontsize=14pt",
        "-V", "linestretch=1.5",
        "-V", "geometry:top=2cm,bottom=2cm,left=3cm,right=1.5cm",
    ]

    if bib.exists() and bib.stat().st_size > 100:  # не пустой stub
        cmd.extend(["--bibliography", str(bib), "--citeproc"])
        csl = base / "vkr" / "gost-r-7-0-5-2008.csl"
        if csl.exists():
            cmd.extend(["--csl", str(csl)])

    if preamble.exists():
        cmd.extend(["-H", str(preamble)])

    print(f"Запуск: pandoc → {out_pdf}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + "\n" + result.stderr + "\n")
        sys.stderr.write(f"pandoc вернул {result.returncode}\n")
        return result.returncode
    print(f"✓ PDF собран: {out_pdf}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка diploma.pdf через pandoc + XeLaTeX.")
    parser.add_argument("--base", type=Path, default=Path("."), help="Корень проекта")
    parser.add_argument("--check-only", action="store_true", help="Только проверить, что собирается")
    args = parser.parse_args(argv)
    v2 = args.base / "vkr" / "v2"
    if args.check_only:
        return check_only(v2)
    return build(v2, args.base)


if __name__ == "__main__":
    sys.exit(main())
