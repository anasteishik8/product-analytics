"""make_fig4_1.py — fig4_1: Архитектура системы / поток данных.

Рисунок собирается из PlantUML-исходника `scripts/fig4_1.puml`:
PlantUML рендерит PNG (200 DPI), затем ImageMagick оборачивает его в PDF.
В отличие от прежней matplotlib-версии, layout — orthogonal, без перекрытия
стрелок и подписей, терминология полностью на русском языке.

Зависимости:
- Java (для plantuml.jar)
- ImageMagick (`magick`) для PNG → PDF
- plantuml.jar (по умолчанию ищется в Cursor-расширении jebbs.plantuml)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUML = ROOT / "scripts" / "fig4_1.puml"
PNG = ROOT / "scripts" / "fig4_1.png"
OUT = ROOT / "vkr" / "v2" / "artifacts" / "figures" / "fig4_1.pdf"

DEFAULT_PLANTUML_JAR = Path(
    r"c:/Users/Анастасия/.cursor/extensions/jebbs.plantuml-2.18.1-universal/plantuml.jar"
)


def _resolve_plantuml_jar() -> Path:
    env = os.environ.get("PLANTUML_JAR")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    if DEFAULT_PLANTUML_JAR.is_file():
        return DEFAULT_PLANTUML_JAR
    raise FileNotFoundError(
        "plantuml.jar не найден. Укажите путь через переменную окружения "
        "PLANTUML_JAR или установите Cursor-расширение jebbs.plantuml."
    )


def _resolve_magick() -> str:
    exe = shutil.which("magick") or shutil.which("convert")
    if exe is None:
        raise FileNotFoundError(
            "ImageMagick (magick/convert) не найден в PATH — требуется для PNG → PDF."
        )
    return exe


def main() -> Path:
    jar = _resolve_plantuml_jar()
    magick = _resolve_magick()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "java", "-jar", str(jar),
            "-charset", "UTF-8",
            "-Sdpi=200",
            "-tpng",
            str(PUML),
        ],
        check=True,
        cwd=str(ROOT),
    )

    if not PNG.is_file():
        raise RuntimeError(f"PlantUML не сгенерировал PNG: {PNG}")

    subprocess.run([magick, str(PNG), str(OUT)], check=True)
    return OUT


if __name__ == "__main__":
    out = main()
    size_kb = out.stat().st_size / 1024
    print(f"OK fig4_1 saved: {out} ({size_kb:.1f} KB)")
