"""Pre-defense demo check.

Запуск: python scripts/demo_check.py
Exit 0 — всё готово к защите.
Exit != 0 — что-то блокирующее.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "data/processed/floodit_final.parquet",
    "results/final_config.json",
    "results/horizon_safe.csv",
    "results/forecast_validation_recursive_summary.csv",
    "results/scenario_analysis_summary.csv",
    "results/viability_summary.csv",
    "results/viability_decomposition.csv",
    "results/demo_forecast_predictions.csv",
    "app/streamlit/Home.py",
    "app/streamlit/pages/01_Data.py",
    "app/streamlit/pages/02_Forecast.py",
    "app/streamlit/pages/03_Scenarios.py",
    "app/streamlit/pages/04_Verdict.py",
    "requirements.lock",
]


def _check_files() -> list[str]:
    missing = []
    for rel in REQUIRED_FILES:
        if not (PROJECT_ROOT / rel).exists():
            missing.append(rel)
    return missing


def _check_exporter_deterministic() -> bool:
    out = PROJECT_ROOT / "results" / "demo_forecast_predictions.csv"
    h1 = hashlib.sha256(out.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "export_demo_artifacts.py")],
                   check=True, cwd=PROJECT_ROOT)
    h2 = hashlib.sha256(out.read_bytes()).hexdigest()
    return h1 == h2


def _run_tests() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "not extrapolation"],
        cwd=PROJECT_ROOT,
    )
    return result.returncode == 0


def main() -> int:
    print("== Demo Check ==")

    print("\n[1/3] Files...")
    missing = _check_files()
    if missing:
        print("FAIL: missing files:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print("OK")

    print("\n[2/3] Exporter determinism...")
    if not _check_exporter_deterministic():
        print("FAIL: exporter is non-deterministic")
        return 2
    print("OK")

    print("\n[3/3] Tests (non-extrapolation)...")
    if not _run_tests():
        print("FAIL: tests")
        return 3
    print("OK")

    print("\n== ALL CHECKS PASSED ==")
    print("Ready for defense. Run: streamlit run app/streamlit/Home.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
