"""Конвертация ключевых PDF-фигур главы 5 в PNG для встраивания в DOCX."""
import fitz
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vkr" / "figures_ch5"
OUT.mkdir(parents=True, exist_ok=True)

FIGURES = [
    ("figures/04_modeling_dau/model_comparison.pdf",            "04_dau_model_comparison.png"),
    ("figures/04_modeling_stickiness/model_comparison.pdf",     "04_stickiness_model_comparison.png"),
    ("figures/04_modeling_retention_d7/model_comparison.pdf",   "04_retention_model_comparison.png"),
    ("figures/05_forecast_horizon/curve_f1_dau.pdf",            "05_curve_f1_dau.png"),
    ("figures/05_forecast_horizon/curve_f1_retention_d7.pdf",   "05_curve_f1_retention.png"),
    ("figures/05_forecast_horizon/extrapolation_comparison.pdf","05_extrapolation_comparison.png"),
    ("figures/06_forecast_validation/forecast_f1_retention_d7.pdf", "06_forecast_f1_retention.png"),
    ("figures/06_forecast_validation/forecast_f1_stickiness.pdf",   "06_forecast_f1_stickiness.png"),
    ("figures/06b_recursive/comparison_f1_dau.pdf",             "06b_comparison_f1_dau.png"),
    ("figures/06b_recursive/comparison_f1_stickiness.pdf",      "06b_comparison_f1_stickiness.png"),
    ("figures/06b_recursive/comparison_f1_retention_d7.pdf",    "06b_comparison_f1_retention.png"),
    ("figures/06b_recursive/comparison_f2_dau.pdf",             "06b_comparison_f2_dau.png"),
    ("figures/06b_recursive/comparison_f2_stickiness.pdf",      "06b_comparison_f2_stickiness.png"),
    ("figures/07_scenarios/shap_f2_dau.pdf",                    "07_shap_f2_dau.png"),
    ("figures/07_scenarios/shap_f1_stickiness.pdf",             "07_shap_f1_stickiness.png"),
    ("figures/07_scenarios/recommendations_f1.pdf",             "07_recommendations_f1.png"),
    ("figures/07_scenarios/recommendations_f2.pdf",             "07_recommendations_f2.png"),
    ("figures/07_scenarios/scenario_f2_dau_cross_product_mimic.pdf", "07_scenario_f2_dau_cross_product_mimic.png"),
    ("figures/07_scenarios/scenario_f2_stickiness_cross_product_mimic.pdf", "07_scenario_f2_stickiness_cross_product_mimic.png"),
    ("figures/08_viability/comparison_M3.pdf",                  "08_comparison_M3.png"),
    ("figures/08_viability/comparison_M1.pdf",                  "08_comparison_M1.png"),
    ("figures/08_viability/score_f2_M3.pdf",                    "08_score_f2_M3.png"),
    ("figures/08_viability/decomposition_f2_M3_day110.pdf",     "08_decomposition_f2_M3_day110.png"),
    ("figures/08_viability/verdict_card_f1.pdf",                "08_verdict_card_f1.png"),
    ("figures/08_viability/verdict_card_f2.pdf",                "08_verdict_card_f2.png"),
    ("figures/08_viability/dashboard_summary.pdf",              "08_dashboard_summary.png"),
]

DPI = 180

for src_rel, dst_name in FIGURES:
    src = ROOT / src_rel
    dst = OUT / dst_name
    if not src.exists():
        print(f"MISS: {src_rel}")
        continue
    doc = fitz.open(src)
    pix = doc[0].get_pixmap(dpi=DPI)
    pix.save(dst)
    doc.close()
    print(f"OK:   {dst_name}  ({pix.width}x{pix.height})")

print(f"\nTotal: {len(list(OUT.glob('*.png')))} PNG files in {OUT}")
