"""Конвертация ключевых PDF-фигур главы 2 в PNG для встраивания в DOCX."""
import fitz
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vkr" / "figures_ch2"
OUT.mkdir(parents=True, exist_ok=True)

FIGURES = [
    # §2.1 — источники данных (NaN-карты как индикатор покрытия)
    ("figures/01_source_product/nan_heatmap.pdf",     "01_product_nan_heatmap.png"),
    ("figures/01_source_product/timeseries_product.pdf", "01_product_timeseries.png"),
    ("figures/01_source_store/nan_heatmap.pdf",       "01_store_nan_heatmap.png"),
    ("figures/01_source_store/histograms_store.pdf",  "01_store_histograms.png"),
    ("figures/01_source_market/histograms_market.pdf","01_market_histograms.png"),
    ("figures/01_source_market/nan_heatmap.pdf",      "01_market_nan_heatmap.png"),
    ("figures/01_source_external/timeseries_external.pdf", "01_external_timeseries.png"),

    # §2.4 — EDA Flood-It!
    ("figures/02_eda_floodit/histograms_product.pdf", "02_f1_histograms_product.png"),
    ("figures/02_eda_floodit/timeseries_product.pdf", "02_f1_timeseries_product.png"),
    ("figures/02_eda_floodit/corr_key_20x20.pdf",     "02_f1_corr_key_20x20.png"),
    ("figures/02_eda_floodit/pairplot_key6.pdf",      "02_f1_pairplot_key6.png"),
    ("figures/02_eda_floodit/nan_heatmap.pdf",        "02_f1_nan_heatmap.png"),
    ("figures/02_eda_floodit/unified_timeseries.pdf", "02_f1_unified_timeseries.png"),

    # §2.4 — EDA Flood-It! 2 (главное — nan_heatmap, store_partial)
    ("figures/02_eda_floodit2/nan_heatmap.pdf",       "02_f2_nan_heatmap.png"),
    ("figures/02_eda_floodit2/timeseries_product.pdf","02_f2_timeseries_product.png"),
    ("figures/02_eda_floodit2/timeseries_store_partial.pdf", "02_f2_timeseries_store_partial.png"),
    ("figures/02_eda_floodit2/unified_timeseries.pdf","02_f2_unified_timeseries.png"),

    # §2.5 — Сравнение f1 vs f2
    ("figures/03_comparison/delta_medians_bar.pdf",   "03_delta_medians_bar.png"),
    ("figures/03_comparison/overlay_timeseries_6.pdf","03_overlay_timeseries_6.png"),
    ("figures/03_comparison/early_predictors.pdf",    "03_early_predictors.png"),

    # §2.7 — VIF-фильтрация
    ("figures/02_eda_floodit/vif_top15.pdf",          "02_f1_vif_top15.png"),
    ("figures/02_eda_floodit2/vif_top15.pdf",         "02_f2_vif_top15.png"),
]

DPI = 180
for src_rel, dst_name in FIGURES:
    src = ROOT / src_rel
    if not src.exists():
        print(f"MISS: {src_rel}")
        continue
    dst = OUT / dst_name
    doc = fitz.open(src)
    pix = doc[0].get_pixmap(dpi=DPI)
    pix.save(dst)
    doc.close()
    print(f"OK:   {dst_name} ({pix.width}x{pix.height})")

print(f"\nTotal: {len(list(OUT.glob('*.png')))} PNG files in {OUT}")
