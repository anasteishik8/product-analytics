"""Тесты для scripts/vkr_plot_style.py."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from vkr_plot_style import apply_vkr_style, VKR_PALETTE, save_figure  # noqa: E402


def test_apply_sets_serif_font():
    apply_vkr_style()
    fam = matplotlib.rcParams["font.family"]
    fams = fam if isinstance(fam, list) else [fam]
    joined = " ".join(str(x) for x in fams).lower()
    assert "serif" in joined or "times" in joined


def test_apply_disables_top_right_spines():
    apply_vkr_style()
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False


def test_palette_has_at_least_4_colors():
    assert len(VKR_PALETTE) >= 4
    for c in VKR_PALETTE:
        assert isinstance(c, str) and c.startswith("#")


def test_save_figure_writes_pdf(tmp_path):
    apply_vkr_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    out = tmp_path / "test.pdf"
    save_figure(fig, out)
    assert out.exists()
    assert out.stat().st_size > 100
    plt.close(fig)
