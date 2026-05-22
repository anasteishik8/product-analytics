"""
csv_to_latex_table.py — генератор LaTeX-таблиц (booktabs) для ВКР.

Usage as library:
    from csv_to_latex_table import df_to_latex_booktabs
    df_to_latex_booktabs(df, Path("tab.tex"), columns={...}, caption="...", label="tab:x")

Pure-Python генерация .tex без pandas.to_latex (deprecated в pandas 2.x).
XeLaTeX-совместимый UTF-8 output.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def _escape_latex(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in text)


def _format_cell(value, float_fmt: str) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return float_fmt.format(value)
    return _escape_latex(str(value))


def df_to_latex_booktabs(
    df: pd.DataFrame,
    out_path: Path,
    columns: Mapping[str, str],
    caption: str,
    label: str,
    float_fmt: str = "{:.2f}",
    alignment: str | None = None,
) -> Path:
    """Записывает DataFrame как booktabs LaTeX-таблицу.

    columns : Mapping src_col → display_name. Порядок ключей определяет порядок колонок.
    """
    src_cols = list(columns.keys())
    display = [columns[c] for c in src_cols]
    n = len(src_cols)
    if alignment is None:
        alignment = "l" * n
    sub = df[src_cols]
    lines: list[str] = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_escape_latex(h) for h in display) + r" \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        cells = [_format_cell(row[c], float_fmt) for c in src_cols]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
