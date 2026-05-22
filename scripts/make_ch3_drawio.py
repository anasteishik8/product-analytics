"""
make_ch3_drawio.py — генерирует .drawio-исходники для 5 диаграмм главы 3.

Создаёт XML-файлы в формате drawio (https://app.diagrams.net),
которые пользователь открывает локально и подгоняет масштаб/расположение вручную.

Файлы создаются в `vkr/v2/artifacts/figures/`:
  fig3_1.drawio — общая методическая схема системы
  fig3_3.drawio — recursive forecast с лаг-признаками
  fig3_5.drawio — Monte Carlo сценарный анализ
  fig3_6.drawio — SHAP-мост через operational whitelist
  fig3_7.drawio — vердикт через Mahalanobis + bad_share

fig3_2 (3 семейства моделей) и fig3_4 (гистограмма bootstrap) — математические
иллюстрации, в drawio не переносятся; остаются только в matplotlib.
"""
from __future__ import annotations

import sys
import xml.sax.saxutils as _xml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = ROOT / "vkr" / "v2" / "artifacts" / "figures"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --- Палитра (та же, что vkr_plot_style.VKR_PALETTE) ---
PRIMARY_BLUE = "#2c5d8a"
RED = "#a83232"
GREEN = "#3a8a64"
OCHRE = "#c08a30"
GRAY = "#555555"

BLUE_BG = "#e3edf5"
RED_BG = "#fbe6e6"
GREEN_BG = "#e6f1ec"
OCHRE_BG = "#fbf2e3"
GRAY_BG = "#f0f0f0"


def _esc(text: str) -> str:
    """XML-escape + перевод строк в drawio-совместимый формат."""
    return _xml.escape(text).replace("\n", "&#10;")


class Diagram:
    """Минимальный билдер drawio-XML диаграммы."""

    def __init__(self, name: str, page_w: int = 1100, page_h: int = 1500) -> None:
        self.name = name
        self.page_w = page_w
        self.page_h = page_h
        self.cells: list[str] = []
        self.next_id = 2  # 0 и 1 зарезервированы (root + default parent)

    def box(
        self, x: int, y: int, w: int, h: int, label: str,
        fill: str = "#ffffff", stroke: str = GRAY,
        stroke_width: int = 1, font_size: int = 12,
        bold: bool = False, align: str = "center",
    ) -> int:
        cid = self.next_id
        self.next_id += 1
        font_style = "fontStyle=1;" if bold else ""
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"fontSize={font_size};{font_style}align={align};verticalAlign=middle;"
            f"fontFamily=Times New Roman;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f"</mxCell>"
        )
        return cid

    def text(
        self, x: int, y: int, w: int, h: int, label: str,
        font_size: int = 11, italic: bool = False, align: str = "center",
    ) -> int:
        cid = self.next_id
        self.next_id += 1
        font_style = "fontStyle=2;" if italic else ""
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;"
            f"fontSize={font_size};{font_style}align={align};verticalAlign=middle;"
            f"fontFamily=Times New Roman;fontColor={GRAY};"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f"</mxCell>"
        )
        return cid

    def arrow(
        self, source: int, target: int, label: str = "",
        style_extra: str = "",
    ) -> int:
        cid = self.next_id
        self.next_id += 1
        style = (
            f"endArrow=classic;html=1;strokeColor={GRAY};strokeWidth=1.5;"
            f"fontSize=10;fontFamily=Times New Roman;{style_extra}"
        )
        label_attr = f'value="{_esc(label)}" ' if label else ""
        self.cells.append(
            f'<mxCell id="{cid}" {label_attr}'
            f'style="{style}" edge="1" source="{source}" target="{target}" parent="1">'
            f'<mxGeometry relative="1" as="geometry"/>'
            f"</mxCell>"
        )
        return cid

    def diamond(
        self, x: int, y: int, w: int, h: int, label: str,
        fill: str = OCHRE_BG, stroke: str = OCHRE,
    ) -> int:
        cid = self.next_id
        self.next_id += 1
        style = (
            f"rhombus;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
            f"fontSize=11;fontFamily=Times New Roman;align=center;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f"</mxCell>"
        )
        return cid

    def render(self) -> str:
        return (
            '<mxfile host="app.diagrams.net" version="22.0.0">'
            f'<diagram name="{_esc(self.name)}">'
            f'<mxGraphModel dx="800" dy="600" grid="1" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{self.page_w}" pageHeight="{self.page_h}" math="0" shadow="0">'
            f"<root>"
            f'<mxCell id="0"/>'
            f'<mxCell id="1" parent="0"/>'
            f"{''.join(self.cells)}"
            f"</root>"
            f"</mxGraphModel>"
            f"</diagram>"
            f"</mxfile>"
        )


# ============================================================
# fig3_1 — Общая методическая схема системы
# ============================================================
def build_fig3_1() -> str:
    d = Diagram("fig3_1 — Методическая схема системы", page_w=1100, page_h=1600)

    cx = 550
    box_w = 360
    box_h = 70

    # 1. Данные
    b1 = d.box(cx - box_w // 2, 40, box_w, box_h,
               "Данные и признаки", fill=BLUE_BG, stroke=PRIMARY_BLUE)

    # 2. Wrapper "Модели-кандидаты"
    wrap_w = 920
    wrap_h = 130
    wrap_x = cx - wrap_w // 2
    wrap_y = 160
    d.box(wrap_x, wrap_y, wrap_w, 30,
          "Модели-кандидаты", fill=OCHRE_BG, stroke=OCHRE,
          font_size=12, bold=True)
    # 6 мини-блоков
    candidates = ["Ridge", "Nadaraya–Watson", "LocalLinear",
                  "RandomForest", "XGBoost", "kNN"]
    mini_w = 130
    mini_h = 60
    gap = 15
    total_w = len(candidates) * mini_w + (len(candidates) - 1) * gap
    start_x = cx - total_w // 2
    mini_ids: list[int] = []
    for i, name in enumerate(candidates):
        mx = start_x + i * (mini_w + gap)
        my = wrap_y + 50
        cid = d.box(mx, my, mini_w, mini_h, name,
                    fill="#ffffff", stroke=OCHRE, font_size=11)
        mini_ids.append(cid)
    # invisible "anchor" для стрелки в группу моделей
    b2_anchor = d.box(cx - 10, wrap_y + wrap_h - 5, 20, 10, "",
                       fill="none", stroke="none")

    # 3. Feature selection
    b3 = d.box(cx - box_w // 2, 320, box_w, box_h,
               "Отбор признаков и нормализация",
               fill=BLUE_BG, stroke=PRIMARY_BLUE)

    # 4. CV / initial training
    b4 = d.box(cx - box_w // 2, 430, box_w, box_h,
               "Кросс-валидация и начальное обучение",
               fill=BLUE_BG, stroke=PRIMARY_BLUE)

    # 5. Recursive forecast validation
    b5 = d.box(cx - box_w // 2, 540, box_w, box_h,
               "Recursive forecast validation (14 дней)",
               fill=OCHRE_BG, stroke=OCHRE, font_size=12, bold=True)

    # 6. Working model (KEY HIGHLIGHT)
    b6 = d.box(cx - box_w // 2 - 20, 660, box_w + 40, 90,
               "Рабочая модель\nдля каждой пары (продукт, метрика)",
               fill=GREEN_BG, stroke=GREEN, stroke_width=3,
               font_size=13, bold=True)

    # 7. Scenarios + SHAP
    b7 = d.box(cx - box_w // 2, 800, box_w, box_h,
               "Сценарный анализ + SHAP",
               fill=BLUE_BG, stroke=PRIMARY_BLUE)

    # 8. Viability verdict
    b8 = d.box(cx - box_w // 2, 910, box_w, box_h,
               "Viability verdict",
               fill=RED_BG, stroke=RED, font_size=12, bold=True)

    # Стрелки: 1 → wrapper → 3 → 4 → 5 → 6 → 7 → 8
    # Стрелка от b1 к wrapper (используем верхний центр wrapper-блока — фиктивный anchor сверху)
    wrapper_top = d.box(cx - 10, wrap_y - 5, 20, 10, "", fill="none", stroke="none")
    d.arrow(b1, wrapper_top)
    d.arrow(b2_anchor, b3)
    d.arrow(b3, b4)
    d.arrow(b4, b5)
    d.arrow(b5, b6, label="выбор по метрикам качества",
            style_extra="strokeWidth=2;")
    d.arrow(b6, b7)
    d.arrow(b7, b8)

    return d.render()


# ============================================================
# fig3_3 — Recursive forecast с лаг-признаками
# ============================================================
def build_fig3_3() -> str:
    d = Diagram("fig3_3 — Recursive forecast", page_w=1400, page_h=550)

    box_w = 220
    box_h = 110
    gap = 50
    cy = 180

    titles = [
        "Состояние T\ntarget_lag1, target_lag7,\nпризнаки",
        "Шаг h=1\nпредсказать ŷ_T+1\nобновить lag_1 ← ŷ_T+1",
        "Шаг h=2\nпредсказать ŷ_T+2\nобновить lag_1 ← ŷ_T+2",
        "…",
        "Шаг h=14\nпредсказать ŷ_T+14\nфинальный прогноз",
    ]

    ids: list[int] = []
    for i, t in enumerate(titles):
        x = 30 + i * (box_w + gap)
        is_dots = (t == "…")
        fill = "#ffffff" if is_dots else BLUE_BG
        stroke = GRAY if is_dots else PRIMARY_BLUE
        font_size = 18 if is_dots else 11
        ids.append(d.box(x, cy, box_w, box_h, t,
                         fill=fill, stroke=stroke, font_size=font_size))

    for i in range(len(ids) - 1):
        d.arrow(ids[i], ids[i + 1])

    # Подпись внизу
    d.text(50, cy + box_h + 30, 1300, 30,
           "На каждом шаге предсказанное значение становится новым target_lag_1 для следующего шага",
           font_size=11, italic=True)

    return d.render()


# ============================================================
# fig3_5 — Monte Carlo сценарный анализ
# ============================================================
def build_fig3_5() -> str:
    d = Diagram("fig3_5 — Monte Carlo сценарий", page_w=1100, page_h=900)

    cx = 550
    box_w = 460
    box_h = 80

    b1 = d.box(cx - box_w // 2, 30, box_w, box_h,
               "Опорное состояние:\nзначения признаков в текущей точке",
               fill=GRAY_BG, stroke=GRAY)

    b2 = d.box(cx - box_w // 2, 160, box_w, 100,
               "Изменение управляемых признаков:\ncrash_rate, onboarding_completion_rate, median_session_duration",
               fill=BLUE_BG, stroke=PRIMARY_BLUE)

    b3 = d.box(cx - box_w // 2, 310, box_w, box_h,
               "N симуляций с шумом (seed=42, N=1000)",
               fill=GREEN_BG, stroke=GREEN)

    # Y-split: распределение слева, итог справа
    b4 = d.box(cx - box_w // 2 - 100, 450, 360, box_h,
               "Распределение прогнозов",
               fill=BLUE_BG, stroke=PRIMARY_BLUE)
    b5 = d.box(cx + 60, 450, 380, box_h,
               "Медиана + CI95 +\np(превышение опорной траектории)",
               fill=RED_BG, stroke=RED, font_size=11, bold=True)

    d.arrow(b1, b2)
    d.arrow(b2, b3)
    d.arrow(b3, b4)
    d.arrow(b3, b5)
    d.arrow(b4, b5, label="агрегация")

    return d.render()


# ============================================================
# fig3_6 — SHAP → operational whitelist → сценарии
# ============================================================
def build_fig3_6() -> str:
    d = Diagram("fig3_6 — SHAP и operational whitelist", page_w=1100, page_h=900)

    cx = 550
    box_w = 540
    box_h = 75

    b1 = d.box(cx - box_w // 2, 30, box_w, box_h,
               "Обученная модель",
               fill=BLUE_BG, stroke=PRIMARY_BLUE, font_size=12, bold=True)

    b2 = d.box(cx - box_w // 2, 180, box_w, 90,
               "Важные признаки\n(category_rank, crash_rate, retention_d7, "
               "onboarding_completion_rate, …)",
               fill=BLUE_BG, stroke=PRIMARY_BLUE)

    # Промежуточный блок — фильтр (визуально выделить охрой)
    b3 = d.box(cx - box_w // 2, 340, box_w, 70,
               "Operational whitelist\n(только операционно управляемые признаки)",
               fill=OCHRE_BG, stroke=OCHRE, stroke_width=2,
               font_size=12, bold=True)

    b4 = d.box(cx - box_w // 2, 470, box_w, 90,
               "Управляемые признаки\ncrash_rate · onboarding_completion_rate · "
               "median_session_duration",
               fill=GREEN_BG, stroke=GREEN)

    b5 = d.box(cx - box_w // 2, 620, box_w, box_h,
               "Сценарии Monte Carlo",
               fill=RED_BG, stroke=RED, font_size=12, bold=True)

    d.arrow(b1, b2, label="SHAP")
    d.arrow(b2, b3)
    d.arrow(b3, b4, label="фильтр")
    d.arrow(b4, b5)

    return d.render()


# ============================================================
# fig3_7 — Decision tree вердикта через Mahalanobis + bad_share
# ============================================================
def build_fig3_7() -> str:
    d = Diagram("fig3_7 — Verdict logic", page_w=1300, page_h=1100)

    cx = 650
    box_w = 380
    box_h = 70

    b1 = d.box(cx - box_w // 2, 30, box_w, box_h,
               "Прогнозы и признаки дня t",
               fill=BLUE_BG, stroke=PRIMARY_BLUE)

    b2 = d.box(cx - box_w // 2, 160, box_w, box_h,
               "Расстояние Махаланобиса D²\nотносительно опорной нормы",
               fill=BLUE_BG, stroke=PRIMARY_BLUE)

    # Первый ромб
    dia1 = d.diamond(cx - 100, 290, 200, 100, "D² < T₉₅?")

    # Левая ветвь: DEVELOP
    dev = d.box(cx - 500, 420, 260, 70,
                "DEVELOP-день", fill=GREEN_BG, stroke=GREEN,
                stroke_width=2, font_size=12, bold=True)

    # Второй ромб (справа)
    dia2 = d.diamond(cx + 200, 420, 200, 100, "D² < T₉₉?")

    # MONITOR (под вторым ромбом, влево)
    mon = d.box(cx + 30, 580, 260, 70,
                "MONITOR-день", fill=OCHRE_BG, stroke=OCHRE,
                stroke_width=2, font_size=12, bold=True)

    # DISCONTINUE (под вторым ромбом, вправо)
    disc = d.box(cx + 450, 580, 260, 70,
                 "DISCONTINUE-день", fill=RED_BG, stroke=RED,
                 stroke_width=2, font_size=12, bold=True)

    # Агрегация
    agg = d.box(cx - box_w // 2, 750, box_w, 80,
                "Доля плохих дней (bad_share)\nза период наблюдения",
                fill=GRAY_BG, stroke=GRAY)

    # Финальный вердикт
    final = d.box(cx - box_w // 2 - 30, 880, box_w + 60, 90,
                  "Итоговый вердикт продукта:\nDEVELOP · MONITOR · DISCONTINUE",
                  fill="#ffffff", stroke=PRIMARY_BLUE, stroke_width=3,
                  font_size=13, bold=True)

    d.arrow(b1, b2)
    d.arrow(b2, dia1)
    d.arrow(dia1, dev, label="да")
    d.arrow(dia1, dia2, label="нет")
    d.arrow(dia2, mon, label="да")
    d.arrow(dia2, disc, label="нет")
    d.arrow(dev, agg)
    d.arrow(mon, agg)
    d.arrow(disc, agg)
    d.arrow(agg, final)

    return d.render()


# ============================================================
# main
# ============================================================
def main() -> int:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "fig3_1": build_fig3_1,
        "fig3_3": build_fig3_3,
        "fig3_5": build_fig3_5,
        "fig3_6": build_fig3_6,
        "fig3_7": build_fig3_7,
    }
    for name, build in builders.items():
        path = FIGS_DIR / f"{name}.drawio"
        path.write_text(build(), encoding="utf-8")
        print(f"OK {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
