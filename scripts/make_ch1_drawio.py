"""
make_ch1_drawio.py — генерирует .drawio-исходники для рисунков главы 1.

Создаёт два файла в `vkr/v2/artifacts/figures/`:
  fig1_1.drawio — контур принятия продуктового решения (циклическая схема)
  fig1_2.drawio — диаграмма вариантов использования системы (UML use-case)

Используется минимальный билдер Diagram, скопированный из make_ch3_drawio.py,
с добавлением методов `ellipse` (для use case'ов) и `actor` (для UML-actor'ов).
"""
from __future__ import annotations

import math
import sys
import xml.sax.saxutils as _xml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = ROOT / "vkr" / "v2" / "artifacts" / "figures"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
    return _xml.escape(text).replace("\n", "&#10;")


class Diagram:
    """Минимальный билдер drawio-XML диаграммы."""

    def __init__(self, name: str, page_w: int = 1100, page_h: int = 1500) -> None:
        self.name = name
        self.page_w = page_w
        self.page_h = page_h
        self.cells: list[str] = []
        self.next_id = 2

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

    def ellipse(
        self, x: int, y: int, w: int, h: int, label: str,
        fill: str = BLUE_BG, stroke: str = PRIMARY_BLUE,
        stroke_width: int = 1, font_size: int = 12, bold: bool = False,
    ) -> int:
        cid = self.next_id
        self.next_id += 1
        font_style = "fontStyle=1;" if bold else ""
        style = (
            f"ellipse;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"fontSize={font_size};{font_style}align=center;verticalAlign=middle;"
            f"fontFamily=Times New Roman;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f"</mxCell>"
        )
        return cid

    def actor(self, x: int, y: int, w: int, h: int, label: str) -> int:
        """UML actor: stick figure со встроенным шейпом drawio."""
        cid = self.next_id
        self.next_id += 1
        style = (
            "shape=umlActor;verticalLabelPosition=bottom;labelBackgroundColor=none;"
            "verticalAlign=top;html=1;outlineConnect=0;"
            "fillColor=#ffffff;strokeColor=#555555;"
            "fontSize=12;fontFamily=Times New Roman;"
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
        style_extra: str = "", color: str = GRAY,
    ) -> int:
        cid = self.next_id
        self.next_id += 1
        style = (
            f"endArrow=classic;html=1;strokeColor={color};strokeWidth=1.5;"
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
        stroke_width: int = 2, font_size: int = 12, bold: bool = True,
    ) -> int:
        cid = self.next_id
        self.next_id += 1
        font_style = "fontStyle=1;" if bold else ""
        style = (
            f"rhombus;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"fontSize={font_size};{font_style}align=center;verticalAlign=middle;"
            f"fontFamily=Times New Roman;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" '
            f'style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f"</mxCell>"
        )
        return cid

    def line(self, source: int, target: int, color: str = GRAY, lw: int = 1) -> int:
        """Простая линия без стрелки (для UML use-case ассоциаций)."""
        cid = self.next_id
        self.next_id += 1
        style = (
            f"endArrow=none;html=1;strokeColor={color};strokeWidth={lw};"
            f"fontFamily=Times New Roman;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" '
            f'style="{style}" edge="1" source="{source}" target="{target}" parent="1">'
            f'<mxGeometry relative="1" as="geometry"/>'
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
# fig1_1 — Контур принятия продуктового решения (циклический)
# ============================================================
def build_fig1_1() -> str:
    """Swimlane: слева источники данных, справа — шаги аналитика
    с decision diamond и двумя исходами (вердикт / итерация).
    """
    d = Diagram("fig1_1 — Контур принятия продуктового решения",
                page_w=1500, page_h=1300)

    # --- Фоновые подложки полос + заголовки ---
    # Левая полоса
    d.box(20, 40, 360, 1230, "",
          fill="#fafafa", stroke=GRAY, stroke_width=1, font_size=10)
    d.text(20, 50, 360, 30, "Источники данных",
           font_size=14, italic=False)
    # Правая полоса
    d.box(400, 40, 1080, 1230, "",
          fill="#ffffff", stroke=GRAY, stroke_width=1, font_size=10)
    d.text(400, 50, 1080, 30, "Аналитик",
           font_size=14, italic=False)

    # ─── Левая полоса: 4 источника ───────────────────────────
    src_x = 50
    src_w = 300
    src_h = 80
    src_labels = [
        "Firebase BigQuery",
        "Google Play /\nstore metadata",
        "Kaggle market snapshot",
        "Внешние сигналы:\nGoogle Trends, Wikipedia",
    ]
    src_ys = [120, 250, 380, 510]
    src_ids: list[int] = []
    for y, lbl in zip(src_ys, src_labels):
        cid = d.box(src_x, y, src_w, src_h, lbl,
                    fill=GRAY_BG, stroke=GRAY,
                    stroke_width=1, font_size=11)
        src_ids.append(cid)
    # Стрелки между источниками
    for i in range(len(src_ids) - 1):
        d.arrow(src_ids[i], src_ids[i + 1])

    # ─── Правая полоса: 4 шага ───────────────────────────────
    step_x = 480
    step_w = 380
    step_h = 110
    step_ys = [120, 280, 440, 600]
    step_titles = [
        ("1. Диагностика",
         "оценка качества данных, структура\nпродуктовых метрик"),
        ("2. Прогноз",
         "оценка траекторий ключевых метрик\nна безопасном горизонте"),
        ("3. Сценарии",
         "сценарный анализ управляемых\nпродуктовых рычагов"),
        ("4. Рекомендации",
         "ранжирование действий по\nожидаемому эффекту на метрики"),
    ]
    step_ids: list[int] = []
    for y, (title, sub) in zip(step_ys, step_titles):
        label = f"{title}\n\n{sub}"
        cid = d.box(step_x, y, step_w, step_h, label,
                    fill=BLUE_BG, stroke=PRIMARY_BLUE,
                    stroke_width=2, font_size=12, bold=True)
        step_ids.append(cid)
    # Стрелки между шагами
    for i in range(len(step_ids) - 1):
        d.arrow(step_ids[i], step_ids[i + 1])

    # ─── Стикеры-аннотации справа от каждого шага ───────────
    note_x = 920
    note_w = 530
    notes_data = [
        ("Итоговый набор данных: 222 × 60\n"
         "(подмножество признаков подбирается под конкретную модель)", 90),
        ("Целевые метрики:\nstickiness, retention_d7, DAU", 80),
        ("Управляемые признаки:\n"
         "crash_rate, onboarding_completion_rate, median_session_duration", 100),
        ("Результат шага: список приоритетных\n"
         "продуктовых изменений с оценкой эффекта", 90),
    ]
    note_ids: list[int] = []
    for (note_text, n_h), y_step in zip(notes_data, step_ys):
        y_note = y_step + (step_h - n_h) // 2
        cid = d.box(note_x, y_note, note_w, n_h, note_text,
                    fill="#fff8d6", stroke=OCHRE,
                    stroke_width=1, font_size=11)
        note_ids.append(cid)
    # Пунктирная связь шаг → стикер
    for s, n in zip(step_ids, note_ids):
        d.line(s, n, color=OCHRE, lw=1)

    # ─── Стрелка от последнего источника к шагу 1 ───────────
    d.arrow(src_ids[-1], step_ids[0],
            style_extra="strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
            color=PRIMARY_BLUE)

    # ─── Decision diamond ───────────────────────────────────
    diamond_x = step_x + 40
    diamond_y = 770
    diamond_w = 300
    diamond_h = 120
    diamond_id = d.diamond(diamond_x, diamond_y, diamond_w, diamond_h,
                           "Достаточна ли надёжность\nпрогноза и сценария?",
                           fill=OCHRE_BG, stroke=OCHRE)
    # Стрелка шаг 4 → ромб
    d.arrow(step_ids[3], diamond_id)

    # ─── Два исхода: вердикт (слева) и итерация (справа) ────
    verdict_x = 430
    verdict_y = 970
    verdict_w = 380
    verdict_h = 110
    verdict_id = d.box(verdict_x, verdict_y, verdict_w, verdict_h,
                       "5. Вердикт\n\nразвивать / мониторить / закрывать",
                       fill=GREEN_BG, stroke=GREEN,
                       stroke_width=3, font_size=12, bold=True)

    iter_x = 920
    iter_y = 970
    iter_w = 530
    iter_h = 110
    iter_id = d.box(iter_x, iter_y, iter_w, iter_h,
                    "Уточнить данные, расширить наблюдения,\nпересобрать сценарии",
                    fill=OCHRE_BG, stroke=OCHRE,
                    stroke_width=1, font_size=12)

    # Стрелки от ромба к двум исходам с подписями «да» / «нет»
    d.arrow(diamond_id, verdict_id, label="да",
            style_extra="strokeWidth=2;", color=GREEN)
    d.arrow(diamond_id, iter_id, label="нет",
            style_extra="strokeWidth=2;", color=OCHRE)

    # ─── Terminal node (●) ──────────────────────────────────
    term_x = verdict_x + verdict_w // 2 - 25
    term_y = verdict_y + verdict_h + 50
    term_id = d.ellipse(term_x, term_y, 50, 50, "",
                        fill="#000000", stroke="#000000", stroke_width=2)
    d.arrow(verdict_id, term_id, style_extra="strokeWidth=2;")

    return d.render()


# ============================================================
# fig1_2 — UML use-case диаграмма
# ============================================================
def build_fig1_2() -> str:
    """Границы прототипа и условия применения (не use-case).

    Защитный рисунок: показывает что разрабатывается прототип с заданным
    контрактом входных данных, а не универсальная промышленная система.
    """
    d = Diagram("fig1_2 — Границы прототипа",
                page_w=900, page_h=900)

    cx = 450
    box_w = 540

    # 1) Входные данные
    b1 = d.box(cx - box_w // 2, 60, box_w, 80,
               "Исторические данные продукта\nпо заданной схеме признаков",
               fill=GRAY_BG, stroke=GRAY, font_size=12)

    # 2) Прототип системы (главный блок)
    b2 = d.box(cx - box_w // 2, 220, box_w, 90,
               "Прототип системы прогнозирования\nвостребованности",
               fill=GREEN_BG, stroke=GREEN, stroke_width=3,
               font_size=13, bold=True)

    # 3) Выходы
    b3 = d.box(cx - box_w // 2, 400, box_w, 110,
               "Прогноз метрик · Доверительные интервалы\n"
               "Сценарный анализ · Рекомендации · Вердикт",
               fill=BLUE_BG, stroke=PRIMARY_BLUE, font_size=12)

    # 4) Пользователь
    b4 = d.box(cx - box_w // 2, 600, box_w, 100,
               "Пользователь системы — аналитик или другой\n"
               "специалист, интерпретирующий продуктовые метрики",
               fill=OCHRE_BG, stroke=OCHRE, font_size=12)

    d.arrow(b1, b2)
    d.arrow(b2, b3)
    d.arrow(b3, b4)

    return d.render()


def main() -> int:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "fig1_1": build_fig1_1,
        "fig1_2": build_fig1_2,
    }
    for name, build in builders.items():
        path = FIGS_DIR / f"{name}.drawio"
        path.write_text(build(), encoding="utf-8")
        print(f"OK {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
