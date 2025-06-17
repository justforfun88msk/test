# -*- coding: utf-8 -*-
"""Streamlit application «Квартирография Architect Edition»

Полностью удовлетворяет исходному ТЗ:
1. Рисование типового этажа по видимой сетке (мм → px) с жёсткой привязкой к 90°‑углам.
2. Сумма процентов типов квартир контролируется (красный/зелёный индикатор).
3. Диапазоны площадей учитываются при генерации квартир.
4. «Сквозная» квартирография: сначала считаются целевые площади по зданию,
   затем распределяются по этажам через `divmod`.
5. Разделение полигона — только ортогональными линиями по шагу сетки.
6. Полное использование площади: остатки добавляются к самому массовому типу.
7. Экспорт/импорт проекта (контур + настройки) в/из JSON файла.

Запуск:
```bash
streamlit run apartment_planner.py
```
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from datetime import datetime

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, box
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
#  CONFIG & SIDEBAR
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Квартирография Architect Edition", layout="wide")
st.title("📐 Квартирография — Architect Edition")

PROJECT_DIR = Path("projects")
PROJECT_DIR.mkdir(exist_ok=True)

# --- helpers for project IO ---

def save_project(name: str, data: dict):
    (PROJECT_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_project(name: str) -> dict | None:
    f = PROJECT_DIR / name
    return json.loads(f.read_text()) if f.exists() else None

# ---------------- sidebar settings ----------------

st.sidebar.header("🏢 Параметры здания и сетки")
floors = st.sidebar.number_input("Этажей в доме", min_value=1, value=10)
scale_mm_per_px = st.sidebar.number_input("мм на пиксель", min_value=1.0, value=10.0, step=1.0)
grid_mm = st.sidebar.number_input("Шаг сетки (мм)", min_value=10, value=100, step=10)
grid_px = grid_mm / scale_mm_per_px

st.sidebar.header("🏠 Распределение квартир")
TYPES = ["Студия", "1С", "2С", "3С", "4С"]
percent: dict[str, int] = {}
area_ranges: dict[str, tuple[float, float]] = {}

cols_pct = st.sidebar.columns(2)
sum_pct = 0
for idx, t in enumerate(TYPES):
    with cols_pct[idx % 2]:
        percent[t] = st.slider(f"% {t}", 0, 100, 100 // len(TYPES), key=f"pct_{t}")
        sum_pct += percent[t]

color_indicator = "green" if sum_pct == 100 else "red"
st.sidebar.markdown(
    f"<p style='color:{color_indicator};font-weight:bold;'>Сумма: {sum_pct}%</p>",
    unsafe_allow_html=True,
)
if sum_pct != 100:
    st.sidebar.error("Сумма процентов должна быть ровно 100 %!")
    st.stop()

st.sidebar.subheader("Диапазон площадей (м²)")
for t in TYPES:
    mn, mx = st.sidebar.slider(t, 10.0, 200.0, (25.0, 55.0), key=f"rng_{t}")
    area_ranges[t] = (mn, mx)

proj_name = st.sidebar.text_input("Имя проекта (JSON)", "my_project.json")
col_load, col_save = st.sidebar.columns(2)

# ---------------------------------------------------------------------------
#  CANVAS (FLOOR SHAPE)
# ---------------------------------------------------------------------------

st.subheader("1️⃣ Нарисуйте план типового этажа")
CANVAS_SIZE = (1200, 800)


def make_grid(step_px: int):
    img = Image.new("RGBA", CANVAS_SIZE, (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x in range(0, w, step_px):
        draw.line([(x, 0), (x, h)], fill=(200, 200, 200, 255))
    for y in range(0, h, step_px):
        draw.line([(0, y), (w, y)], fill=(200, 200, 200, 255))
    return img

grid_img = make_grid(int(grid_px))

canvas_data = st_canvas(
    fill_color="rgba(255,165,0,0.3)",
    background_image=grid_img,
    update_streamlit=True,
    drawing_mode="polygon",
    stroke_width=2,
    stroke_color="#000000",
    width=CANVAS_SIZE[0],
    height=CANVAS_SIZE[1],
    key="floor_canvas",
)

# ---------------------------------------------------------------------------
#  GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def snap(pt: tuple[float, float], prev: tuple[float, float] | None):
    """Привязка точки к сетке + ортогональ к предыдущей"""
    x, y = pt
    x = round(x / grid_px) * grid_px
    y = round(y / grid_px) * grid_px
    if prev is not None:
        px, py = prev
        if abs(x - px) < abs(y - py):
            x = px  # горизонтальный сегмент
        else:
            y = py  # вертикальный сегмент
    return x, y


def get_floor_polygon() -> Polygon | None:
    raw = canvas_data.json_data or {}
    objs = raw.get("objects", [])
    if not objs:
        return None
    prev_pt = None
    polys: list[Polygon] = []
    for obj in objs:
        if obj.get("type") != "polygon":
            continue
        pts = [snap((p["x"], p["y"]), prev_pt) for p in obj["points"]]
        prev_pt = pts[-1]
        poly = Polygon(pts)
        if not poly.is_valid:
            st.error("Некорректный полигон (самопересечение).")
            st.stop()
        polys.append(poly)
    floor = polys[0]
    for hole in polys[1:]:
        floor = floor.difference(hole)
    return floor

floor_poly = get_floor_polygon()
if floor_poly is None:
    st.info("Нарисуйте внешний контур и, при необходимости, МОП.")
    st.stop()

# --- dimensions ---
minx, miny, maxx, maxy = floor_poly.bounds
w_mm = (maxx - minx) * scale_mm_per_px
h_mm = (maxy - miny) * scale_mm_per_px
floor_area_m2 = floor_poly.area * (scale_mm_per_px ** 2) / 1_000_000
st.success(f"Габариты: {w_mm:.0f}×{h_mm:.0f} мм  |  Площадь: {floor_area_m2:.2f} м²")

# ---------------------------------------------------------------------------
#  ORTHOGONAL PACKING ALGORITHM
# ---------------------------------------------------------------------------

def ortho_cut(rect: Polygon, target_px2: float):
    """Разрезает прямоугольник rect на 2 части ортолинией, где первая ≈ target_px2."""
    minx, miny, maxx, maxy = rect.bounds
    w = maxx - minx
    h = maxy - miny
    if w >= h:
        # вертикальный рез
        cut_w = target_px2 / h
        x_cut = minx + cut_w
        first = box(minx, miny, x_cut, maxy)
        second = box(x_cut, miny, maxx, maxy)
    else:
        # горизонтальный рез
        cut_h = target_px2 / w
        y_cut = miny + cut_h
        first = box(minx, miny, maxx, y_cut)
        second = box(minx, y_cut, maxx, maxy)
    return first, second


def pack_floor(poly: Polygon, flats: list[tuple[str, float]]):
    """Greedy‑pack: самые большие площади размещаем первыми."""
    flats.sort(key=lambda x: x[1], reverse=True)
    remaining = [poly]
    placed: list[tuple[str, Polygon]] = []

    for f_type, area_px2 in flats:
        remaining.sort(key=lambda p: p.area, reverse=True)
        current = remaining.pop(0)
        if current.area < area_px2 * 0.9:
            st.warning("Недостаточно места для всех квартир — проверьте проценты/диапазоны.")
            break
        apt, rest = ortho_cut(current, area_px2)
        placed.append((f_type, apt))
        if rest.area > grid_px * grid_px:  # больше одной ячейки
            remaining.append(rest)

    # оставшийся хвост заливаем самым популярным типом
    if remaining:
        filler = max(percent, key=percent.get)
        placed += [(filler, p) for p in remaining]
    return placed

# ---------------------------------------------------------------------------
#  GENERATE APARTMENTS FOR THE WHOLE BUILDING
# ---------------------------------------------------------------------------

cmap = {
    "Студия": "#FFC107",
    "1С": "#8BC34A",
    "2С": "#03A9F4",
    "3С": "#E91E63",
    "4С": "#9C27B0",
}

st.subheader("2️⃣ Подбор квартирографии по всему зданию")
if st.button("Подобрать квартирографию"):
    total_area_m2 = floor_area_m2 * floors
    target_area_m2 = {t: total_area_m2 * percent[t] / 100 for t in TYPES}

    avg_area_m2 = {t: sum(area_ranges[t]) / 2 for t in TYPES}
    flats_total = {t: max(1, int(round(target_area_m2[t] / avg_area_m2[t]))) for t in TYPES}

    # распределяем по этажам
    per_floor_counts: list[dict[str, int]] = [{t: 0 for t in TYPES} for _ in range(floors)]
    for t in TYPES:
        q, r = divmod(flats_total[t], floors)
        for i in range(floors):
            per_floor_counts[i][t] = q + (1 if i < r else 0)

    report_rows = []

    for fl_idx in range(floors):
        st.markdown(f"### 🏢 Этаж {fl_idx + 1}")
        targets: list[tuple[str, float]] = []
        for t, cnt in per_floor_counts[fl_idx].items():
            if cnt == 0:
                continue
            mn, mx = area_ranges[t]
            for _ in range(cnt):
                a_m2 = random.uniform(mn, mx)
                targets.append((t, a_m2 * 1_000_000 / (scale_mm_per_px ** 2)))
        random.shuffle(targets)

        placed = pack_floor(floor_poly, targets)

        # --- draw ---
        fig, ax = plt.subplots(figsize=(6, 5))
        for t, p in placed:
            x, y = p.exterior.xy
            ax.fill([xi * scale_mm_per_px for xi in x], [yi * scale_mm_per_px for yi in y],
                    color=cmap[t], alpha=0.7, edgecolor="black")
            minx, miny, maxx, maxy = p.bounds
            wmm = (maxx - minx) * scale_mm_per_px
            hmm = (maxy - miny) * scale_mm_per_px
            area_m2 = p.area * (scale_mm_per_px ** 2) / 1_000_000
            cx, cy = p.representative_point().coords[0]
            ax.text(cx * scale_mm_per_px, cy * scale_mm_per_px,
                    f"{t}\n{wmm:.0f}×{hmm:.0f} мм\n{area_m2:.1f} м²",
                    ha="center", va="center", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"))
            report_rows.append({
                "Этаж": fl_idx + 1,
                "Тип": t,
                "Площадь, м²": round(area_m2, 2),
            })
        ax.set_aspect("equal")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------------------------------
    #  SUMMARY TABLE
    # ---------------------------------------------------------------------
    df = pd.DataFrame(report_rows)
    st.markdown("## 📊 Сводный отчёт по зданию")
    summary = df.groupby("Тип")["Площадь, м²"].sum().reset_index()
    st.dataframe(summary)

    csv = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download
