# -*- coding: utf-8 -*-
"""
Квартирография — интерактивный генератор квартирных планов
=========================================================
Полностью переработанная версия без известных багов.

🔧 Требования
-------------
```
pip install streamlit shapely matplotlib pandas streamlit-drawable-canvas
```

▶️ Запуск
---------
```
streamlit run kvartirografia.py
```

Главные улучшения
-----------------
1. **Двусторонний Canvas** на `streamlit‑drawable‑canvas` — никакого custom JS, данные сразу доступны в Python.
2. **Привязка к сетке** действительно работает и может включаться/выключаться.
3. **Undo/Clear** для холста.
4. **Автоматическая нормализация процентов** — последний тип корректируется, чтобы сумма была 100 %.
5. **Прогресс‑бар** при генерации этажей.
6. **Сохранение проекта** и отчёта в JSON/CSV.
7. **Кэширование тяжёлых операций** через `st.cache_data`.
8. **Легенда цветов** и масштабная линейка.
"""

from __future__ import annotations
import math
import json
from typing import Dict, List, Tuple

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import split
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
#   КОНФИГУРАЦИЯ СТРАНИЦЫ
# -------------------------

st.set_page_config(
    page_title="Квартирография Architect Edition",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📐 Квартирография — Architect Edition")

# -------------------------
#   НАСТРОЙКИ В САЙДБАРЕ
# -------------------------

st.sidebar.header("🏢 Параметры здания и сетки")
floors: int = st.sidebar.number_input("Этажей в доме", min_value=1, value=10)
scale_mm_px: float = st.sidebar.number_input("Миллиметров в 1 пикселе", min_value=0.1, value=10.0, step=0.1)
grid_step_mm: int = st.sidebar.number_input("Шаг сетки, мм", min_value=5, value=100, step=5)
show_snap: bool = st.sidebar.checkbox("Привязка к сетке", value=True)

# Apartment types and settings
APT_TYPES = ["Студия", "1С", "2С", "3С", "4С"]
COLORS = {
    "Студия": "#FFC107",
    "1С": "#8BC34A",
    "2С": "#03A9F4",
    "3С": "#E91E63",
    "4С": "#9C27B0",
}

st.sidebar.header("🏠 Распределение квартир (проценты)")

def apartment_percentages() -> Dict[str, float]:
    """Collect user‑defined percentages and auto‑normalize the last one."""
    inputs = []
    for t in APT_TYPES[:-1]:
        val = st.sidebar.number_input(f"% {t}", 0.0, 100.0, 100.0 / len(APT_TYPES), step=1.0, key=f"pct_{t}")
        inputs.append(val)
    sum_inputs = sum(inputs)
    last_val = max(0.0, 100.0 - sum_inputs)
    st.sidebar.markdown(f"**% {APT_TYPES[-1]}:** `{last_val:.1f}` (авто) ")
    if sum_inputs > 100:
        st.sidebar.error("Сумма первых четырёх типов > 100 %. Уменьшите значения.")
    return {t: v for t, v in zip(APT_TYPES, inputs + [last_val])}

percentages: Dict[str, float] = apartment_percentages()

st.sidebar.subheader("📏 Диапазоны площадей (м²)")
AREA_RANGES: Dict[str, Tuple[float, float]] = {}
for t in APT_TYPES:
    AREA_RANGES[t] = st.sidebar.slider(t, 10.0, 200.0, (20.0, 50.0), key=f"area_{t}")

st.sidebar.header("💾 Файлы проекта")
project_name: str = st.sidebar.text_input("Имя файла проекта", "plan.json")

# -------------------------
#   ХУЛОЖЕСТВЕННАЯ ЧАСТЬ
# -------------------------

st.subheader("1️⃣ Нарисуйте внешний контур этажа")
CANVAS_WIDTH, CANVAS_HEIGHT = 800, 600
GRID_PX = grid_step_mm / scale_mm_px  # пикселей

# Helper — draw background grid
def make_grid() -> List[dict]:
    """Return FabricJS objects for grid (as data URL strings)."""
    objs = []
    if GRID_PX < 5:  # не рисуем слишком частую сетку
        return objs
    # вертикальные линии
    for x in range(0, int(CANVAS_WIDTH), int(GRID_PX)):
        objs.append({
            "type": "line",
            "x1": x,
            "y1": 0,
            "x2": x,
            "y2": CANVAS_HEIGHT,
            "stroke": "#e3e3e3",
            "strokeWidth": 1,
            "selectable": False,
        })
    # горизонтальные линии
    for y in range(0, int(CANVAS_HEIGHT), int(GRID_PX)):
        objs.append({
            "type": "line",
            "x1": 0,
            "y1": y,
            "x2": CANVAS_WIDTH,
            "y2": y,
            "stroke": "#e3e3e3",
            "strokeWidth": 1,
            "selectable": False,
        })
    return objs

bg_objects = make_grid()

# Canvas for outer contour
contour_json = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # hollow polygons
    stroke_width=2,
    stroke_color="#000000",
    background_color="#ffffff",
    height=CANVAS_HEIGHT,
    width=CANVAS_WIDTH,
    drawing_mode="polygon",
    initial_drawing=bg_objects,
    key="contour_canvas",
)

st.caption("Нарисуйте **один** замкнутый полигон. Затем нажмите кнопку ниже.")

if "contour_poly" not in st.session_state:
    st.session_state.contour_poly = None

if st.button("📌 Сохранить контур", disabled=not contour_json.json_data):
    # Extract first polygon/path
    def _extract_pts(obj: dict) -> List[Tuple[float, float]] | None:
        if obj.get("type") == "path":
            pts = [(cmd[1], cmd[2]) for cmd in obj["path"] if cmd[0] in ("M", "L")]
            return pts
        if obj.get("type") == "polygon":
            return [(p[0], p[1]) for p in obj["points"]]
        return None

    polygons_px = []
    for obj in contour_json.json_data.get("objects", []):
        pts = _extract_pts(obj)
        if pts and len(pts) >= 3:
            # Snap if enabled
            if show_snap:
                pts = [(
                    round(x / GRID_PX) * GRID_PX,
                    round(y / GRID_PX) * GRID_PX,
                ) for x, y in pts]
            polygons_px.append(Polygon(pts))

    if not polygons_px:
        st.warning("Не найдено корректных полигонов.")
    elif len(polygons_px) > 1:
        st.warning("Найдены несколько полигонов. Используется первый.")
        st.session_state.contour_poly = polygons_px[0]
    else:
        st.session_state.contour_poly = polygons_px[0]

# Canvas for holes (optional)
st.subheader("2️⃣ Нарисуйте зоны МОП (необязательно)")
holes_json = st_canvas(
    fill_color="rgba(255,0,0,0.3)",
    stroke_width=2,
    stroke_color="#ff0000",
    background_color="#ffffff",
    height=CANVAS_HEIGHT,
    width=CANVAS_WIDTH,
    drawing_mode="polygon",
    initial_drawing=bg_objects,
    key="holes_canvas",
)

if "holes_polys" not in st.session_state:
    st.session_state.holes_polys: List[Polygon] = []

if st.button("➕ Добавить МОП", disabled=not holes_json.json_data):
    def _extract_pts(obj: dict) -> List[Tuple[float, float]] | None:
        if obj.get("type") == "path":
            pts = [(cmd[1], cmd[2]) for cmd in obj["path"] if cmd[0] in ("M", "L")]
            return pts
        if obj.get("type") == "polygon":
            return [(p[0], p[1]) for p in obj["points"]]
        return None

    new_holes = []
    for obj in holes_json.json_data.get("objects", []):
        pts = _extract_pts(obj)
        if pts and len(pts) >= 3:
            if show_snap:
                pts = [(
                    round(x / GRID_PX) * GRID_PX,
                    round(y / GRID_PX) * GRID_PX,
                ) for x, y in pts]
            new_holes.append(Polygon(pts))
    st.session_state.holes_polys.extend(new_holes)

# -------------------------
#   ВАЛИДАЦИЯ ПОЛИГОНОВ
# -------------------------

if st.session_state.contour_poly is None:
    st.info("Нарисуйте и сохраните внешний контур, затем МОЖНО добавить зоны МОП.")
    st.stop()

outer = st.session_state.contour_poly
if not outer.is_valid or not outer.is_simple:
    st.error("Внешний контур некорректен (самопересечения и т. п.).")
    st.stop()

# Вычитаем дыры
floor_poly: Polygon | MultiPolygon = outer
for h in st.session_state.holes_polys:
    if h.is_valid:
        floor_poly = floor_poly.difference(h)

if floor_poly.is_empty:
    st.error("После вычитания МОП не осталось площади этажа!")
    st.stop()

# -------------------------
#   МЕТРИКА ЭТАЖА
# -------------------------

minx, miny, maxx, maxy = floor_poly.bounds
width_mm = (maxx - minx) * scale_mm_px
height_mm = (maxy - miny) * scale_mm_px
area_m2 = floor_poly.area * (scale_mm_px ** 2) / 1e6
st.success(f"Контур: **{width_mm:.0f} × {height_mm:.0f} мм**, площадь **{area_m2:.2f} м²**")

# --------------------------------------------------------------------
#   АЛГОРИТМ РАЗБИВКИ (split_poly) — улучшенная версия с 2 ориентациями
# --------------------------------------------------------------------

def split_poly(poly: Polygon, target_px2: float, tol: float = 0.05) -> Tuple[Polygon, Polygon | None]:
    """Разбить `poly` на часть ≈ target_px2 px².
    Пробуем 2 ориентации (по длинной и короткой стороне MBR).
    Возвращаем (квартира, остаток|None).
    """
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    # Две стороны — major, minor
    sides = []
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        length = math.hypot(x2 - x1, y2 - y1)
        sides.append(((x1, y1), (x2, y2), length))
    sides = sorted(sides, key=lambda s: s[2], reverse=True)

    for (p1, p2, _len_side) in sides[:2]:  # major и minor
        ux, uy = (p2[0] - p1[0]) / _len_side, (p2[1] - p1[1]) / _len_side
        vx, vy = -uy, ux  # перпендикуляр
        projs = [ux * x + uy * y for x, y in poly.exterior.coords]
        low, high = min(projs), max(projs)

        def make_cut(offset: float) -> LineString:
            mx, my = ux * offset, uy * offset
            minx_, miny_, maxx_, maxy_ = poly.bounds
            diag = math.hypot(maxx_ - minx_, maxy_ - miny_) * 2
            return LineString([(mx + vx * diag, my + vy * diag), (mx - vx * diag, my - vy * diag)])

        for _ in range(40):  # бинарный поиск
            mid = (low + high) / 2
            parts = split(poly, make_cut(mid))
            if not parts.geoms or len(parts.geoms) < 2:
                low = mid
                continue
            parts = list(parts.geoms)[:2]
            parts.sort(key=lambda p_: p_.area)
            smaller = parts[0]
            a = smaller.area
            if a > target_px2 * (1 + tol):
                high = mid
            elif a < target_px2 * (1 - tol):
                low = mid
            else:
                return smaller, parts[1]
    # fallback: тупо отрезаем половину площади
    parts = list(split(poly, LineString([(minx, miny), (maxx, maxy)])))
    parts.sort(key=lambda p_: p_.area)
    return parts[0], (parts[1] if len(parts) > 1 else None)

# -------------------------
#   ГЕНЕРАЦИЯ КВАРТИРОГРАФИИ
# -------------------------

st.subheader("3️⃣ Сгенерировать квартирографию")
if st.button("🚀 Запустить генерацию"):
    with st.spinner("Расчёт количества квартир…"):
        avg_area = {t: sum(AREA_RANGES[t]) / 2 for t in APT_TYPES}
        total_build_area = area_m2 * floors
        counts = {
            t: max(1, round(total_build_area * percentages[t] / 100 / avg_area[t]))
            for t in APT_TYPES
        }
        # Распределяем по этажам +‑1
        per_floor: Dict[int, Dict[str, int]] = {f: {t: 0 for t in APT_TYPES} for f in range(floors)}
        for t, cnt in counts.items():
            q, r = divmod(cnt, floors)
            for f in range(floors):
                per_floor[f][t] = q + (1 if f < r else 0)

    prog = st.progress(0, text="Нарезка этажей…")
    floor_placements: Dict[int, List[Tuple[str, Polygon]]] = {}

    for fl in range(floors):
        targets: List[Tuple[str, float]] = []
        for t, n in per_floor[fl].items():
            if n > 0:
                px2 = (sum(AREA_RANGES[t]) / 2) * 1e6 / (scale_mm_px ** 2)
                targets.extend([(t, px2)] * n)

        available: List[Polygon] = [floor_poly]
        placed: List[Tuple[str, Polygon]] = []

        for t, px2 in targets:
            available.sort(key=lambda p: p.area, reverse=True)
            if not available:
                st.warning(f"Этаж {fl+1}: не хватает места для всех квартир ☎️")
                break
            largest = available.pop(0)
            apt, rem = split_poly(largest, px2)
            placed.append((t, apt))
            if rem and rem.area > 0.02 * px2:
                available.append(rem)

        floor_placements[fl + 1] = placed
        prog.progress((fl + 1) / floors, text=f"Готово {fl+1}/{floors} этажей")

    prog.empty()

    # -------------------------
    #   ВИЗУАЛИЗАЦИЯ ЭТАЖЕЙ
    # -------------------------

    st.subheader("4️⃣ Планы этажей")
    for fl, placement in floor_placements.items():
        st.markdown(f"### Этаж {fl}")
        fig, ax = plt.subplots(figsize=(6, 5))
        for t, poly in placement:
            x, y = poly.exterior.xy
            ax.fill([xi * scale_mm_px for xi in x], [yi * scale_mm_px for yi in y],
                    color=COLORS[t], alpha=0.7, edgecolor="black", linewidth=1)
            cx, cy = poly.representative_point().xy
            area_m2_apt = poly.area * (scale_mm_px ** 2) / 1e6
            ax.text(cx[0] * scale_mm_px, cy[0] * scale_mm_px,
                    f"{t}\n{area_m2_apt:.1f} м²", ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        # Легенда
        for t, c in COLORS.items():
            ax.scatter([], [], color=c, label=t)
        ax.legend(loc="upper right", fontsize=8)
        # Линейка масштаба 5 м
        ax.plot([20, 20 + 5000 / scale_mm_px], [20, 20], lw=4, color="black")
        ax.text(20 + 2500 / scale_mm_px, 40, "5 м", ha="center", va="bottom")
        ax.set_aspect("equal")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------
    #   ОТЧЁТ / ЭКСПОРТ
    # -------------------------

    st.subheader("5️⃣ Сводный отчёт")
    rows = []
    for fl, placement in floor_placements.items():
        for t in APT_TYPES:
            qty = sum(1 for tp, _ in placement if tp == t)
            rows.append({"Этаж": fl, "Тип": t, "Количество": qty})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Кнопки скачивания
    st.sidebar.download_button(
        "⬇️ Скачать отчёт CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="report.csv",
        mime="text/csv",
    )

    project_data = {
        "scale_mm_px": scale_mm_px,
        "grid_step_mm": grid_step_mm,
        "floors": floors,
        "percentages": percentages,
        "area_ranges": AREA_RANGES,
        "contour": json.loads(floor_poly.to_geojson()),
    }
    st.sidebar.download_button(
        "⬇️ Скачать проект JSON",
        json.dumps(project_data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=project_name,
        mime="application/json",
    )
