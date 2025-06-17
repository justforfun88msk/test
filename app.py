# -*- coding: utf-8 -*-
"""
Квартирография — интерактивный генератор квартирных планов (Architect Edition)
================================================================================
Исправленная и оптимизированная версия 2025-06-17.

Основные улучшения:
- Нормализация процентов в реальном времени.
- Интерактивная визуализация с Plotly.
- Проверка минимальных размеров в split_poly.
- Валидация полигонов в процессе рисования.
- Оптимизированная производительность и UX.
"""
import base64
import json
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import shapely
import streamlit as st
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
from streamlit_drawable_canvas import st_canvas

# Конфигурация страницы
st.set_page_config(
    page_title="Квартирография Architect Edition",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📐 Квартирография — Architect Edition")
st.markdown("Создавайте планы этажей, задавайте параметры квартир и генерируйте оптимальную квартирографию.")

# Константы
APT_TYPES = ["Студия", "1С", "2С", "3С", "4С"]
COLORS = {
    "Студия": "#FFC107",
    "1С": "#8BC34A",
    "2С": "#03A9F4",
    "3С": "#E91E63",
    "4С": "#9C27B0",
}
CANVAS_WIDTH, CANVAS_HEIGHT = 800, 600
MIN_APT_DIM_MM = 1000  # Минимальная ширина/длина квартиры (1 м)
MIN_FLOOR_AREA_M2 = 10  # Минимальная площадь этажа после вычитания МОП

# Sidebar: Настройки
st.sidebar.header("🏢 Параметры здания и сетки")
floors: int = st.sidebar.number_input("Этажей в доме", min_value=1, value=10, step=1)
scale_mm_px: float = st.sidebar.number_input(
    "Масштаб (мм на пиксель)", min_value=0.1, value=10.0, step=0.1
)
grid_step_mm: int = st.sidebar.number_input(
    "Шаг сетки (мм)", min_value=5, value=100, step=5
)
grid_snap: bool = st.sidebar.checkbox("Привязка к сетке", value=True)

st.sidebar.header("🏠 Распределение квартир")
def apartment_percentages() -> Dict[str, float]:
    """Собирает проценты с нормализацией."""
    percentages = {}
    total = 0
    cols = st.sidebar.columns(2)
    for i, t in enumerate(APT_TYPES):
        with cols[i % 2]:
            val = st.number_input(
                f"% {t}", 0.0, 100.0, 100.0 / len(APT_TYPES), step=1.0, key=f"pct_{t}"
            )
            percentages[t] = val
            total += val
    if abs(total - 100) > 0.01:
        st.sidebar.warning(f"Сумма ({total:.1f}%) нормализована до 100%.")
        for t in percentages:
            percentages[t] = percentages[t] * 100 / total if total > 0 else 100 / len(APT_TYPES)
    color = "green" if abs(total - 100) <= 0.01 else "orange"
    st.sidebar.markdown(f"<p style='color:{color};'>Сумма: {total:.1f}%</p>", unsafe_allow_html=True)
    return percentages

percentages: Dict[str, float] = apartment_percentages()

st.sidebar.subheader("📏 Диапазоны площадей (м²)")
AREA_RANGES: Dict[str, Tuple[float, float]] = {}
for t in APT_TYPES:
    lo, hi = st.sidebar.slider(t, 1.0, 200.0, (20.0, 50.0), key=f"area_{t}")
    if lo >= hi:
        st.sidebar.error(f"Для {t} минимальная площадь не может быть больше или равна максимальной.")
        lo = hi - 1 if hi > 1 else 1
    if lo < 1:
        st.sidebar.warning(f"Минимальная площадь для {t} установлена в 1 м².")
        lo = 1
    AREA_RANGES[t] = (lo, hi)

# Canvas Helpers
GRID_PX = grid_step_mm / scale_mm_px

@st.cache_data(show_spinner=False)
def make_grid_png(width: int, height: int, step_px: float) -> str:
    """Генерирует PNG-сетку как base64."""
    step_px = max(5, step_px)  # Минимальный шаг для видимости
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    for x in range(0, width, int(step_px)):
        draw.line([(x, 0), (x, height)], fill=(227, 227, 227, 255))
    for y in range(0, height, int(step_px)):
        draw.line([(0, y), (width, y)], fill=(227, 227, 227, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

bg_png_b64 = make_grid_png(CANVAS_WIDTH, CANVAS_HEIGHT, GRID_PX)

# Сериализация полигонов
def poly_to_wkb(p: Polygon) -> bytes:
    return p.wkb

def poly_from_wkb(b: bytes) -> Polygon:
    return shapely.wkb.loads(b)

# Извлечение полигонов из FabricJS
def _extract_user_polygons(json_data: dict) -> List[Polygon]:
    polys: List[Polygon] = []
    if not json_data:
        return polys
    for obj in json_data.get("objects", []):
        pts: Optional[List[Tuple[float, float]]] = None
        if obj.get("type") == "path":
            pts = [(cmd[1], cmd[2]) for cmd in obj["path"] if cmd[0] in ("M", "L")]
            if pts and pts[0] != pts[-1]:
                pts.append(pts[0])
        elif obj.get("type") == "polygon":
            pts = [(p[0], p[1]) for p in obj["points"]]
        if pts and len(pts) >= 3:
            if grid_snap:
                pts = [
                    (round(x / GRID_PX) * GRID_PX, round(y / GRID_PX) * GRID_PX)
                    for x, y in pts
                ]
            try:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)  # Попытка исправить самопересечения
                    if not poly.is_valid:
                        continue
                polys.append(poly)
            except ValueError:
                continue
    return polys

# Рисование внешнего контура
st.subheader("1️⃣ Нарисуйте внешний контур этажа")
st.markdown("Нарисуйте **один** замкнутый полигон. Используйте кнопку 'Сохранить контур' для продолжения.")

contour_json = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=2,
    stroke_color="#000000",
    background_image=f"data:image/png;base64,{bg_png_b64}",
    height=CANVAS_HEIGHT,
    width=CANVAS_WIDTH,
    drawing_mode="polygon",
    key="contour_canvas",
)

if "contour_poly_wkb" not in st.session_state:
    st.session_state["contour_poly_wkb"] = None

contour_polys = _extract_user_polygons(contour_json.json_data)
if contour_polys and not all(p.is_valid for p in contour_polys):
    st.warning("Обнаружен некорректный полигон (например, самопересечение). Исправьте контур.")

save_contour = st.button(
    "📌 Сохранить контур",
    disabled=not contour_polys or not all(p.is_valid for p in contour_polys),
)
if st.button("🗑 Очистить контур"):
    st.session_state["contour_poly_wkb"] = None
    st.experimental_rerun()

if save_contour:
    if len(contour_polys) > 1:
        st.warning("Найдены несколько полигонов. Используется первый.")
    st.session_state["contour_poly_wkb"] = poly_to_wkb(contour_polys[0])
    st.experimental_rerun()

# Рисование зон МОП
st.subheader("2️⃣ Нарисуйте зоны МОП (необязательно)")
st.markdown("Нарисуйте замкнутые полигоны для зон МОП. Добавляйте по одной зоне кнопкой 'Добавить МОП'.")

holes_json = st_canvas(
    fill_color="rgba(255,0,0,0.3)",
    stroke_width=2,
    stroke_color="#ff0000",
    background_image=f"data:image/png;base64,{bg_png_b64}",
    height=CANVAS_HEIGHT,
    width=CANVAS_WIDTH,
    drawing_mode="polygon",
    key="holes_canvas",
)

if "holes_polys_wkb" not in st.session_state:
    st.session_state["holes_polys_wkb"] = []

holes_polys = _extract_user_polygons(holes_json.json_data)
if holes_polys and not all(p.is_valid for p in holes_polys):
    st.warning("Обнаружена некорректная зона МОП. Исправьте полигон.")

add_hole = st.button(
    "➕ Добавить МОП",
    disabled=not holes_polys or not all(p.is_valid for p in holes_polys),
)
if add_hole:
    st.session_state["holes_polys_wkb"].extend(poly_to_wkb(h) for h in holes_polys)
    st.experimental_rerun()

if st.session_state["holes_polys_wkb"]:
    if st.button("🗑 Очистить МОП"):
        st.session_state["holes_polys_wkb"].clear()
        st.experimental_rerun()

# Валидация полигонов
if st.session_state["contour_poly_wkb"] is None:
    st.info("Нарисуйте и сохраните внешний контур, затем можно добавить зоны МОП.")
    st.stop()

outer: Polygon = poly_from_wkb(st.session_state["contour_poly_wkb"])
if not outer.is_valid:
    st.error("Внешний контур некорректен. Перерисуйте контур.")
    st.stop()

floor_poly: Polygon | MultiPolygon = outer
for h_wkb in st.session_state["holes_polys_wkb"]:
    h = poly_from_wkb(h_wkb)
    if h.is_valid:
        floor_poly = floor_poly.difference(h)

if isinstance(floor_poly, MultiPolygon):
    st.warning("После вычитания МОП получено несколько зон. Используется самая большая.")
    floor_poly = max(floor_poly.geoms, key=lambda p: p.area)

if floor_poly.is_empty:
    st.error("После вычитания МОП не осталось площади этажа!")
    st.stop()

area_m2 = floor_poly.area * (scale_mm_px ** 2) / 1e6
if area_m2 < MIN_FLOOR_AREA_M2:
    st.error(f"Площадь этажа ({area_m2:.2f} м²) слишком мала. Уменьшите зоны МОП или увеличьте контур.")
    st.stop()

# Метрики этажа
minx, miny, maxx, maxy = floor_poly.bounds
width_mm = (maxx - minx) * scale_mm_px
height_mm = (maxy - miny) * scale_mm_px
st.success(
    f"Контур: **{width_mm:.0f} × {height_mm:.0f} мм**, площадь **{area_m2:.2f} м²**"
)

# Функция разделения полигона
@st.cache_data(show_spinner=False)
def split_poly(poly_wkb: bytes, target_px2: float, tol: float = 0.05) -> Tuple[bytes, Optional[bytes]]:
    poly = poly_from_wkb(poly_wkb)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)

    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    sides: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        length = math.hypot(x2 - x1, y2 - y1)
        sides.append(((x1, y1), (x2, y2), length))
    sides.sort(key=lambda s: s[2], reverse=True)

    max_iter = max(20, int(math.log2(max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])) + 10))
    for (p1, p2, len_side) in sides[:2]:
        ux, uy = (p2[0] - p1[0]) / len_side, (p2[1] - p1[1]) / len_side
        vx, vy = -uy, ux
        projs = [ux * x + uy * y for x, y in poly.exterior.coords]
        low, high = min(projs), max(projs)
        precision = min(1e-3, (high - low) / 1000)

        def make_cut(offset: float) -> LineString:
            mx, my = ux * offset, uy * offset
            minx_, miny_, maxx_, maxy_ = poly.bounds
            diag = math.hypot(maxx_ - minx_, maxy_ - miny_) * 2
            return LineString(
                [(mx + vx * diag, my + vy * diag), (mx - vx * diag, my - vy * diag)]
            )

        for _ in range(max_iter):
            mid = (low + high) / 2
            parts = split(poly, make_cut(mid))
            if not parts.geoms or len(parts.geoms) < 2:
                low = mid
                if abs(high - low) < precision:
                    break
                continue
            parts = list(parts.geoms)[:2]
            parts.sort(key=lambda p_: p_.area)
            smaller = parts[0]
            a = smaller.area
            minx_s, miny_s, maxx_s, maxy_s = smaller.bounds
            w_mm = (maxx_s - minx_s) * scale_mm_px
            h_mm = (maxy_s - miny_s) * scale_mm_px
            if w_mm < MIN_APT_DIM_MM or h_mm < MIN_APT_DIM_MM:
                high = mid
                continue
            if a > target_px2 * (1 + tol):
                high = mid
            elif a < target_px2 * (1 - tol):
                low = mid
            else:
                return poly_to_wkb(smaller), poly_to_wkb(parts[1])
            if abs(high - low) < precision:
                break

    st.warning("Не удалось разделить полигон идеально. Используется резервный разрез.")
    minx_, miny_, maxx_, maxy_ = poly.bounds
    parts = list(split(poly, LineString([(minx_, miny_), (maxx_, maxy_)])))
    parts.sort(key=lambda p_: p_.area)
    if len(parts) == 1:
        return poly_to_wkb(parts[0]), None
    return poly_to_wkb(parts[0]), poly_to_wkb(parts[1])

# Генерация квартирографии
st.subheader("3️⃣ Сгенерировать квартирографию")
if st.button("🚀 Запустить генерацию", disabled=not any(percentages.values())):
    if not any(percentages.values()):
        st.error("Задайте ненулевой процент хотя бы для одного типа квартир.")
        st.stop()

    with st.spinner("Расчет количества квартир..."):
        avg_area = {t: sum(AREA_RANGES[t]) / 2 for t in APT_TYPES}
        total_build_area = area_m2 * floors
        counts_target = {
            t: round(total_build_area * percentages[t] / 100 / avg_area[t])
            for t in APT_TYPES
        }
        per_floor: Dict[int, Dict[str, int]] = {f: {t: 0 for t in APT_TYPES} for f in range(floors)}
        for t, cnt in counts_target.items():
            if cnt == 0:
                continue
            q, r = divmod(cnt, floors)
            for f in range(floors):
                per_floor[f][t] = q + (1 if f < r else 0)

    prog = st.progress(0, text="Нарезка этажей...")
    floor_placements: Dict[int, List[Tuple[str, bytes]]] = {}
    missing: Dict[str, int] = {t: 0 for t in APT_TYPES}
    floor_poly_wkb = poly_to_wkb(floor_poly)

    for fl in range(floors):
        targets: List[Tuple[str, float]] = []
        for t, n in per_floor[fl].items():
            for _ in range(n):
                m2 = random.uniform(*AREA_RANGES[t])
                px2 = m2 * 1e6 / (scale_mm_px ** 2)
                targets.append((t, px2))
        available: List[bytes] = [floor_poly_wkb]
        placed: List[Tuple[str, bytes]] = []
        for t, px2 in targets:
            available.sort(key=lambda p: poly_from_wkb(p).area, reverse=True)
            if not available:
                missing[t] += 1
                continue
            largest = available.pop(0)
            apt_wkb, rem_wkb = split_poly(largest, px2)
            placed.append((t, apt_wkb))
            if rem_wkb and poly_from_wkb(rem_wkb).area > 0.02 * px2:
                available.append(rem_wkb)
        floor_placements[fl + 1] = placed
        prog.progress((fl + 1) / floors, text=f"Готово {fl + 1}/{floors} этажей")

    prog.empty()

    # Визуализация с Plotly
    st.subheader("4️⃣ Планы этажей")
    for fl, placement in floor_placements.items():
        st.markdown(f"### Этаж {fl}")
        fig = go.Figure()
        for t, poly_wkb in placement:
            poly = poly_from_wkb(poly_wkb)
            x, y = poly.exterior.xy
            area_m2_apt = poly.area * (scale_mm_px ** 2) / 1e6
            minx_a, miny_a, maxx_a, maxy_a = poly.bounds
            w_mm = (maxx_a - minx_a) * scale_mm_px
            h_mm = (maxy_a - miny_a) * scale_mm_px
            fig.add_trace(go.Scatter(
                x=[xi * scale_mm_px for xi in x],
                y=[yi * scale_mm_px for yi in y],
                fill="toself",
                fillcolor=COLORS[t],
                line_color="black",
                opacity=0.7,
                text=f"{t}<br>{w_mm:.0f}×{h_mm:.0f} мм<br>{area_m2_apt:.2f} м²",
                hoverinfo="text"
            ))
        # Масштабная линейка 5 м
        x0 = minx * scale_mm_px + 20
        fig.add_trace(go.Scatter(
            x=[x0, x0 + 5000],
            y=[20, 20],
            mode="lines+text",
            line=dict(color="black", width=4),
            text=["", "5 м"],
            textposition="top center",
            showlegend=False
        ))
        # Легенда
        for t, c in COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color=c, size=10),
                name=t,
                showlegend=True
            ))
        fig.update_layout(
            showlegend=True,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    # Сводный отчет
    st.subheader("5️⃣ Сводный отчет")
    rows = []
    for fl, placement in floor_placements.items():
        for t in APT_TYPES:
            qty = sum(1 for tp, _ in placement if tp == t)
            rows.append({"Этаж": fl, "Тип": t, "Количество": qty})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if any(missing.values()):
        miss_txt = ", ".join(f"{t}: {n}" for t, n in missing.items() if n)
        st.warning(f"⚠️ Не удалось разместить: {miss_txt}")

    st.download_button(
        "⬇️ Скачать отчет CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="report.csv",
        mime="text/csv",
    )
