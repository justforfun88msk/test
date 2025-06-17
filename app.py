# -*- coding: utf-8 -*-
"""
Квартирография — интерактивный генератор квартирных планов (Architect Edition)
=============================================================================
Исправленная версия 2025-06-17.

Основные изменения
------------------
1. **Фон-сетка** рендерится в PNG и передаётся как background_image,
   поэтому не попадает в json_data, не участвует в Undo/Clear и не тормозит.
2. Кнопки «Сохранить контур» и «Добавить МОП» становятся активными **только**
   если найдены валидные пользовательские полигоны.
3. Добавлены кнопки «🗑 Очистить контур» и «🗑 Очистить МОП».
4. Возможность удалять отдельные зоны МОП.
5. Если сумма процентов > 100 % — генерация блокируется и подсвечивается.
6. split_poly не полагается на внешние minx/miny, имеет защиту от зацикливания
   и корректный fallback.
7. Площади квартир выбираются случайно в заданном диапазоне, чтобы была вариативность.
8. Отчёт дополняется сводкой о неразмещённых квартирах.
9. Экспорт в JSON теперь сохраняет контур и отдельные МОП правильно через geojson.
10. Функции с тяжёлыми расчётами помечены st.cache_data для ускорения.

🔧 Требования
-------------
pip install streamlit shapely matplotlib pandas streamlit-drawable-canvas pillow

▶️ Запуск
---------
streamlit run kvartirografia_fixed.py
"""
from __future__ import annotations
import math
import json
import random
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString, MultiPolygon, mapping
from shapely.ops import split
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

# -------------------------
#   КОНФИГУРАЦИЯ СТРАНИЦЫ
# -------------------------
st.set_page_config(
    page_title="Квартирография Architect Edition",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("📐 Квартирография — Architect Edition (fixed)")

# -------------------------
#   НАСТРОЙКИ В САЙДБАРЕ
# -------------------------
floors: int = st.sidebar.number_input("Этажей в доме", min_value=1, value=10)
scale_mm_px: float = st.sidebar.number_input("Миллиметров в 1 пикселе", min_value=0.1, value=10.0, step=0.1)
grid_step_mm: int = st.sidebar.number_input("Шаг сетки, мм", min_value=5, value=100, step=5)
show_snap: bool = st.sidebar.checkbox("Привязка к сетке", value=True)

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
    """Collect user-defined percentages and auto-normalize the last one."""
    vals = {}
    inputs = []
    for t in APT_TYPES[:-1]:
        val = st.sidebar.number_input(
            f"% {t}", 0.0, 100.0, 100.0 / len(APT_TYPES), step=1.0, key=f"pct_{t}")
        inputs.append(val)
    total = sum(inputs)
    if total > 100:
        st.sidebar.error("Сумма первых четырёх типов > 100 %. Уменьшите значения.")
        return {}
    last = 100.0 - total
    st.sidebar.markdown(f"**% {APT_TYPES[-1]}:** {last:.1f} (авто)")
    for t, v in zip(APT_TYPES, inputs + [last]):
        vals[t] = v
    return vals

percentages: Dict[str, float] = apartment_percentages()

st.sidebar.subheader("📏 Диапазоны площадей (м²)")
AREA_RANGES: Dict[str, Tuple[float, float]] = {}
for t in APT_TYPES:
    AREA_RANGES[t] = st.sidebar.slider(t, 10.0, 200.0, (20.0, 50.0), key=f"area_{t}")

st.sidebar.header("💾 Файлы проекта")
project_name: str = st.sidebar.text_input("Имя файла проекта", "plan.json")

# -------------------------
#   ХУДОЖЕСТВЕННАЯ ЧАСТЬ
# -------------------------
st.subheader("1️⃣ Нарисуйте внешний контур этажа")
CANVAS_W, CANVAS_H = 800, 600
GRID_PX = grid_step_mm / scale_mm_px  # пикселей

@st.cache_data(show_spinner=False)
def make_grid_png(w: int, h: int, step_px: float) -> str:
    if step_px < 5:
        img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    else:
        img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        for x in range(0, w, int(step_px)):
            draw.line([(x, 0), (x, h)], fill=(227, 227, 227, 255))
        for y in range(0, h, int(step_px)):
            draw.line([(0, y), (w, y)], fill=(227, 227, 227, 255))
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

bg_png = make_grid_png(CANVAS_W, CANVAS_H, GRID_PX)

def _extract_user_polygons(json_data: dict) -> List[Polygon]:
    polys: List[Polygon] = []
    if not json_data:
        return polys
    for obj in json_data.get("objects", []):
        pts: Optional[List[Tuple[float, float]]] = None
        if obj.get("type") == "path":
            pts = [(cmd[1], cmd[2]) for cmd in obj["path"] if cmd[0] in ("M", "L")]
        elif obj.get("type") == "polygon":
            pts = [(p[0], p[1]) for p in obj["points"]]
        if pts and len(pts) >= 3:
            if show_snap:
                pts = [(round(x / GRID_PX) * GRID_PX, round(y / GRID_PX) * GRID_PX) for x, y in pts]
            polys.append(Polygon(pts))
    return polys

# Контур
contour_json = st_canvas(
    fill_color="rgba(0, 0, 0, 0)", stroke_width=2, stroke_color="#000000",
    background_image=f"data:image/png;base64,{bg_png}", height=CANVAS_H, width=CANVAS_W,
    drawing_mode="polygon", key="contour_canvas"
)

if "contour_poly" not in st.session_state:
    st.session_state.contour_poly: Optional[Polygon] = None

valid_contour = bool(_extract_user_polygons(contour_json.json_data))
save_contour = st.button("📌 Сохранить контур", disabled=not valid_contour)
if save_contour:
    polys = _extract_user_polygons(contour_json.json_data)
    st.session_state.contour_poly = polys[0]
    st.session_state.clear_contour = True
    st.experimental_rerun()

if st.session_state.get("contour_poly"):
    if st.button("🗑 Очистить контур"):
        st.session_state.contour_poly = None
        st.experimental_rerun()

st.caption("Нарисуйте **один** замкнутый полигон и сохраните его")

# МОП
st.subheader("2️⃣ Нарисуйте зоны МОП (необязательно)")
holes_json = st_canvas(
    fill_color="rgba(255,0,0,0.3)", stroke_width=2, stroke_color="#ff0000",
    background_image=f"data:image/png;base64,{bg_png}", height=CANVAS_H, width=CANVAS_W,
    drawing_mode="polygon", key="holes_canvas"
)

if "holes_polys" not in st.session_state:
    st.session_state.holes_polys: List[Polygon] = []

valid_holes = bool(_extract_user_polygons(holes_json.json_data))
add_hole = st.button("➕ Добавить МОП", disabled=not valid_holes)
if add_hole:
    new = _extract_user_polygons(holes_json.json_data)
    st.session_state.holes_polys.extend(new)
    st.experimental_rerun()

if st.session_state.holes_polys:
    if st.button("🗑 Очистить МОП"):
        st.session_state.holes_polys.clear()
        st.experimental_rerun()
    st.write("Текущие зоны МОП:")
    for idx, hole in enumerate(st.session_state.holes_polys):
        st.write(f"- МОП #{idx+1}")
        if st.button(f"Удалить МОП #{idx+1}", key=f"del_hole_{idx}"):
            st.session_state.holes_polys.pop(idx)
            st.experimental_rerun()

# Валидация контура
if st.session_state.contour_poly is None:
    st.info("Нарисуйте и сохраните внешний контур, затем можно добавить МОП.")
    st.stop()

outer = st.session_state.contour_poly
if not outer.is_valid or not outer.is_simple:
    st.error("Внешний контур некорректен (самопересечения и т. п.).")
    st.stop()

# Вычитаем дыры
floor_poly: Polygon | MultiPolygon = outer
for h in st.session_state.holes_polys:
    if h.is_valid:
        floor_poly = floor_poly.difference(h)

if floor_poly.is_empty:
    st.error("После вычитания МОП не осталось площади этажа!")
    st.stop()

# Метрика этажа
minx, miny, maxx, maxy = floor_poly.bounds
width_mm = (maxx - minx) * scale_mm_px
height_mm = (maxy - miny) * scale_mm_px
area_m2 = floor_poly.area * (scale_mm_px ** 2) / 1e6
st.success(f"Контур: **{width_mm:.0f} × {height_mm:.0f} мм**, площадь **{area_m2:.2f} м²**")

# Алгоритм разделения
@st.cache_data(show_spinner=False)
def split_poly(poly: Polygon, target_px2: float, tol: float = 0.05) -> Tuple[Polygon, Optional[Polygon]]:
    # защита от нулевой цели
    if target_px2 <= 0:
        return poly, None
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    sides = []
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]; x2, y2 = coords[i+1]
        length = math.hypot(x2 - x1, y2 - y1)
        sides.append(((x1, y1), (x2, y2), length))
    sides = sorted(sides, key=lambda s: s[2], reverse=True)
    for p1, p2, length in sides[:2]:
        ux, uy = ((p2[0]-p1[0])/length, (p2[1]-p1[1])/length)
        vx, vy = -uy, ux
        projs = [ux*x+uy*y for x,y in poly.exterior.coords]
        low, high = min(projs), max(projs)
        def make_cut(off: float) -> LineString:
            mx, my = ux*off, uy*off
            diag = math.hypot(*(poly.bounds[2:]-poly.bounds[:2]))*2
            return LineString([(mx+vx*diag, my+vy*diag), (mx-vx*diag, my-vy*diag)])
        for _ in range(40):
            mid = (low+high)/2
            parts = split(poly, make_cut(mid))
            if len(parts.geoms) < 2:
                low = mid
                if abs(high-low)<1e-3: break
                continue
            ps = sorted(list(parts.geoms)[:2], key=lambda p: p.area)
            a = ps[0].area
            if a>target_px2*(1+tol): high=mid
            elif a<target_px2*(1-tol): low=mid
            else: return ps[0], ps[1]
            if abs(high-low)<1e-3: break
    # fallback
    minx_, miny_, maxx_, maxy_ = poly.bounds
    parts = split(poly, LineString([(minx_,miny_),(maxx_,maxy_)]))
    geoms = list(parts.geoms)
    if len(geoms)<2:
        return poly, None
    ps = sorted(geoms[:2], key=lambda p: p.area)
    return ps[0], ps[1]

# Генерация квартирографии
st.subheader("3️⃣ Сгенерировать квартирографию")
launch_disabled = not percentages
launch_help = "Сумма процентов квартир превышает 100 %" if launch_disabled else None
if st.button("🚀 Запустить генерацию", disabled=launch_disabled, help=launch_help):
    with st.spinner("Расчёт количества квартир…"):
        avg_area = {t: sum(AREA_RANGES[t])/2 for t in APT_TYPES}
        total_area = area_m2 * floors
        counts_target = {t: max(1, round(total_area*percentages[t]/100/avg_area[t])) for t in APT_TYPES}
        per_floor = {f: {t:0 for t in APT_TYPES} for f in range(floors)}
        for t,c in counts_target.items():
            q,r = divmod(c, floors)
            for f in range(floors): per_floor[f][t] = q + (1 if f<r else 0)
    prog = st.progress(0, text="Нарезка этажей…")
    placements = {}
    missing = {t:0 for t in APT_TYPES}
    for fi in range(floors):
        targets=[]
        for t,n in per_floor[fi].items():
            for _ in range(n):
                m2 = random.uniform(*AREA_RANGES[t])
                px2 = m2*1e6/(scale_mm_px**2)
                targets.append((t,px2))
        avail=[floor_poly]
        placed=[]
        for t,px2 in targets:
            avail.sort(key=lambda p: p.area, reverse=True)
            if not avail: missing[t]+=1; continue
            largest = avail.pop(0)
            apt, rem = split_poly(largest, px2)
            placed.append((t,apt))
            if rem and rem.area>0.02*px2: avail.append(rem)
        placements[fi+1]=placed
        prog.progress((fi+1)/floors, text=f"Готово {fi+1}/{floors} этажей")
    prog.empty()

    # Визуализация
    st.subheader("4️⃣ Планы этажей")
    for fl, pl in placements.items():
        st.markdown(f"### Этаж {fl}")
        fig, ax = plt.subplots(figsize=(6,5))
        for t, poly in pl:
            x,y = poly.exterior.xy
            ax.fill([xi*scale_mm_px for xi in x],[yi*scale_mm_px for yi in y],
                    color=COLORS[t], alpha=0.7, edgecolor="black", linewidth=1)
            cx, cy = poly.representative_point().xy
            area_m2_apt = poly.area*(scale_mm_px**2)/1e6
            ax.text(cx[0]*scale_mm_px, cy[0]*scale_mm_px,
                    f"{t}\n{area_m2_apt:.1f} м²", ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        for t,c in COLORS.items(): ax.scatter([],[], color=c, label=t)
        # Шкала 5 м (мм)
        ax.plot([20, 20+5000], [20,20], lw=4)
        ax.text(20+2500, 40, "5 м", ha="center", va="bottom")
        ax.set_aspect("equal")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # Отчет
    st.subheader("5️⃣ Сводный отчёт")
    rows=[]
    for fl, pl in placements.items():
        for t in APT_TYPES:
            rows.append({"Этаж": fl, "Тип": t, "Количество": sum(1 for tp,_ in pl if tp==t)})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    if any(missing.values()):
        miss = ", ".join(f"{t}: {n}" for t,n in missing.items() if n)
        st.warning(f"⚠️ Не удалось разместить: {miss}")

    # Скачать
    st.sidebar.download_button(
        "⬇️ Скачать отчёт CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="report.csv", mime="text/csv"
    )
    project_data = {
        "scale_mm_px": scale_mm_px,
        "grid_step_mm": grid_step_mm,
        "floors": floors,
        "percentages": percentages,
        "area_ranges": AREA_RANGES,
        "contour": mapping(st.session_state.contour_poly),
        "holes": [mapping(h) for h in st.session_state.holes_polys]
    }
    st.sidebar.download_button(
        "⬇️ Скачать проект JSON",
        json.dumps(project_data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=project_name, mime="application/json"
    )
