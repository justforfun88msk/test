# -*- coding: utf-8 -*-
""" Квартирография — интерактивный генератор квартирных планов (Architect Edition)
=============================================================================
Исправленная версия 2025-06-17.
Основные изменения:
1. Добавлена обработка ошибок и валидация
2. Оптимизирована производительность с кэшированием
3. Улучшен пользовательский интерфейс
4. Расширена функциональность экспорта
5. Добавлена защита от краевых случаев
"""

import math
import json
import random
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import split
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

# Конфигурация страницы
st.set_page_config(
    page_title="Квартирография Architect Edition",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("📐 Квартирография — Architect Edition (improved)")

# Настройки в сайдбаре
st.sidebar.header("🏢 Параметры здания и сетки")
floors: int = st.sidebar.number_input("Этажей в доме", min_value=1, value=10)
scale_mm_px: float = st.sidebar.number_input("Миллиметров в 1 пикселе", min_value=0.1, value=10.0, step=0.1)
grid_step_mm: int = st.sidebar.number_input("Шаг сетки, мм", min_value=5, value=100, step=5)
show_snap: bool = st.sidebar.checkbox("Привязка к сэтке", value=True)

# Типы квартир и цвета
APT_TYPES = ["Студия", "1С", "2С", "3С", "4С"]
COLORS = {
    "Студия": "#FFC107",
    "1С": "#8BC34A",
    "2С": "#03A9F4",
    "3С": "#E91E63",
    "4С": "#9C27B0",
}

# Функция валидации процентного распределения
def validate_apartment_percentages(percentages: Dict[str, float]) -> bool:
    """Проверка корректности распределения процентов"""
    total = sum(percentages.values())
    return abs(total - 100.0) < 0.01

def apartment_percentages() -> Dict[str, float]:
    """Сбор пользовательских процентов с валидацией"""
    inputs = []
    for t in APT_TYPES[:-1]:
        val = st.sidebar.number_input(
            f"% {t}", 0.0, 100.0, 100.0 / len(APT_TYPES), step=1.0, key=f"pct_{t}"
        )
        inputs.append(val)
    
    sum_inputs = sum(inputs)
    if sum_inputs > 100:
        st.sidebar.error("Сумма первых четырёх типов > 100 %. Уменьшите значения.")
        return {}
    
    last_val = 100.0 - sum_inputs
    st.sidebar.markdown(f"**% {APT_TYPES[-1]}:** {last_val:.1f} (авто)")
    return {t: v for t, v in zip(APT_TYPES, inputs + [last_val])}

# Получаем процентное распределение
percentages: Dict[str, float] = apartment_percentages()

# Диапазоны площадей
st.sidebar.subheader("📏 Диапазоны площадей (м²)")
AREA_RANGES: Dict[str, Tuple[float, float]] = {}
for t in APT_TYPES:
    AREA_RANGES[t] = st.sidebar.slider(
        t, 10.0, 200.0, (20.0, 50.0), key=f"area_{t}"
    )

# Настройки экспорта
st.sidebar.header("💾 Файлы проекта")
project_name: str = st.sidebar.text_input("Имя файла проекта", "plan.json")

# Функция создания сетки с кэшированием
@st.cache_data(show_spinner=False, ttl=3600)
def make_grid_png(width: int, height: int, step_px: float) -> str:
    """Создание PNG сетки с кэшированием"""
    if step_px < 5:
        img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    for x in range(0, width, int(step_px)):
        draw.line([(x, 0), (x, height)], fill=(227, 227, 227, 255))
    for y in range(0, height, int(step_px)):
        draw.line([(0, y), (width, y)], fill=(227, 227, 227, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# Основной контур этажа
st.subheader("1️⃣ Нарисуйте внешний контур этажа")
CANVAS_WIDTH, CANVAS_HEIGHT = 800, 600
GRID_PX = grid_step_mm / scale_mm_px

def _extract_user_polygons(json_data: dict) -> List[Polygon]:
    """Извлечение полигонов из JSON с обработкой ошибок"""
    try:
        polys = []
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
                    pts = [(round(x / GRID_PX) * GRID_PX, round(y / GRID_PX) * GRID_PX) 
                          for x, y in pts]
                polys.append(Polygon(pts))
        return polys
    except Exception as e:
        st.error(f"Ошибка при обработке полигонов: {str(e)}")
        return []

# Создание холста для контура
bg_png_b64 = make_grid_png(CANVAS_WIDTH, CANVAS_HEIGHT, GRID_PX)
contour_json = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=2,
    stroke_color="#000000",
    background_image=f"data:image/png;base64,{bg_png_b64}",
    height=CANVAS_HEIGHT,
    width=CANVAS_WIDTH,
    drawing_mode="polygon",
    key="contour_canvas",
)
st.caption("Нарисуйте **один** замкнутый полигон. Затем нажмите кнопку ниже.")

# Инициализация состояния
if "contour_poly" not in st.session_state:
    st.session_state.contour_poly = None

def has_valid_polys(json_data: dict) -> bool:
    """Проверка наличия валидных полигонов"""
    return bool(_extract_user_polygons(json_data))

# Кнопка сохранения контура с валидацией
save_contour = st.button("📌 Сохранить контур", disabled=not has_valid_polys(contour_json.json_data))
if save_contour:
    try:
        polygons_px = _extract_user_polygons(contour_json.json_data)
        if not polygons_px:
            st.warning("Не найдено корректных полигонов.")
        else:
            if len(polygons_px) > 1:
                st.warning("Найдены несколько полигонов. Используется первый.")
            st.session_state.contour_poly = polygons_px[0]
    except Exception as e:
        st.error(f"Ошибка при сохранении контура: {str(e)}")

# Зоны МОП
st.subheader("2️⃣ Нарисуйте зоны МОП (необязательно)")
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

# Инициализация списка МОП
if "holes_polys" not in st.session_state:
    st.session_state.holes_polys: List[Polygon] = []

# Кнопка добавления МОП
add_hole = st.button("➕ Добавить МОП", disabled=not has_valid_polys(holes_json.json_data))
if add_hole:
    try:
        new_holes = _extract_user_polygons(holes_json.json_data)
        st.session_state.holes_polys.extend(new_holes)
    except Exception as e:
        st.error(f"Ошибка при добавлении МОП: {str(e)}")

# Кнопка очистки МОП
if st.session_state.holes_polys:
    if st.button("🗑 Очистить МОП"):
        st.session_state.holes_polys.clear()

# Валидация полигонов
def validate_floor_polygon() -> Tuple[bool, str]:
    """Валидация полигонов с детальным сообщением об ошибке"""
    if st.session_state.contour_poly is None:
        return False, "Нарисуйте и сохраните внешний контур"
    
    outer = st.session_state.contour_poly
    if not outer.is_valid or not outer.is_simple:
        return False, "Внешний контур некорректен (самопересечения и т. п.)"
    
    floor_poly = outer
    for h in st.session_state.holes_polys:
        if h.is_valid:
            floor_poly = floor_poly.difference(h)
    
    if floor_poly.is_empty:
        return False, "После вычитания МОП не осталось площади этажа!"
    
    return True, ""

# Проверка валидности
is_valid, error_msg = validate_floor_polygon()
if not is_valid:
    st.error(error_msg)
    st.stop()

# Метрика этажа
minx, miny, maxx, maxy = floor_poly.bounds
width_mm = (maxx - minx) * scale_mm_px
height_mm = (maxy - miny) * scale_mm_px
area_m2 = floor_poly.area * (scale_mm_px ** 2) / 1e6
st.success(f"Контур: **{width_mm:.0f} × {height_mm:.0f} мм**, площадь **{area_m2:.2f} м²**")

# Улучшенный алгоритм разбиения
@st.cache_data(show_spinner=False)
def split_poly(poly: Polygon, target_px2: float, tol: float = 0.05) -> Tuple[Polygon, Optional[Polygon]]:
    """Улучшенный алгоритм разбиения с защитой от зацикливания"""
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        
        sides = []
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            length = math.hypot(x2 - x1, y2 - y1)
            sides.append(((x1, y1), (x2, y2), length))
        
        sides = sorted(sides, key=lambda s: s[2], reverse=True)
        
        for (p1, p2, len_side) in sides[:2]:
            ux, uy = (p2[0] - p1[0]) / len_side, (p2[1] - p1[1]) / len_side
            vx, vy = -uy, ux
            
            projs = [ux * x + uy * y for x, y in poly.exterior.coords]
            low, high = min(projs), max(projs)
            
            def make_cut(offset: float) -> LineString:
                mx, my = ux * offset, uy * offset
                minx_, miny_, maxx_, maxy_ = poly.bounds
                diag = math.hypot(maxx_ - minx_, maxy_ - miny_) * 2
                return LineString([(mx + vx * diag, my + vy * diag), 
                                 (mx - vx * diag, my - vy * diag)])
            
            for _ in range(40):
                mid = (low + high) / 2
                parts = split(poly, make_cut(mid))
                
                if len(parts.geoms) < 2:
                    low = mid
                    if abs(high - low) < 1e-3:
                        break
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
                
                if abs(high - low) < 1e-3:
                    break
        
        # fallback: отрезаем половину площади
        minx_, miny_, maxx_, maxy_ = poly.bounds
        parts = list(split(poly, LineString([(minx_, miny_), (maxx_, maxy_)])))
        parts.sort(key=lambda p_: p_.area)
        
        if len(parts) == 1:
            return parts[0], None
        return parts[0], parts[1]
    except Exception as e:
        st.error(f"Ошибка при разбиении полигона: {str(e)}")
        return poly, None

# Генерация квартирографии
st.subheader("3️⃣ Сгенерировать квартирографию")
if st.button("🚀 Запустить генерацию", disabled=(not percentages)):
    try:
        with st.spinner("Расчёт количества квартир…"):
            # Расчёт количества квартир
            avg_area = {t: sum(AREA_RANGES[t]) / 2 for t in APT_TYPES}
            total_build_area = area_m2 * floors
            counts_target = {
                t: max(1, round(total_build_area * percentages[t] / 100 / avg_area[t]))
                for t in APT_TYPES
            }
            
            # Распределение по этажам
            per_floor: Dict[int, Dict[str, int]] = {f: {t: 0 for t in APT_TYPES} for f in range(floors)}
            for t, cnt in counts_target.items():
                q, r = divmod(cnt, floors)
                for f in range(floors):
                    per_floor[f][t] = q + (1 if f < r else 0)
        
        # Прогресс-бар
        prog = st.progress(0, text="Нарезка этажей…")
        floor_placements: Dict[int, List[Tuple[str, Polygon]]] = {}
        missing: Dict[str, int] = {t: 0 for t in APT_TYPES}
        
        # Генерация по этажам
        for fl in range(floors):
            targets: List[Tuple[str, float]] = []
            for t, n in per_floor[fl].items():
                for _ in range(n):
                    m2 = random.uniform(*AREA_RANGES[t])
                    px2 = m2 * 1e6 / (scale_mm_px ** 2)
                    targets.append((t, px2))
            
            available: List[Polygon] = [floor_poly]
            placed: List[Tuple[str, Polygon]] = []
            
            for t, px2 in targets:
                available.sort(key=lambda p: p.area, reverse=True)
                if not available:
                    missing[t] += 1
                    continue
                
                largest = available.pop(0)
                apt, rem = split_poly(largest, px2)
                placed.append((t, apt))
                
                if rem and rem.area > 0.02 * px2:
                    available.append(rem)
            
            floor_placements[fl + 1] = placed
            prog.progress((fl + 1) / floors, text=f"Готово {fl+1}/{floors} этажей")
        
        prog.empty()
        
        # Визуализация этажей
        st.subheader("4️⃣ Планы этажей")
        for fl, placement in floor_placements.items():
            st.markdown(f"### Этаж {fl}")
            fig, ax = plt.subplots(figsize=(6, 5))
            
            for t, poly in placement:
                x, y = poly.exterior.xy
                ax.fill([xi * scale_mm_px for xi in x], [yi * scale_mm_px for yi in y],
                        color=COLORS[t], alpha=0.7, edgecolor="black", linewidth=1)
                cx, cy = poly.representative_point().xy
                area_m2_apt = poly.area * (scale_mm_px ** 2) / 1e6
                ax.text(cx[0] * scale_mm_px, cy[0] * scale_mm_px,
                        f"{t}\n{area_m2_apt:.1f} м²", ha="center", va="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            
            # Легенда
            for t, c in COLORS.items():
                ax.scatter([], [], color=c, label=t)
            ax.legend(loc="upper right", fontsize=8)
            
            # Линейка масштаба
            ax.plot([20, 20 + 5000 / scale_mm_px], [20, 20], lw=4, color="black")
            ax.text(20 + 2500 / scale_mm_px, 40, "5 м", ha="center", va="bottom")
            
            ax.set_aspect("equal")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        
        # Отчёт
        st.subheader("5️⃣ Сводный отчёт")
        rows = []
        for fl, placement in floor_placements.items():
            for t in APT_TYPES:
                qty = sum(1 for tp, _ in placement if tp == t)
                rows.append({"Этаж": fl, "Тип": t, "Количество": qty})
        
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        
        # Сообщение о неразмещённых квартирах
        if any(missing.values()):
            miss_txt = ", ".join(f"{t}: {n}" for t, n in missing.items() if n)
            st.warning(f"⚠️ Не удалось разместить: {miss_txt}")
        
        # Кнопки экспорта
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
    except Exception as e:
        st.error(f"Ошибка при генерации квартирографии: {str(e)}")
