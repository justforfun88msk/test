import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import matplotlib.pyplot as plt
import math

# --- Настройки страницы ---
st.set_page_config(layout="wide")
st.title("Квартирография Streamlit")

# --- Sidebar параметры здания и рисования ---
st.sidebar.header("Параметры здания")
floors = st.sidebar.number_input("Этажей в доме", min_value=1, value=5)
scale = st.sidebar.number_input("мм на пиксель", min_value=0.1, value=1.0, step=0.1)

st.sidebar.header("Рисование плана")
object_type = st.sidebar.radio("Тип объекта", ("Контур", "МОП"))
ext_color = st.sidebar.color_picker("Цвет заливки контура", "#FFE0B2")
int_color = st.sidebar.color_picker("Цвет заливки МОП", "#FFCC80")
strict_angles = st.sidebar.checkbox("Строгие углы (90°)", value=False)
mode = st.sidebar.selectbox("Инструмент", ("polygon", "transform"))
show_grid = st.sidebar.checkbox("Показывать сетку", value=True)
grid_spacing = st.sidebar.number_input("Шаг сетки (px)", min_value=5, value=20, step=1)
grid_color = st.sidebar.color_picker("Цвет сетки", "#CCCCCC")

st.sidebar.header("Доли типов квартир (%)")
types = ["studio", "1BR", "2BR", "3BR", "4BR"]
percentages = {
    t: st.sidebar.slider(f"% {t}", 0, 100, 0, key=f"pct_{t}")
    for t in types
}

st.sidebar.header("Диапазоны площадей (м²)")
areas = {}
for t in types:
    min_a = st.sidebar.number_input(f"Мин {t}", min_value=1.0, value=20.0, step=1.0, key=f"min_{t}")
    max_a = st.sidebar.number_input(f"Макс {t}", min_value=1.0, value=50.0, step=1.0, key=f"max_{t}")
    areas[t] = (min_a, max_a)

# --- Canvas для рисования ---
st.write("### Нарисуйте план этажа")
canvas_result = st_canvas(
    fill_color=ext_color if object_type == "Контур" else int_color,
    stroke_width=2,
    stroke_color="black",
    background_color="#FFFFFF",
    drawing_mode=mode,
    display_grid=show_grid,
    grid_color=grid_color,
    grid_spacing=grid_spacing,
    width=800,
    height=600,
    key="canvas",
)

# --- Функции обработки ---
def adjust_strict(coords):
    """Принудительная корректировка углов к 90°"""
    new = [coords[0]]
    for x, y in coords[1:]:
        px, py = new[-1]
        dx, dy = x - px, y - py
        if abs(dx) >= abs(dy):
            new.append((x, py))
        else:
            new.append((px, y))
    return new


def get_shapely_polygons(data):
    """Парсинг JSON из Canvas в Shapely-полигоны с учётом строгих углов"""
    objs = data.get("objects", [])
    polys = []
    for o in objs:
        if o.get("type") == "polygon":
            pts = [(p["x"], p["y"]) for p in o.get("points", [])]
            if strict_angles:
                pts = adjust_strict(pts)
            if len(pts) >= 3:
                polys.append(Polygon(pts))
    return polys


def split_polygon_at_area(polygon, target_area, tol=1e-3):
    """Разбиение полигона на часть площадью target_area и остаток"""
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    # Найти главную ось
    max_len = 0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        d = math.hypot(x2 - x1, y2 - y1)
        if d > max_len:
            max_len = d
            major = ((x1, y1), (x2, y2))
    (x1, y1), (x2, y2) = major
    ux, uy = (x2 - x1) / max_len, (y2 - y1) / max_len
    # Проекции вершин
    # ... (как в первоначальной версии)
    projs = [ux * x + uy * y for x, y in polygon.exterior.coords]
    low, high = min(projs), max(projs)

    def make_cut(offset):
        mx, my = ux * offset, uy * offset
        vx, vy = -uy, ux
        minx, miny, maxx, maxy = polygon.bounds
        diag = math.hypot(maxx - minx, maxy - miny) * 2
        p1 = (mx + vx * diag, my + vy * diag)
        p2 = (mx - vx * diag, my - vy * diag)
        return LineString([p1, p2])

    for _ in range(30):
        mid = (low + high) / 2
        line = make_cut(mid)
        parts = split(polygon, line)
        if len(parts) < 2:
            low = mid
            continue
        # Выбрать нужную часть
        areas_list = []
        for part in parts:
            cx, cy = part.representative_point().coords[0]
            proj = ux * cx + uy * cy
            areas_list.append((proj, part))
        smaller = min(areas_list, key=lambda x: x[0])[1]
        a = smaller.area
        if a > target_area * (1 + tol):
            high = mid
        elif a < target_area * (1 - tol):
            low = mid
        else:
            remainder = [p for p in parts if not p.equals(smaller)][0]
            return smaller, remainder
    # Финальный разрез
    parts = split(polygon, make_cut((low + high) / 2))
    sorted_parts = sorted(parts, key=lambda p: p.area)
    return sorted_parts[0], sorted_parts[1]


def layout_floor(floor_poly):
    """
    Разметка этажа: рассчитывает, сколько квартир каждого типа нужно
    и режет полигон на участки под целевые площади.
    """
    # Общая площадь в м²
    total_m2 = floor_poly.area * scale**2 / 1e6
    # Список целевых площадей
    desired = []
    for t, pct in percentages.items():
        if pct <= 0:
            continue
        area_m2 = total_m2 * pct / 100
        mn, mx = areas[t]
        avg = (mn + mx) / 2
        count = max(1, int(round(area_m2 / avg)))
        for _ in range(count):
            desired.append((t, area_m2 / count))
    desired.sort(key=lambda x: x[1], reverse=True)

    available = [floor_poly]
    apartments = []
    for t, area_m2 in desired:
        # Перевести в px²
        target_px2 = area_m2 * 1e6 / scale**2
        available.sort(key=lambda p: p.area, reverse=True)
        poly = available.pop(0)
        if abs(poly.area - target_px2) / target_px2 < 0.01:
            apt, rem = poly, None
        else:
            apt, rem = split_polygon_at_area(poly, target_px2)
        apartments.append((t, apt))
        if rem and rem.area > 0:
            available.append(rem)
    return apartments

# --- Показать координаты курсора (последняя точка) ---
if canvas_result.json_data:
    objs = canvas_result.json_data.get("objects", [])
    if objs:
        last = objs[-1]
        if last.get("type") == "polygon":
            pt = last["points"][-1]
            x_mm, y_mm = pt["x"] * scale, pt["y"] * scale
            st.write(f"Координаты последней точки: x={x_mm:.1f} мм, y={y_mm:.1f} мм")

# --- Кнопка запуска разметки ---
if st.button("Подобрать квартирографию"):
    data = canvas_result.json_data
    if not data or not data.get("objects"):
        st.error("Нарисуйте внешний контур и, при необходимости, МОПы.")
    else:
        polys = get_shapely_polygons(data)
        if not polys:
            st.error("Нет корректных полигонов для разметки.")
        else:
            floor = polys[0]
            for hole in polys[1:]:
                floor = floor.difference(hole)
            apartments = layout_floor(floor)

            # Визуализация результата
            fig, ax = plt.subplots()
            cmap = {"studio":"#FFC107","1BR":"#8BC34A",
                    "2BR":"#03A9F4","3BR":"#E91E63","4BR":"#9C27B0"}
            for t, poly in apartments:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.7, fc=cmap[t], ec="black", linewidth=0.5)
            ax.set_aspect("equal")
            ax.axis("off")
            st.pyplot(fig)

            # Итоги по количеству квартир
            counts = {}
            for t, _ in apartments:
                counts[t] = counts.get(t, 0) + 1
            st.markdown("### Результат расчёта")
            st.write(f"- Этажей: **{floors}**")
            st.write("- Квартир на этаже:")
            for t, cnt in counts.items():
                st.write(f"  - {t}: {cnt}")
            st.write("- Всего квартир в доме:")
            for t, cnt in counts.items():
                st.write(f"  - {t}: {cnt * floors}")
