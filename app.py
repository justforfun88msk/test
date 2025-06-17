import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import matplotlib.pyplot as plt
import math

st.set_page_config(layout="wide")
st.title("Квартирография Streamlit")

# ——— Sidebar: параметры ———
st.sidebar.header("Параметры здания")
floors = st.sidebar.number_input("Этажей в доме", min_value=1, value=5)

st.sidebar.header("Доли типов квартир (%)")
types = ["studio", "1BR", "2BR", "3BR", "4BR"]
percentages = {}
for t in types:
    percentages[t] = st.sidebar.slider(f"% {t}", 0, 100, 0)

st.sidebar.header("Диапазоны площадей (м²)")
areas = {}
for t in types:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_a = st.number_input(f"Мин {t}", min_value=1.0, value=20.0, step=1.0, key=f"min_{t}")
    with col2:
        max_a = st.number_input(f"Макс {t}", min_value=1.0, value=50.0, step=1.0, key=f"max_{t}")
    areas[t] = (min_a, max_a)

st.sidebar.markdown("---")
st.sidebar.write("**Инструкция:**")
st.sidebar.write("1. Выберите инструмент «Polygon».")
st.sidebar.write("2. Нарисуйте **внешний контур** (один полигон).")
st.sidebar.write("3. Нарисуйте **внутренние полости (МОП)** — дополнительные полигоны.")

# ——— Canvas для рисования плана ———
st.write("### Нарисуйте план этажа")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=2,
    stroke_color="#000",
    background_color="#eee",
    drawing_mode="polygon",
    key="canvas",
    height=600,
    width=800,
)

# ——— Утилиты геометрии ———
def get_shapely_polygons(json_data):
    objs = json_data.get("objects", [])
    polys = []
    for o in objs:
        if o["type"] == "polygon":
            pts = o["points"]
            coords = [(p["x"], p["y"]) for p in pts]
            if len(coords) >= 3:
                polys.append(Polygon(coords))
    return polys

def split_polygon_at_area(polygon, target_area, tol=1e-3):
    """Разбивает polygon на кусок площадью target_area и остаток."""
    # 1) Найти ориентированный MRR и главную ось
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    # выбрать самую длинную сторону
    max_len = 0
    for i in range(len(coords)-1):
        x1,y1 = coords[i]; x2,y2 = coords[i+1]
        d = math.hypot(x2-x1, y2-y1)
        if d > max_len:
            max_len = d
            major = ((x1,y1),(x2,y2))
    (x1,y1),(x2,y2) = major
    ux, uy = (x2-x1)/max_len, (y2-y1)/max_len

    # проекции вершин
    projs = [ux*x + uy*y for x,y in polygon.exterior.coords]
    low, high = min(projs), max(projs)

    # функция для линии реза на смещении offset по оси (ux,uy)
    def make_cut(offset):
        # точка на оси
        mx, my = ux*offset, uy*offset
        # вектор перпендикулярный
        vx, vy = -uy, ux
        # длина линии — двойной диагональ bbox
        minx, miny, maxx, maxy = polygon.bounds
        diag = math.hypot(maxx-minx, maxy-miny) * 2
        p1 = (mx + vx*diag, my + vy*diag)
        p2 = (mx - vx*diag, my - vy*diag)
        return LineString([p1,p2])

    # бинарный поиск смещения
    for _ in range(30):
        mid = (low + high) / 2
        line = make_cut(mid)
        parts = split(polygon, line)
        if len(parts) < 2:
            low = mid
            continue
        # выбрать часть с центроидом на меньшей проекции
        areas_list = []
        for part in parts:
            cx, cy = part.representative_point().coords[0]
            proj = ux*cx + uy*cy
            areas_list.append((proj, part))
        smaller = min(areas_list, key=lambda x: x[0])[1]
        a = smaller.area
        if a > target_area * (1 + tol):
            high = mid
        elif a < target_area * (1 - tol):
            low = mid
        else:
            # нашли подходящий кусок
            remainder = [p for p in parts if not p.equals(smaller)][0]
            return smaller, remainder

    # финальный рез
    parts = split(polygon, make_cut((low+high)/2))
    sorted_parts = sorted(parts, key=lambda p: p.area)
    return sorted_parts[0], sorted_parts[1]

def layout_floor(floor_poly, percentages, areas):
    total_area = floor_poly.area
    desired = []
    # генерируем список целевых площадей по типам
    for t,pct in percentages.items():
        if pct <= 0: continue
        area_t = total_area * pct / 100
        min_a, max_a = areas[t]
        avg_a = (min_a + max_a) / 2
        count = max(1, int(round(area_t / avg_a)))
        per = area_t / count
        desired += [(t, per)] * count

    # сортируем от большего к меньшему
    desired.sort(key=lambda x: x[1], reverse=True)

    available = [floor_poly]
    apartments = []

    for t, target in desired:
        # берём самый большой доступный кусок
        available.sort(key=lambda p: p.area, reverse=True)
        poly = available.pop(0)
        # резать или сразу брать
        if abs(poly.area - target) / target < 0.01:
            apt = poly
            rem = None
        else:
            apt, rem = split_polygon_at_area(poly, target)
        apartments.append((t, apt))
        if rem and rem.area > 0:
            available.append(rem)

    return apartments

# ——— Обработка кнопки ———
if st.button("Подобрать квартирографию"):
    data = canvas_result.json_data
    if not data:
        st.error("Нарисуйте план на холсте.")
    else:
        polys = get_shapely_polygons(data)
        if len(polys) < 1:
            st.error("Нарисуйте внешний контур.")
        else:
            # первый — контур, остальные — МОПы
            floor = polys[0]
            for hole in polys[1:]:
                floor = floor.difference(hole)

            apartments = layout_floor(floor, percentages, areas)

            # ——— Визуализация результата ———
            fig, ax = plt.subplots()
            cmap = {
                "studio":"#FFC107","1BR":"#8BC34A",
                "2BR":"#03A9F4","3BR":"#E91E63","4BR":"#9C27B0"
            }
            for t, poly in apartments:
                x,y = poly.exterior.xy
                ax.fill(x, y, alpha=0.7, fc=cmap[t], ec="black", linewidth=0.5)
            ax.set_aspect("equal"); ax.axis("off")
            st.pyplot(fig)

            # ——— Итоги по количеству ———
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
