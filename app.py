import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import matplotlib.pyplot as plt
import math

# ——— Настройка страницы ———
st.set_page_config(layout="wide")
st.title("Квартирография Streamlit")

# ——— Sidebar: параметры здания ———
st.sidebar.header("Параметры здания")
floors = st.sidebar.number_input("Этажей в доме", min_value=1, value=5)
# По умолчанию 10 мм на пиксель
scale = st.sidebar.number_input("мм на пиксель", min_value=0.1, value=10.0, step=0.1)

# ——— Sidebar: проценты по типам квартир ———
st.sidebar.header("Доли типов квартир (%)")
types = ["studio", "1BR", "2BR", "3BR", "4BR"]
percentages = { t: st.sidebar.slider(f"% {t}", 0, 100, 0, key=f"pct_{t}") for t in types }

# ——— Sidebar: диапазоны площадей ———
st.sidebar.header("Диапазоны площадей (м²)")
areas = {}
for t in types:
    min_a = st.sidebar.number_input(f"Мин {t}", min_value=1.0, value=20.0, step=1.0, key=f"min_{t}")
    max_a = st.sidebar.number_input(f"Макс {t}", min_value=1.0, value=50.0, step=1.0, key=f"max_{t}")
    areas[t] = (min_a, max_a)

# ——— Canvas для рисования ———
st.write("### Нарисуйте план этажа (сначала внешний контур, затем внутренние МОП)")
canvas_result = st_canvas(
    stroke_width=2,
    stroke_color="black",
    fill_color="rgba(255, 165, 0, 0.3)",
    background_color="#EEE",
    drawing_mode="polygon",
    key="canvas",
    width=800,
    height=600,
)

# ——— Отображение размеров контура сразу после рисования ———
if canvas_result.json_data:
    data = canvas_result.json_data
    # Получаем первый полигон как внешний контур
    objs = data.get("objects", [])
    if objs and objs[0].get("type") == "polygon":
        pts = [(p["x"], p["y"]) for p in objs[0].get("points", [])]
        if len(pts) >= 3:
            poly = Polygon(pts)
            minx, miny, maxx, maxy = poly.bounds
            width_mm = (maxx - minx) * scale
            height_mm = (maxy - miny) * scale
            area_m2 = poly.area * scale**2 / 1e6
            st.markdown(f"**Контур:** ширина {width_mm:.0f} мм, высота {height_mm:.0f} мм, площадь {area_m2:.2f} м²")

# ——— Shapely-утилиты ———
def get_shapely_polygons(data):
    objs = data.get("objects", [])
    polys = []
    for o in objs:
        if o.get("type") == "polygon":
            pts = [(p["x"], p["y"]) for p in o.get("points", [])]
            if len(pts) >= 3:
                polys.append(Polygon(pts))
    return polys

def split_polygon_at_area(polygon, target_px2, tol=1e-2):
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    max_len = 0
    for i in range(len(coords)-1):
        x1,y1 = coords[i]; x2,y2 = coords[i+1]
        d = math.hypot(x2-x1, y2-y1)
        if d > max_len:
            max_len = d
            major = ((x1,y1),(x2,y2))
    (x1,y1),(x2,y2) = major
    ux, uy = (x2-x1)/max_len, (y2-y1)/max_len
    projs = [ux*x + uy*y for x,y in polygon.exterior.coords]
    low, high = min(projs), max(projs)
    def make_cut(offset):
        mx, my = ux*offset, uy*offset
        vx, vy = -uy, ux
        minx, miny, maxx, maxy = polygon.bounds
        diag = math.hypot(maxx-minx, maxy-miny)*2
        p1 = (mx+vx*diag, my+vy*diag)
        p2 = (mx-vx*diag, my-vy*diag)
        return LineString([p1,p2])
    for _ in range(30):
        mid = (low+high)/2
        parts = split(polygon, make_cut(mid))
        if len(parts)<2:
            low = mid
            continue
        areas_list = []
        for part in parts:
            cx,cy = part.representative_point().coords[0]
            proj = ux*cx + uy*cy
            areas_list.append((proj, part))
        smaller = min(areas_list, key=lambda x: x[0])[1]
        a = smaller.area
        if a > target_px2*(1+tol): high = mid
        elif a < target_px2*(1-tol): low = mid
        else:
            rem = [p for p in parts if not p.equals(smaller)]
            return smaller, (rem[0] if rem else None)
    parts = split(polygon, make_cut((low+high)/2))
    parts = sorted(parts, key=lambda p: p.area)
    return parts[0], (parts[1] if len(parts)>1 else None)

def layout_floor(floor_poly):
    total_m2 = floor_poly.area * scale**2 / 1e6
    targets = []
    for t,pct in percentages.items():
        if pct<=0: continue
        area_m2 = total_m2*pct/100
        mn,mx = areas[t]
        avg = (mn+mx)/2
        cnt = max(1, int(round(area_m2/avg)))
        for _ in range(cnt):
            px2 = (area_m2/cnt)*1e6/scale**2
            targets.append((t, px2))
    targets.sort(key=lambda x: x[1], reverse=True)
    available=[floor_poly]; apartments=[]
    for t,px2 in targets:
        available.sort(key=lambda p: p.area, reverse=True)
        poly=available.pop(0)
        if abs(poly.area-px2)/px2<0.02: apt,rem=poly,None
        else: apt,rem=split_polygon_at_area(poly,px2)
        apartments.append((t,apt))
        if rem and rem.area>0: available.append(rem)
    return apartments

# ——— Кнопка запуска ———
if st.button("Подобрать квартирографию"):
    data = canvas_result.json_data
    if not data or not data.get("objects"):
        st.error("Нарисуйте внешн...
