import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import matplotlib.pyplot as plt
import ezdxf
import json
import math

# ——— Настройка страницы ———
st.set_page_config(layout="wide")
st.title("Квартирография Streamlit")

# ——— Sidebar: параметры здания и сетки ———
st.sidebar.header("Параметры здания и сетки")
floors = st.sidebar.number_input("Этажей в доме", min_value=1, value=5)
scale = st.sidebar.number_input("мм на пиксель", min_value=0.1, value=10.0, step=0.1)
show_snap = st.sidebar.checkbox("Привязка к сетке (snap-to-grid)", value=True)
grid_step_mm = st.sidebar.number_input("Шаг сетки (мм)", min_value=1, value=100)

# ——— Sidebar: проценты по типам квартир ———
st.sidebar.header("Доли типов квартир (%)")
types = ["studio", "1BR", "2BR", "3BR", "4BR"]
percentages = {t: st.sidebar.slider(f"% {t}", 0, 100, 0, key=f"pct_{t}") for t in types}

# ——— Sidebar: диапазоны площадей ———
st.sidebar.header("Диапазоны площадей (м²)")
areas = {}
for t in types:
    mn = st.sidebar.number_input(f"Мин {t}", 1.0, 1000.0, 20.0, key=f"mn_{t}")
    mx = st.sidebar.number_input(f"Макс {t}", 1.0, 1000.0, 50.0, key=f"mx_{t}")
    areas[t] = (mn, mx)

# ——— Canvas: рисуем план ———
st.write("### Нарисуйте план этажа (щелк — новая вершина полигон)")
canvas_result = st_canvas(
    stroke_width=2,
    stroke_color="black",
    fill_color="rgba(255,165,0,0.3)",
    background_color="#EEE",
    drawing_mode="polygon",
    key="canvas",
    width=800,
    height=600
)

# ——— Сохранение/загрузка проекта ———
if canvas_result.json_data:
    js = json.dumps(canvas_result.json_data)
else:
    js = None
save_name = st.sidebar.text_input("Имя файла проекта", value="plan.json")
if js:
    st.sidebar.download_button("Скачать проект JSON", js, file_name=save_name, mime="application/json")
uploaded = st.sidebar.file_uploader("Загрузить проект JSON", type=["json"])
initial_data = None
if uploaded:
    initial_data = json.load(uploaded)

# ——— Утилиты: конвертация в Shapely ———
def get_polygons(data):
    polys = []
    for o in data.get("objects", []):
        if o.get("type") == "polygon":
            pts = []
            for p in o.get("points", []):
                x, y = p["x"], p["y"]
                if show_snap:
                    x = round((x*scale)/grid_step_mm) * grid_step_mm/scale
                    y = round((y*scale)/grid_step_mm) * grid_step_mm/scale
                pts.append((x, y))
            if len(pts) >= 3:
                polys.append(Polygon(pts))
    return polys

# ——— Получаем полигон этажа с учетом МОП ———
polys = get_polygons(canvas_result.json_data or initial_data or {})
floor_poly = None
if polys:
    floor_poly = polys[0]
    for hole in polys[1:]:
        floor_poly = floor_poly.difference(hole)

# ——— Сквозной расчет квартир ———
area_floor_m2 = floor_poly.area * scale**2 / 1e6 if floor_poly else 0
building_area = area_floor_m2 * floors
# количество квартир всего
apartment_counts = {}
for t, pct in percentages.items():
    avg_area = (areas[t][0] + areas[t][1]) / 2
    apartment_counts[t] = int(round(building_area * pct/100 / avg_area))
# делим по этажам
floor_distribution = {t: [] for t in types}
for t, ct in apartment_counts.items():
    q, r = divmod(ct, floors)
    for i in range(floors):
        floor_distribution[t].append(q + (1 if i < r else 0))

# ——— Функции нарезки полигонов ———
def split_polygon_at_area(poly, target_px2, tol=1e-2):
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    # главная ось
    max_len = 0
    for i in range(len(coords)-1):
        x1,y1 = coords[i]; x2,y2 = coords[i+1]
        d = math.hypot(x2-x1, y2-y1)
        if d > max_len:
            max_len = d; major = ((x1,y1),(x2,y2))
    (x1,y1),(x2,y2) = major
    ux, uy = (x2-x1)/max_len, (y2-y1)/max_len
    projs = [ux*x + uy*y for x,y in poly.exterior.coords]
    low, high = min(projs), max(projs)
    def make_cut(off):
        mx, my = ux*off, uy*off; vx,vy = -uy,ux
        minx,miny,maxx,maxy = poly.bounds
        diag = math.hypot(maxx-minx, maxy-miny)*2
        p1 = (mx+vx*diag, my+vy*diag)
        p2 = (mx-vx*diag, my-vy*diag)
        return LineString([p1,p2])
    for _ in range(30):
        mid = (low+high)/2
        parts = split(poly, make_cut(mid))
        if len(parts)<2:
            low = mid; continue
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
    parts = split(poly, make_cut((low+high)/2))
    parts = sorted(parts, key=lambda p: p.area)
    return parts[0], (parts[1] if len(parts)>1 else None)

def layout_floor(floor_poly, targets):
    available = [floor_poly]
    apartments = []
    for t, px2 in targets:
        available.sort(key=lambda p: p.area, reverse=True)
        poly = available.pop(0)
        if abs(poly.area-px2)/px2 < 0.02:
            apt, rem = poly, None
        else:
            apt, rem = split_polygon_at_area(poly, px2)
        apartments.append((t, apt))
        if rem and rem.area>0: available.append(rem)
    return apartments

# ——— Кнопка: показать первый этаж ———
if st.button("Показать разметку первого этажа") and floor_poly:
    # готовим targets именно для 1 этажа
    targets = []
    for t in types:
        cnt = floor_distribution[t][0]
        if cnt>0:
            total_area_t = building_area * percentages[t]/100
            avg_m2 = total_area_t / apartment_counts[t] if apartment_counts[t]>0 else 0
            for _ in range(cnt):
                targets.append((t, avg_m2*1e6/scale**2))
    apartments = layout_floor(floor_poly, targets)
    fig, ax = plt.subplots()
    cmap = {"studio":"#FFC107","1BR":"#8BC34A","2BR":"#03A9F4","3BR":"#E91E63","4BR":"#9C27B0"}
    for t, poly in apartments:
        x,y = poly.exterior.xy
        ax.fill([xi*scale for xi in x], [yi*scale for yi in y], fc=cmap[t], ec="black", linewidth=0.5)
    ax.set_aspect('equal'); ax.axis('off')
    st.pyplot(fig)

# ——— Сводный отчет ———
st.header("Сводный отчет по этажам")
rows=[]
for i in range(floors):
    for t in types:
        rows.append({"этаж":i+1, "тип":t, "кол-во":floor_distribution[t][i]})
st.dataframe(rows)
if st.button("Скачать отчет CSV"):
    csv = "этаж,тип,кол-во\n" + "\n".join(f"{r['этаж']},{r['тип']},{r['кол-во']}" for r in rows)
    st.download_button("Скачать CSV", csv, file_name="report.csv", mime="text/csv")
