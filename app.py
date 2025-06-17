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

# ——— Sidebar: основные параметры ———
st.sidebar.header("Параметры здания и сетки")
floors = st.sidebar.number_input("Этажей в доме", min_value=1, value=5)
# по умолчанию 10 мм/px
scale = st.sidebar.number_input("мм на пиксель", min_value=0.1, value=10.0, step=0.1)
# сетка
show_grid = st.sidebar.checkbox("Показывать сетку", value=True)
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

# ——— Сохранение/загрузка проекта ———
st.sidebar.header("Проект")
proj_file = st.sidebar.text_input("Имя проекта", value="plan.json")
if st.sidebar.button("Сохранить проект"):
    st.sidebar.download_button("Скачать JSON", json.dumps(canvas_result.json_data), file_name=proj_file)
uploaded = st.sidebar.file_uploader("Загрузить JSON проекта", type=["json"])
initial_data = None
if uploaded:
    initial_data = json.load(uploaded)

# ——— Canvas с grid и snap-to-grid ———
st.write("### Рисунок плана (щелк — вершина нового полигона)")
# background grid SVG
grid_svg = ""
if show_grid:
    step_px = grid_step_mm / scale
    lines = []
    for i in range(int(800/step_px)+1):
        x = i*step_px
        lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="600" stroke="#ccc" stroke-width="0.5"/>')
    for j in range(int(600/step_px)+1):
        y = j*step_px
        lines.append(f'<line x1="0" y1="{y}" x2="800" y2="{y}" stroke="#ccc" stroke-width="0.5"/>')
    grid_svg = """<svg xmlns='http://www.w3.org/2000/svg' width='800' height='600'>%s</svg>""" % "".join(lines)
canvas_result = st_canvas(
    background_color="#eee",
    background_image=None,
    initial_drawing=initial_data,
    drawing_mode="polygon",
    stroke_width=2,
    stroke_color="black",
    fill_color="rgba(255,165,0,0.3)",
    width=800,
    height=600,
    help="Каждая вершина привязывается к сетке",
    svg=grid_svg,
)

# ——— Конвертация JSON в Shapely-полигоны с snap to grid ———
def get_polygons(data):
    objs = data.get("objects", [])
    polys = []
    for o in objs:
        if o.get("type")=="polygon":
            pts = []
            for p in o.get("points", []):
                x = round((p["x"]*scale)/grid_step_mm)*grid_step_mm/scale
                y = round((p["y"]*scale)/grid_step_mm)*grid_step_mm/scale
                pts.append((x,y))
            if len(pts)>=3:
                polys.append(Polygon(pts))
    return polys

# ——— Сквозной расчёт квартирографии ———
# Общее количество квартир по типам
total_perc = sum(percentages.values())
# общая площадь этажа без МОП
polys = get_polygons(canvas_result.json_data or initial_data or {})
floor_poly = polys[0] if polys else None
for hole in polys[1:]:
    floor_poly = floor_poly.difference(hole)
area_floor_m2 = floor_poly.area*scale**2/1e6 if floor_poly else 0
building_area = area_floor_m2 * floors
# считаем целые числа квартир
apartment_counts = {}
for t, pct in percentages.items():
    apartment_counts[t] = int(round(building_area*pct/100/((areas[t][0]+areas[t][1])/2)))
# распределяем по этажам divmod
floor_distribution = {t: [] for t in types}
for t, count in apartment_counts.items():
    q, r = divmod(count, floors)
    for i in range(floors):
        floor_distribution[t].append(q + (1 if i<r else 0))

# ——— Генерация DXF и SVG ———
if floor_poly and st.button("Экспорт DXF/SVG"):
    # создаём DXF
    doc = ezdxf.new()
    msp = doc.modelspace()
    # рисуем квартиры первого этажа
    targets = []
    for t in types:
        for _ in range(floor_distribution[t][0]):
            targets.append((t, area_floor_m2*pct/100/floor_distribution[t][0]))
    # упрощённо: прямое splitом
    apartments = layout_floor(floor_poly)
    for t, poly in apartments:
        pts = [(x*scale, y*scale) for x,y in poly.exterior.coords]
        msp.add_lwpolyline(pts, close=True)
    doc.saveas("apartments.dxf")
    # SVG
    fig, ax = plt.subplots()
    cmap = {"studio":"#FFC107","1BR":"#8BC34A","2BR":"#03A9F4","3BR":"#E91E63","4BR":"#9C27B0"}
    for t, poly in apartments:
        x,y = poly.exterior.xy
        ax.fill([xi*scale for xi in x], [yi*scale for yi in y], fc=cmap[t], ec="black", linewidth=0.5)
    ax.set_aspect('equal'); ax.axis('off')
    fig.savefig("apartments.svg", bbox_inches='tight')
    st.success("Экспорт завершён: apartments.dxf, apartments.svg")

# ——— Функция нарезки этажа (отдельно для DXF/SVG) ———
def split_polygon_at_area(polygon, target_px2, tol=1e-2):
    # ... (код без изменений)
    return parts[0], (parts[1] if len(parts)>1 else None)

def layout_floor(floor_poly):
    # ... (код без изменений)
    return apartments

# ——— Показ результатов для первого этажа ———
if st.button("Показать разметку первого этажа") and floor_poly:
    apartments = layout_floor(floor_poly)
    fig, ax = plt.subplots()
    cmap = {"studio":"#FFC107","1BR":"#8BC34A","2BR":"#03A9F4","3BR":"#E91E63","4BR":"#9C27B0"}
    for t, poly in apartments:
        x,y = poly.exterior.xy
        ax.fill([xi*scale for xi in x],[yi*scale for yi in y], fc=cmap[t], ec="black", linewidth=0.5)
    ax.set_aspect('equal'); ax.axis('off')
    st.pyplot(fig)

# ——— Отчёт по всем этажам ———
st.header("Сводный отчёт")
df = []
for i in range(floors):
    for t in types:
        df.append({"этаж": i+1, "тип": t, "кол-во": floor_distribution[t][i]})
st.dataframe(df)
if st.button("Скачать отчёт CSV"):
    st.download_button("Скачать CSV", json.dumps(df), file_name="report.json")
