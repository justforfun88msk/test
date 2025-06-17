import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import matplotlib.pyplot as plt
import pandas as pd
import json, math

# ==========================
#   КОНФИГУРАЦИЯ И UI
# ==========================

st.set_page_config(layout="wide", page_title="Квартирография")
st.title("📐 Квартирография Architect Edition")

# — Sidebar: Общие настройки —
st.sidebar.header("🏢 Параметры здания и сетки")
floors = st.sidebar.number_input("Этажей в доме", min_value=1, value=10)
scale = st.sidebar.number_input("мм на пиксель", min_value=0.1, value=10.0, step=0.1)
show_snap = st.sidebar.checkbox("Привязка к сетке (snap-to-grid)", value=True)
grid_mm = st.sidebar.number_input("Шаг сетки (мм)", min_value=5, value=100, step=5)

# — Sidebar: Квартирография —
st.sidebar.header("🏠 Распределение квартир")
types = ['Студия','1С','2С','3С','4С']
percentages = {}
cols = st.sidebar.columns(2)
for i, t in enumerate(types):
    with cols[i % 2]:
        percentages[t] = st.slider(f"% {t}", 0, 100, 100//len(types), key=f"pct_{t}")
if sum(percentages.values()) != 100:
    st.sidebar.error(f"Сумма % должна быть 100% (сейчас {sum(percentages.values())}%)")
    st.stop()

st.sidebar.subheader("Диапазоны площадей (м²)")
areas = {}
for t in types:
    mn, mx = st.sidebar.slider(f"{t}", 1.0, 200.0, (20.0,50.0), key=f"area_{t}")
    areas[t] = (mn, mx)

# — Sidebar: Проект —
st.sidebar.header("💾 Проект")
proj_name = st.sidebar.text_input("Имя проекта (JSON)", "plan.json")

# ==========================
#   CANVAS: ЧЕРЧЕНИЕ ПЛАНА
# ==========================

st.subheader("1️⃣ Нарисуйте план этажа")
st.markdown("Первый полигон — внешний контур; остальные — зоны МОП.")
canvas_data = st_canvas(
        stroke_width=2,
        stroke_color='#000',
        fill_color='rgba(255,165,0,0.3)',
        background_color='#F0F0F0',
        drawing_mode='polygon',
        key='canvas2',
        width=800, height=600,
        initial_drawing=initial
    )
    )

# ==========================
#   ПОЛИГОН ЭТАЖА + МОП
# ==========================

def snap(pt):
    x,y = pt
    if not show_snap: return (x,y)
    g = grid_mm/scale
    return (round(x/g)*g, round(y/g)*g)

raw = canvas_data.json_data or {}
objs = raw.get('objects', [])
polys = []
for o in objs:
    if o.get('type') == 'polygon':
        pts = [snap((p['x'],p['y'])) for p in o['points']]
        if len(pts) >= 3: polys.append(Polygon(pts))
if not polys:
    st.error("Нарисуйте внешний контур!"); st.stop()
floor = polys[0]
for hole in polys[1:]:
    floor = floor.difference(hole)
# Отображение размеров
minx,miny,maxx,maxy = floor.bounds
w_mm = (maxx-minx)*scale; h_mm = (maxy-miny)*scale
area_m2 = floor.area * scale**2 / 1e6
st.info(f"Контур: {w_mm:.0f}×{h_mm:.0f} мм, площадь {area_m2:.2f} м²")

# ==========================
#   ФУНКЦИИ НАРЕЗКИ
# ==========================

def split_poly(poly, target_px2, tol=1e-2):
    # Главное: бинарный поиск по главной оси MRR
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    max_len = 0
    for i in range(len(coords)-1):
        x1,y1 = coords[i]; x2,y2 = coords[i+1]
        d = math.hypot(x2-x1, y2-y1)
        if d > max_len:
            max_len = d; major = ((x1,y1),(x2,y2))
    (x1,y1),(x2,y2) = major; ux, uy = (x2-x1)/max_len, (y2-y1)/max_len
    projs = [ux*x + uy*y for x,y in poly.exterior.coords]
    low, high = min(projs), max(projs)

    def make_cut(offset):
        mx,my = ux*offset, uy*offset; vx,vy = -uy,ux
        minx,miny,maxx,maxy = poly.bounds
        diag = math.hypot(maxx-minx, maxy-miny)*2
        return LineString([(mx+vx*diag, my+vy*diag),(mx-vx*diag, my-vy*diag)])

    for _ in range(30):
        mid = (low+high)/2; parts = split(poly, make_cut(mid))
        if len(parts) < 2:
            low = mid; continue
        # Выбираем ту часть, чей центр ближе к low
        areas_list = []
        for part in parts:
            cx,cy = part.representative_point().coords[0]
            proj = ux*cx + uy*cy
            areas_list.append((proj, part))
        smaller = min(areas_list, key=lambda x: x[0])[1]; a = smaller.area
        if a > target_px2*(1+tol): high = mid
        elif a < target_px2*(1-tol): low = mid
        else:
            rem = [p for p in parts if not p.equals(smaller)]
            return smaller, (rem[0] if rem else None)

    parts = split(poly, make_cut((low+high)/2))
    parts = sorted(parts, key=lambda p: p.area)
    return parts[0], (parts[1] if len(parts)>1 else None)

# ==========================
#   РАСПРЕДЕЛЕНИЕ КВАРТИР
# ==========================

st.subheader("2️⃣ Подбор квартирографии по всему зданию")
if st.button("🚀 Распределить квартиры"):
    total_area = area_m2 * floors
    avg_area = {t:(areas[t][0]+areas[t][1])/2 for t in types}
    counts = {t: max(1,int(round(total_area*percentages[t]/100/avg_area[t]))) for t in types}
    per_floor = {i:{} for i in range(floors)}
    for t,c in counts.items():
        q,r = divmod(c, floors)
        for i in range(floors): per_floor[i][t] = q + (1 if i < r else 0)

    fl = st.slider("Выберите этаж", 1, floors, 1)
    targets = []
    for t,n in per_floor[fl-1].items():
        if n > 0:
            tot_t = total_area * percentages[t] / 100
            avg_t = tot_t / counts[t]
            px2 = avg_t * 1e6 / scale**2
            targets += [(t, px2)]*n

    # Разметка
    avail = [floor]; placements = []
    for t,px2 in targets:
        avail.sort(key=lambda p: p.area, reverse=True)
        poly = avail.pop(0)
        apt, rem = split_poly(poly, px2)
        placements.append((t, apt))
        if rem and rem.area > 0: avail.append(rem)

    # Отрисовка этажа
    fig, ax = plt.subplots(figsize=(8,6))
    cmap = {'Студия':'#FFC107','1С':'#8BC34A','2С':'#03A9F4','3С':'#E91E63','4С':'#9C27B0'}
    for t, poly in placements:
        x,y = poly.exterior.xy
        ax.fill([xi*scale for xi in x], [yi*scale for yi in y], color=cmap[t], alpha=0.7, edgecolor='black')
    ax.set_aspect('equal'); ax.axis('off')
    st.pyplot(fig)

    # Отчет
    df = pd.DataFrame([{'Этаж':i+1,'Тип':t,'Кол-во':per_floor[i][t]} for i in range(floors) for t in types])
    st.subheader("3️⃣ Сводный отчет по этажам")
    st.dataframe(df)
    st.sidebar.download_button("📥 Скачать CSV", df.to_csv(index=False), file_name='report.csv', mime='text/csv')
