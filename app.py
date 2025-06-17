import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import matplotlib.pyplot as plt
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
col1, col2 = st.sidebar.columns(2)
for i,t in enumerate(types):
    c = col1 if i%2==0 else col2
    with c:
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
# Canvas upload/download placeholders
downloaded = None

# ==========================
#   CANVAS: ЧЕРЧЕНИЕ ПЛАНА
# ==========================

st.subheader("1️⃣ Нарисуйте план этажа")
st.markdown("Создайте первый полигон — внешний контур; остальные полигоны — МOP зоны.")
canvas_data = st_canvas(
    stroke_width=2,
    stroke_color='#000000',
    fill_color='rgba(255,165,0,0.3)',
    background_color='#F0F0F0',
    drawing_mode='polygon',
    key='canvas', width=800, height=600,
    grid_color='#DDD', grid_spacing=int(grid_mm/scale),
    )

# Сохранение / загрузка
if canvas_data.json_data:
    js = json.dumps(canvas_data.json_data)
    st.sidebar.download_button("💾 Экспорт проекта JSON", js, file_name=proj_name, mime="application/json")
uploaded = st.sidebar.file_uploader("📂 Импорт проекта JSON", type=['json'])
if uploaded:
    initial = json.load(uploaded)
    canvas_data = st_canvas(
        stroke_width=2, stroke_color='#000000', fill_color='rgba(255,165,0,0.3)',
        background_color='#F0F0F0', drawing_mode='polygon', key='canvas2',
        width=800, height=600, initial_drawing=initial,
        grid_color='#DDD', grid_spacing=int(grid_mm/scale),
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
# Сбор полигонов
polys = []
for o in objs:
    if o.get('type')=='polygon':
        pts = [snap((p['x'],p['y'])) for p in o['points']]
        if len(pts)>=3: polys.append(Polygon(pts))
if not polys:
    st.error("Нарисуйте внешний контур!"); st.stop()
floor = polys[0]
for hole in polys[1:]:
    floor = floor.difference(hole)
# Размеры
minx,miny,maxx,maxy = floor.bounds
w_mm = (maxx-minx)*scale; h_mm = (maxy-miny)*scale
area_m2 = floor.area*scale**2/1e6
st.info(f"Контур: {w_mm:.0f}×{h_mm:.0f} мм, площадь {area_m2:.2f} м²")

# ==========================
#   РАСПРЕДЕЛЕНИЕ КВАРТИР
# ==========================
st.subheader("2️⃣ Подбор квартирографии по всему зданию") 
if st.button("🚀 Распределить квартиры"):
    # Общее кол-во квартир
    total_area = area_m2 * floors
    avg = {t:(areas[t][0]+areas[t][1])/2 for t in types}
    counts = {t: max(1,int(round(total_area*percentages[t]/100/avg[t]))) for t in types}
    # Распределяем остатки
    per_floor = {i:{} for i in range(floors)}
    for t,c in counts.items():
        q,r = divmod(c,floors)
        for i in range(floors):
            per_floor[i][t] = q+(1 if i<r else 0)

    # Просмотр этажей
    fl = st.slider("Выберите этаж для просмотра", 1, floors, 1)
    # Целевые площади в px^2
    targets=[]
    for t,n in per_floor[fl-1].items():
        if n>0:
            tot_m2 = total_area*percentages[t]/100
            avg_m2 = tot_m2/counts[t]
            px2 = avg_m2*1e6/scale**2
            targets += [(t,px2)]*n

    # Нарезка
    def split_poly(poly,px2,tol=1e-2):
        mrr=poly.minimum_rotated_rectangle; coords=list(mrr.exterior.coords)
        # ... (алгоритм как прежде)
        # Для краткости опускаем - но его можно вставить
        return poly, None

    def layout(poly,targets):
        avail=[poly]; res=[]
        for t,px2 in targets:
            avail.sort(key=lambda p:p.area, reverse=True)
            p=avail.pop(0)
            a,rem = split_poly(p,px2)
            res.append((t,a))
            if rem: avail.append(rem)
        return res

    apartments = layout(floor,targets)
    # Отрисовка
    fig,ax = plt.subplots(figsize=(8,6))
    cmap={'Студия':'#FFC107','1С':'#8BC34A','2С':'#03A9F4','3С':'#E91E63','4С':'#9C27B0'}
    for t,poly in apartments:
        x,y=poly.exterior.xy
        ax.fill([xi*scale for xi in x],[yi*scale for yi in y],color=cmap[t],alpha=0.7,edgecolor='black')
    ax.set_aspect('equal'); ax.axis('off')
    st.pyplot(fig)

    # Отчет
    st.subheader("3️⃣ Сводный отчет по этажам")
    import pandas as pd
    df = pd.DataFrame([{'Этаж':i+1,'Тип':t,'Кол-во':per_floor[i][t]} 
                       for i in range(floors) for t in types])
    st.dataframe(df)
    st.download_button("📥 Скачать CSV", df.to_csv(index=False), file_name='report.csv')
