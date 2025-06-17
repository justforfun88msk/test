import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import shapely.geometry as geom
import shapely.ops as ops
import numpy as np

def calculate_apartment_counts(total_area, percents, area_ranges):
    # calculate average area per type
    avg_areas = {t: (rng[0] + rng[1]) / 2 for t, rng in area_ranges.items()}
    # target areas
    target_areas = {t: total_area * p / 100 for t, p in percents.items()}
    # counts
    counts = {t: max(1, int(round(target_areas[t] / avg_areas[t]))) for t in percents}
    return counts

def distribute_per_floor(counts, floors):
    # distribute counts per floor using divmod
    per_floor = {f: {} for f in range(floors)}
    for t, cnt in counts.items():
        q, r = divmod(cnt, floors)
        for i in range(floors):
            per_floor[i][t] = q + (1 if i < r else 0)
    return per_floor

def layout_floor(perimeter_poly, mop_polys, apt_list, scale):
    # use guillotine packing on free rectangles
    minx, miny, maxx, maxy = perimeter_poly.bounds
    free_rects = [(minx, miny, maxx - minx, maxy - miny)]
    # subtract mops
    for mop in mop_polys:
        # partition free_rects that intersect mop
        new_free = []
        for (x, y, w, h) in free_rects:
            rect_poly = geom.box(x, y, x + w, y + h)
            if rect_poly.intersects(mop):
                diff = rect_poly.difference(mop)
                for sub in getattr(diff, 'geoms', [diff]):
                    bx, by, bx2, by2 = sub.bounds
                    new_free.append((bx, by, bx2 - bx, by2 - by))
            else:
                new_free.append((x, y, w, h))
        free_rects = new_free
    placements = []
    for t in apt_list:
        # pick largest free rect
        if not free_rects:
            break
        free_rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = free_rects.pop(0)
        # target area and size
        area = (scale[t][0] + scale[t][1]) / 2  # avg m²
        side = np.sqrt(area)
        rw = min(w, side)
        rh = area / rw
        # ensure fits
        if rh > h:
            rh = min(h, side)
            rw = area / rh
        placements.append((t, x, y, rw, rh, rw * rh))
        # split leftover by guillotine
        right = (x + rw, y, w - rw, h)
        top = (x, y + rh, rw, h - rh)
        for rect in [right, top]:
            if rect[2] > 1 and rect[3] > 1:
                free_rects.append(rect)
    return placements

# Streamlit UI
st.sidebar.title('Настройки')
floors = st.sidebar.number_input('Число этажей', min_value=1, value=3, step=1)
grid_size_mm = st.sidebar.number_input('Размер сетки (мм)', min_value=10, value=100, step=10)
scale_px_per_mm = st.sidebar.number_input('Масштаб (px на 1 мм)', min_value=1, value=2)
# apartment percent sliders
st.sidebar.subheader('Проценты квартир по площади')
types = ['Студии', '1С', '2С', '3С', '4С']
percents = {}
for t in types:
    percents[t] = st.sidebar.slider(f'{t} (%)', 0, 100, 20, key=f'p_{t}')
sum_p = sum(percents.values())
col = 'green' if sum_p == 100 else 'red'
st.sidebar.markdown(f"<h3 style='color:{col}'>Сумма: {sum_p}%</h3>", unsafe_allow_html=True)

# area ranges
st.sidebar.subheader('Диапазон площадей (м²)')
area_ranges = {}
def_area = {'Студии':(20,35),'1С':(35,50),'2С':(50,70),'3С':(70,90),'4С':(90,120)}
for t in types:
    min_a = st.sidebar.number_input(f'Мин {t}', min_value=5.0, value=def_area[t][0], key=f'min_{t}')
    max_a = st.sidebar.number_input(f'Макс {t}', min_value=min_a, value=def_area[t][1], key=f'max_{t}')
    area_ranges[t] = (min_a, max_a)

st.title('Планировщик квартирографии')
canvas_result = st_canvas(
    fill_color='transparent',
    stroke_width=2,
    background_color='#eeeeee',
    update_streamlit=True,
    height=600,
    width=800,
    drawing_mode='polygon',
    key='canvas',
    grid_color='#cccccc',
    grid_spacing=grid_size_mm * scale_px_per_mm,
    grid_width=1
)

if st.button('Сгенерировать квартирографию'):
    if canvas_result.json_data and 'objects' in canvas_result.json_data:
        objs = canvas_result.json_data['objects']
        polys = [obj for obj in objs if obj['type']=='polygon']
        if not polys:
            st.error('Нарисуйте периметр')
        else:
            # first polygon is perimeter, others are MOPs
            coords = polys[0]['points']
            perimeter = geom.Polygon(coords)
            mops = [geom.Polygon(obj['points']) for obj in polys[1:]]
            # compute total area in m² (assuming 1 px = 1 mm*scale)
            area_px = perimeter.area - sum([m.area for m in mops])
            area_mm2 = area_px / (scale_px_per_mm**2)
            total_area = area_mm2 / 1e6  # mm² to m²
            # calculate counts
            counts = calculate_apartment_counts(total_area, percents, area_ranges)
            per_floor = distribute_per_floor(counts, floors)
            # layout each floor
            for i in range(floors):
                st.subheader(f'Этаж {i+1}')
                apt_list = []
                for t, cnt in per_floor[i].items():
                    apt_list += [t] * cnt
                placements = layout_floor(perimeter, mops, apt_list, area_ranges)
                fig, ax = plt.subplots()
                # draw perimeter and mops
                xs, ys = perimeter.exterior.xy
                ax.plot(xs, ys, 'k-')
                for mop in mops:
                    xs, ys = mop.exterior.xy
                    ax.fill(xs, ys, color='grey', alpha=0.5)
                # colors
                cmap = plt.get_cmap('tab20')
                type_idx = {t: idx for idx, t in enumerate(types)}
                # draw apartments
                for t, x, y, w, h, a in placements:
                    color = cmap(type_idx[t])
                    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(x + w/2, y + h/2, f'{t}\n{w*scale_px_per_mm:.0f}×{h*scale_px_per_mm:.0f} мм\n{a*1e-6:.1f} м²',
                            ha='center', va='center', fontsize=6)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                st.pyplot(fig)
    else:
        st.error('Нет данных с холста')
