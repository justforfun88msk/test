import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import shapely.geometry as geom
import numpy as np

# Функции расчёта

def calculate_apartment_counts(total_area, percents, area_ranges):
    avg_areas = {t: (rng[0] + rng[1]) / 2 for t, rng in area_ranges.items()}
    target_areas = {t: total_area * p / 100 for t, p in percents.items()}
    counts = {t: max(1, int(round(target_areas[t] / avg_areas[t]))) for t in percents}
    return counts


def distribute_per_floor(counts, floors):
    per_floor = {f: {} for f in range(floors)}
    for t, cnt in counts.items():
        q, r = divmod(cnt, floors)
        for i in range(floors):
            per_floor[i][t] = q + (1 if i < r else 0)
    return per_floor


def layout_floor(perimeter_poly, mop_polys, apt_list, area_ranges):
    free_rects = [perimeter_poly.bounds]
    # Убираем МОПы
    for mop in mop_polys:
        new_free = []
        for (x, y, w, h) in free_rects:
            rect = geom.box(x, y, x + w, y + h)
            if rect.intersects(mop):
                diff = rect.difference(mop)
                for sub in getattr(diff, 'geoms', [diff]):
                    bx, by, bx2, by2 = sub.bounds
                    new_free.append((bx, by, bx2 - bx, by2 - by))
            else:
                new_free.append((x, y, w, h))
        free_rects = new_free

    placements = []
    for t in apt_list:
        if not free_rects:
            break
        free_rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = free_rects.pop(0)
        avg_area = (area_ranges[t][0] + area_ranges[t][1]) / 2
        side = np.sqrt(avg_area)
        rw = min(w, side)
        rh = avg_area / rw
        if rh > h:
            rh = min(h, side)
            rw = avg_area / rh
        placements.append((t, x, y, rw, rh, rw * rh))
        right = (x + rw, y, w - rw, h)
        top = (x, y + rh, rw, h - rh)
        for rect in [right, top]:
            if rect[2] > 0 and rect[3] > 0:
                free_rects.append(rect)
    return placements

# UI Настройки
st.sidebar.title('Настройки')
floors = st.sidebar.number_input('Число этажей', min_value=1, value=3, step=1)
grid_size_mm = st.sidebar.number_input('Размер сетки (мм)', min_value=10, value=100, step=10)
scale_px_per_mm = st.sidebar.number_input('Масштаб (px на 1 мм)', min_value=1, value=2, step=1)

types = ['Студии', '1С', '2С', '3С', '4С']

# Слайдер процентов
st.sidebar.subheader('Проценты квартир по площади')
percents = {}
for t in types:
    percents[t] = st.sidebar.slider(f'{t} (%)', 0, 100, 20, key=f'p_{t}')
sum_p = sum(percents.values())
color = 'green' if sum_p == 100 else 'red'
st.sidebar.markdown(f"<h3 style='color:{color}'>Сумма: {sum_p}%</h3>", unsafe_allow_html=True)

# Диапазоны площадей (float)
st.sidebar.subheader('Диапазон площадей (м²)')
def_area = {'Студии': (20.0, 35.0), '1С': (35.0, 50.0), '2С': (50.0, 70.0), '3С': (70.0, 90.0), '4С': (90.0, 120.0)}
area_ranges = {}
for t in types:
    min_default, max_default = def_area[t]
    min_val = st.sidebar.number_input(f'Мин {t}', min_value=5.0, value=min_default, step=0.1, key=f'min_{t}')
    max_val = st.sidebar.number_input(f'Макс {t}', min_value=min_val, value=max_default, step=0.1, key=f'max_{t}')
    area_ranges[t] = (min_val, max_val)

# Заголовок и холст
st.title('Планировщик квартирографии')
canvas = st_canvas(
    fill_color='transparent',
    stroke_width=2,
    background_color='#eeeeee',
    update_streamlit=True,
    height=600,
    width=800,
    drawing_mode='polygon',
    key='canvas',
    grid_color='#cccccc',
    grid_spacing=(int(grid_size_mm * scale_px_per_mm), int(grid_size_mm * scale_px_per_mm)),
    grid_width=1
)

# Генерация
if st.button('Сгенерировать квартирографию'):
    data = canvas.json_data
    if data and 'objects' in data:
        objs = data['objects']
        polys = [o for o in objs if o['type'] == 'polygon']
        if not polys:
            st.error('Нарисуйте периметр этажа')
        else:
            perimeter = geom.Polygon(polys[0]['points'])
            mops = [geom.Polygon(o['points']) for o in polys[1:]]
            area_px = perimeter.area - sum(m.area for m in mops)
            total_area = (area_px / (scale_px_per_mm ** 2)) / 1e6  # в м²

            counts = calculate_apartment_counts(total_area, percents, area_ranges)
            per_floor = distribute_per_floor(counts, floors)

            for i in range(floors):
                st.subheader(f'Этаж {i+1}')
                apt_list = [t for t, cnt in per_floor[i].items() for _ in range(cnt)]
                placements = layout_floor(perimeter, mops, apt_list, area_ranges)
                fig, ax = plt.subplots()
                xs, ys = perimeter.exterior.xy
                ax.plot(xs, ys, 'k-')
                for mop in mops:
                    xs, ys = mop.exterior.xy
                    ax.fill(xs, ys, color='grey', alpha=0.5)
                cmap = plt.get_cmap('tab20')
                idx_map = {t: i for i, t in enumerate(types)}
                for t, x, y, w, h, a in placements:
                    rect = Rectangle((x, y), w, h, facecolor=cmap(idx_map[t]), edgecolor='black', alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(x + w/2, y + h/2,
                            f"{t}
{int(w*scale_px_per_mm)}×{int(h*scale_px_per_mm)} мм
{a/1e6:.1f} м²",
                            ha='center', va='center', fontsize=6)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                st.pyplot(fig)
    else:
        st.error('Нет данных для построения')
