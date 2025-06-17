import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import shapely.geometry as geom
import numpy as np
from matplotlib.colors import ListedColormap
from shapely.validation import make_valid

# Константы
GRID_MIN = 10
GRID_MAX = 500
SCALE_MIN = 1
SCALE_MAX = 10
DEFAULT_TYPES = ['Студии', '1С', '2С', '3С', '4С']
DEFAULT_AREAS = {
    'Студии': (20.0, 35.0),
    '1С': (35.0, 50.0),
    '2С': (50.0, 70.0),
    '3С': (70.0, 90.0),
    '4С': (90.0, 120.0)
}
COLORS = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']).colors

def validate_polygon(poly):
    """Валидация и исправление полигона"""
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.geom_type == 'MultiPolygon':
        poly = max(poly.geoms, key=lambda x: x.area)
    return poly

def calculate_apartment_counts(total_area, percents, area_ranges):
    """Расчет количества квартир каждого типа"""
    avg_areas = {t: (rng[0] + rng[1]) / 2 for t, rng in area_ranges.items()}
    target_areas = {t: total_area * p / 100 for t, p in percents.items()}
    counts = {t: max(1, int(round(target_areas[t] / avg_areas[t]))) for t in percents}
    return counts

def distribute_per_floor(counts, floors):
    """Распределение квартир по этажам"""
    per_floor = {f: {} for f in range(floors)}
    for t, cnt in counts.items():
        q, r = divmod(cnt, floors)
        for i in range(floors):
            per_floor[i][t] = q + (1 if i < r else 0)
    return per_floor

def split_rectangles(free_rects, obstacle):
    """Разделение прямоугольников вокруг препятствия"""
    new_free = []
    for (x, y, w, h) in free_rects:
        rect = geom.box(x, y, x + w, y + h)
        if rect.intersects(obstacle):
            diff = rect.difference(obstacle)
            if diff.is_empty:
                continue
            for sub in getattr(diff, 'geoms', [diff]):
                bx, by, bx2, by2 = sub.bounds
                new_w, new_h = bx2 - bx, by2 - by
                if new_w > 1 and new_h > 1:  # Минимальный размер
                    new_free.append((bx, by, new_w, new_h))
        else:
            new_free.append((x, y, w, h))
    return new_free

def layout_floor(perimeter_poly, mop_polys, apt_list, area_ranges, scale):
    """Размещение квартир на этаже"""
    # Конвертация в метры
    scale_m = scale / 1000  # px to meters
    
    # Начальное свободное пространство
    free_rects = [perimeter_poly.bounds]
    
    # Удаление МОП
    for mop in mop_polys:
        free_rects = split_rectangles(free_rects, mop)
    
    placements = []
    remaining_apts = apt_list.copy()
    
    while remaining_apts and free_rects:
        free_rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        x, y, w_px, h_px = free_rects.pop(0)
        
        # Конвертация в метры
        w_m = w_px * scale_m
        h_m = h_px * scale_m
        
        # Находим лучшую квартиру для этого пространства
        best_fit = None
        best_idx = None
        best_area_diff = float('inf')
        
        for idx, t in enumerate(remaining_apts):
            min_a, max_a = area_ranges[t]
            target_a = (min_a + max_a) / 2
            
            # Пробуем горизонтальную и вертикальную ориентацию
            for orientation in ['h', 'v']:
                if orientation == 'h':
                    apt_w = min(w_m, np.sqrt(target_a * (w_m / h_m)))
                    apt_h = target_a / apt_w
                else:
                    apt_h = min(h_m, np.sqrt(target_a * (h_m / w_m)))
                    apt_w = target_a / apt_h
                
                if apt_w <= w_m and apt_h <= h_m:
                    area_diff = abs((apt_w * apt_h) - target_a)
                    if area_diff < best_area_diff:
                        best_fit = (t, x, y, apt_w / scale_m, apt_h / scale_m)
                        best_idx = idx
                        best_area_diff = area_diff
        
        if best_fit:
            t, x, y, apt_w_px, apt_h_px = best_fit
            placements.append((t, x, y, apt_w_px, apt_h_px))
            del remaining_apts[best_idx]
            
            # Добавляем оставшееся пространство
            right = (x + apt_w_px, y, w_px - apt_w_px, h_px)
            top = (x, y + apt_h_px, apt_w_px, h_px - apt_h_px)
            
            for rect in [right, top]:
                if rect[2] > 10 and rect[3] > 10:  # Минимальный размер
                    free_rects.append(rect)
    
    return placements

def draw_floor_plan(ax, perimeter, mops, placements, scale, types):
    """Отрисовка плана этажа"""
    # Конвертация в метры
    scale_m = scale / 1000  # px to meters
    
    # Рисуем периметр
    xs, ys = perimeter.exterior.xy
    ax.plot(xs, ys, 'k-', linewidth=2)
    
    # Рисуем МОПы
    for mop in mops:
        xs, ys = mop.exterior.xy
        ax.fill(xs, ys, color='#aaaaaa', alpha=0.5)
        ax.plot(xs, ys, 'k-', linewidth=1)
    
    # Рисуем квартиры
    type_indices = {t: i for i, t in enumerate(types)}
    for t, x, y, w_px, h_px in placements:
        w_m = w_px * scale_m
        h_m = h_px * scale_m
        area = w_m * h_m
        
        color = COLORS[type_indices[t] % len(COLORS)]
        rect = Rectangle((x, y), w_px, h_px, 
                        facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        ax.text(x + w_px/2, y + h_px/2,
               f"{t}\n{w_m:.1f}×{h_m:.1f} м\n{area:.1f} м²",
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Настройки отображения
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('План этажа', fontsize=12)
    ax.grid(False)

def main():
    st.set_page_config(layout="wide")
    st.title('📐 Планировщик квартирографии')
    
    # Настройки в сайдбаре
    with st.sidebar:
        st.title('⚙️ Настройки')
        
        floors = st.number_input('Число этажей', min_value=1, max_value=50, value=3, step=1)
        grid_size_mm = st.number_input('Размер сетки (мм)', min_value=GRID_MIN, max_value=GRID_MAX, 
                                      value=100, step=10)
        scale_px_per_mm = st.number_input('Масштаб (px на 1 мм)', min_value=SCALE_MIN, 
                                         max_value=SCALE_MAX, value=2, step=1)
        
        st.subheader('Распределение квартир')
        percents = {}
        cols = st.columns(2)
        for i, t in enumerate(DEFAULT_TYPES):
            with cols[i % 2]:
                percents[t] = st.slider(f'{t} (%)', 0, 100, 20, key=f'p_{t}')
        
        sum_p = sum(percents.values())
        if sum_p != 100:
            st.error(f'Сумма процентов должна быть 100% (сейчас {sum_p}%)')
        
        st.subheader('Диапазоны площадей (м²)')
        area_ranges = {}
        for t in DEFAULT_TYPES:
            min_default, max_default = DEFAULT_AREAS[t]
            min_val = st.number_input(f'Мин {t}', min_value=5.0, value=min_default, 
                                    step=0.5, key=f'min_{t}')
            max_val = st.number_input(f'Макс {t}', min_value=min_val+1, value=max_default, 
                                     step=0.5, key=f'max_{t}')
            area_ranges[t] = (min_val, max_val)
    
    # Основная область - холст для рисования
    st.subheader('Нарисуйте план этажа')
    st.markdown("""
    1. Нарисуйте **периметр этажа** (первый полигон)
    2. Добавьте **МОПы** (остальные полигоны)
    3. Нажмите кнопку генерации
    """)
    
    canvas = st_canvas(
        height=600,
        width=1000,
        background_color='#f8f9fa',
        update_streamlit=True,
        drawing_mode='polygon',
        key='canvas',
        stroke_width=2,
        stroke_color='#000000',
        fill_color='rgba(0, 0, 0, 0.1)',
        grid_color='#dddddd',
        grid_spacing=int(grid_size_mm * scale_px_per_mm),
        grid_width=1
    )
    
    # Генерация квартирографии
    if st.button('🏗️ Сгенерировать квартирографию', disabled=sum_p!=100):
        data = canvas.json_data
        if not data or 'objects' not in data or len(data['objects']) == 0:
            st.error('Пожалуйста, нарисуйте хотя бы периметр этажа')
            return
        
        try:
            # Извлекаем полигоны
            objs = data['objects']
            polys = [o for o in objs if o['type'] == 'polygon']
            
            if not polys:
                st.error('Не найден периметр этажа')
                return
            
            # Первый полигон - периметр, остальные - МОПы
            perimeter = validate_polygon(geom.Polygon(polys[0]['points']))
            mops = [validate_polygon(geom.Polygon(o['points'])) for o in polys[1:]]
            
            # Рассчет общей площади в м²
            area_px = perimeter.area - sum(m.area for m in mops)
            total_area = (area_px / (scale_px_per_mm ** 2)) / 1e6  # в м²
            
            st.success(f'Общая полезная площадь: {total_area:.1f} м²')
            
            # Расчет количества квартир
            counts = calculate_apartment_counts(total_area, percents, area_ranges)
            per_floor = distribute_per_floor(counts, floors)
            
            # Отображение результатов для каждого этажа
            for i in range(floors):
                st.subheader(f'Этаж {i+1}')
                apt_list = [t for t, cnt in per_floor[i].items() for _ in range(cnt)]
                
                if not apt_list:
                    st.warning('Нет квартир для размещения на этом этаже')
                    continue
                
                placements = layout_floor(perimeter, mops, apt_list, area_ranges, scale_px_per_mm)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                draw_floor_plan(ax, perimeter, mops, placements, scale_px_per_mm, DEFAULT_TYPES)
                
                # Легенда
                handles = [Rectangle((0,0),1,1, color=COLORS[i]) for i in range(len(DEFAULT_TYPES))]
                ax.legend(handles, DEFAULT_TYPES, title='Типы квартир', 
                          loc='upper right', bbox_to_anchor=(1.15, 1))
                
                st.pyplot(fig)
                
                # Статистика по этажу
                floor_stats = {t: apt_list.count(t) for t in set(apt_list)}
                st.write(f"**Распределение:** {', '.join([f'{k}: {v}' for k, v in floor_stats.items()])}")
        
        except Exception as e:
            st.error(f'Ошибка при обработке данных: {str(e)}')

if __name__ == '__main__':
    main()
