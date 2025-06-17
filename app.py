import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import matplotlib.pyplot as plt
import pandas as pd
import math
from PIL import Image, ImageDraw
import io

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
types = ['Студия', '1С', '2С', '3С', '4С']
percentages = {}
st.sidebar.markdown("### Сумма процентов")
total_percent = 0
cols = st.sidebar.columns(2)
for i, t in enumerate(types):
    with cols[i % 2]:
        percentages[t] = st.slider(f"% {t}", 0, 100, 100 // len(types), key=f"pct_{t}")
        total_percent += percentages[t]
color = "green" if abs(total_percent - 100) <= 0.01 else "red"
st.sidebar.markdown(f"<p style='color:{color};'>Сумма: {total_percent:.1f}%</p>", unsafe_allow_html=True)
if abs(total_percent - 100) > 0.01:
    st.sidebar.error(f"Сумма % должна быть 100% (сейчас {total_percent:.1f}%)")
    st.stop()

st.sidebar.subheader("Диапазоны площадей (м²)")
areas = {}
for t in types:
    mn, mx = st.sidebar.slider(f"{t}", 1.0, 200.0, (20.0, 50.0), key=f"area_{t}")
    areas[t] = (mn, mx)

# — Sidebar: Проект —
st.sidebar.header("💾 Проект")
proj_name = st.sidebar.text_input("Имя проекта (JSON)", "plan.json")

# ==========================
#   СОЗДАНИЕ СЕТКИ
# ==========================

def create_grid_image(width, height, grid_size_px):
    image = Image.new('RGB', (width, height), '#F0F0F0')
    draw = ImageDraw.Draw(image)
    for x in range(0, width, grid_size_px):
        draw.line([(x, 0), (x, height)], fill='gray', width=1)
    for y in range(0, height, grid_size_px):
        draw.line([(0, y), (width, y)], fill='gray', width=1)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer

grid_size_px = int(grid_mm / scale)
canvas_width, canvas_height = 800, 600
grid_image = create_grid_image(canvas_width, canvas_height, grid_size_px)

# ==========================
#   CANVAS: ЧЕРЧЕНИЕ ПЛАНА
# ==========================

st.subheader("1️⃣ Нарисуйте план этажа")
st.markdown("Первый полигон — внешний контур; остальные — зоны МОП. Углы корректируются до 90° после рисования.")
canvas_data = st_canvas(
    stroke_width=2,
    stroke_color='#000',
    fill_color='rgba(255,165,0,0.3)',
    background_color='#F0F0F0',
    background_image=Image.open(grid_image),
    drawing_mode='polygon',
    key='canvas2',
    width=canvas_width,
    height=canvas_height,
    initial_drawing=None
)

# ==========================
#   ПОЛИГОН ЭТАЖА + МОП
# ==========================

def snap(pt, prev_pt=None):
    x, y = pt
    if not show_snap:
        return (x, y)
    g = grid_mm / scale
    x, y = round(x / g) * g, round(y / g) * g
    if prev_pt:
        px, py = prev_pt
        dx, dy = x - px, y - py
        if abs(dx) > abs(dy):
            y = py  # Горизонтальная линия
        else:
            x = px  # Вертикальная линия
    return (x, y)

def enforce_90_degrees(pts):
    corrected = [pts[0]]
    for i in range(1, len(pts)):
        prev_pt = corrected[-1]
        curr_pt = snap(pts[i], prev_pt)
        corrected.append(curr_pt)
    return corrected

raw = canvas_data.json_data or {}
objs = raw.get('objects', [])
polys = []
for o in objs:
    if o.get('type') == 'polygon':
        pts = [(p['x'], p['y']) for p in o['points']]
        if len(pts) >= 3:
            pts = enforce_90_degrees(pts)
            try:
                poly = Polygon(pts)
                if not poly.is_valid:
                    st.error(f"Полигон {len(polys)+1} некорректен (например, самопересекается).")
                    st.stop()
                polys.append(poly)
            except Exception as e:
                st.error(f"Ошибка при создании полигона: {str(e)}")
                st.stop()
if not polys:
    st.error("Нарисуйте внешний контур!")
    st.stop()

with st.spinner("Обработка полигонов..."):
    floor = polys[0]
    for hole in polys[1:]:
        try:
            if hole.is_valid and floor.is_valid:
                floor = floor.difference(hole)
            else:
                st.error("Один из полигонов (внешний контур или МОП) некорректен.")
                st.stop()
        except Exception as e:
            st.error(f"Ошибка при вычитании зон МОП: {str(e)}")
            st.stop()

# Отображение размеров этажа на холсте
minx, miny, maxx, maxy = floor.bounds
w_mm = (maxx - minx) * scale
h_mm = (maxy - miny) * scale
area_m2 = floor.area * scale**2 / 1e6
st.info(f"Контур: {w_mm:.0f}×{h_mm:.0f} мм, площадь {area_m2:.2f} м²")

# Отрисовка контура с размерами
fig, ax = plt.subplots(figsize=(8, 6))
x, y = floor.exterior.xy
ax.fill([xi * scale for xi in x], [yi * scale for yi in y], color='rgba(255,165,0,0.3)', edgecolor='black')
cx, cy = floor.representative_point().xy
ax.text(cx * scale, cy * scale, f"{w_mm:.0f}×{h_mm:.0f} мм\n{area_m2:.2f} м²",
        ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
ax.set_aspect('equal')
ax.axis('off')
st.pyplot(fig)
plt.close(fig)

# ==========================
#   ФУНКЦИИ НАРЕЗКИ
# ==========================

def split_poly(poly, target_px2, tol=0.05):
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
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
    projs = [ux * x + uy * y for x, y in poly.exterior.coords]
    low, high = min(projs), max(projs)

    def make_cut(offset):
        mx, my = ux * offset, uy * offset
        vx, vy = -uy, ux
        minx, miny, maxx, maxy = poly.bounds
        diag = math.hypot(maxx - minx, maxy - miny) * 2
        return LineString([(mx + vx * diag, my + vy * diag), (mx - vx * diag, my - vy * diag)])

    for _ in range(30):
        mid = (low + high) / 2
        parts = split(poly, make_cut(mid))
        if not parts.geoms:
            low = mid
            continue
        parts = list(parts.geoms)
        if len(parts) < 2:
            low = mid
            continue
        if len(parts) > 2:
            st.warning("Разделение полигона дало более двух частей, используется только первая.")
            parts = parts[:2]
        areas_list = []
        for part in parts:
            cx, cy = part.representative_point().coords[0]
            proj = ux * cx + uy * cy
            areas_list.append((proj, part))
        smaller = min(areas_list, key=lambda x: x[0])[1]
        a = smaller.area
        if a > target_px2 * (1 + tol):
            high = mid
        elif a < target_px2 * (1 - tol):
            low = mid
        else:
            rem = [p for p in parts if not p.equals(smaller)]
            return smaller, (rem[0] if rem else None)

    parts = split(poly, make_cut((low + high) / 2))
    if not parts.geoms:
        st.error("Не удалось разделить полигон.")
        st.stop()
    parts = sorted(list(parts.geoms), key=lambda p: p.area)
    return parts[0], (parts[1] if len(parts) > 1 else None)

# ==========================
#   РАСПРЕДЕЛЕНИЕ КВАРТИР
# ==========================

st.subheader("2️⃣ Подбор квартирографии по всему зданию")
if st.button("Сгенерировать квартирографию"):
    total_area = area_m2 * floors
    avg_area = {t: (areas[t][0] + areas[t][1]) / 2 for t in types}
    counts = {t: max(1, int(round(total_area * percentages[t] / 100 / avg_area[t]))) for t in types}
    per_floor = {i: {} for i in range(floors)}
    for t, c in counts.items():
        q, r = divmod(c, floors)
        for i in range(floors):
            per_floor[i][t] = q + (1 if i < r else 0)

    # Генерация схем для всех этажей
    st.subheader("3️⃣ Схемы этажей")
    floor_placements = {}
    for fl in range(1, floors + 1):
        targets = []
        for t, n in per_floor[fl - 1].items():
            if n > 0:
                tot_t = total_area * percentages[t] / 100
                avg_t = tot_t / counts[t]
                px2 = avg_t * 1e6 / scale**2
                targets += [(t, px2)] * n

        # Разметка этажа
        avail = [floor]
        placements = []
        for t, px2 in targets:
            avail.sort(key=lambda p: p.area, reverse=True)
            if not avail:
                st.warning(f"Этаж {fl}: Недостаточно пространства для размещения всех квартир.")
                break
            poly = avail.pop(0)
            apt, rem = split_poly(poly, px2)
            placements.append((t, apt))
            if rem and rem.area > 0.01 * px2:
                avail.append(rem)

        floor_placements[fl] = placements

        # Отрисовка этажа
        st.markdown(f"#### Этаж {fl}")
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = {'Студия': '#FFC107', '1С': '#8BC34A', '2С': '#03A9F4', '3С': '#E91E63', '4С': '#9C27B0'}
        for t, poly in placements:
            x, y = poly.exterior.xy
            ax.fill([xi * scale for xi in x], [yi * scale for yi in y], color=cmap[t], alpha=0.7, edgecolor='black')
            # Аннотация с размерами и площадью
            minx, miny, maxx, maxy = poly.bounds
            w_mm = (maxx - minx) * scale
            h_mm = (maxy - miny) * scale
            area_m2 = poly.area * scale**2 / 1e6
            cx, cy = poly.representative_point().xy
            ax.text(cx[0] * scale, cy[0] * scale, f"{t}\n{w_mm:.0f}×{h_mm:.0f} мм\n{area_m2:.2f} м²",
                    ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
        ax.set_aspect('equal')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

    # Отчет
    df = pd.DataFrame([{'Этаж': i + 1, 'Тип': t, 'Кол-во': per_floor[i][t]} for i in range(floors) for t in types])
    st.subheader("4️⃣ Сводный отчет по этажам")
    st.dataframe(df)
    st.sidebar.download_button("📥 Скачать CSV", df.to_csv(index=False), file_name='report.csv', mime='text/csv')
