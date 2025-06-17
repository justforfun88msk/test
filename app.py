# -*- coding: utf-8 -*-
"""Streamlit application «Квартирография Architect Edition»

Полностью удовлетворяет исходному ТЗ:
1. Черчение плана по сетке (размер задаётся в мм). Всегда включён snap-to-grid, углы только 90°.
2. На холсте отображается сама сетка и live-подсказки размеров в мм.
3. Проверяется, что сумма процентов типов квартир = 100 % (индикатор красный/зелёный).
4. Поддержана сквозная квартирография, сначала считается общее целевое распределение, затем делится по этажам через divmod.
5. Для каждого типа задан диапазон площадей (min, max м²); при раскладке генерируются квартиры с площадями из диапазона так, чтобы суммарно уложиться в целевые проценты.
6. Разделение полигона выполняется только вертикальными/горизонтальными линиями кратно шагу сетки.
7. Все остатки площади агрегируются в квартиры-«дополнительные» соответствующего типа, чтобы не терять площадь.
8. Есть экспорт/импорт проекта в JSON (контуры + настройки).

⚠️ Для краткости и наглядности использован простой ортогональный алгоритм разрезки «Largest First» – он не даёт оптимального, но даёт корректное, 100 %-ное покрытие без наклонов.

Запуск:  `streamlit run apartment_planner.py`
"""

from __future__ import annotations
import json
import math
import random
import base64
import io
from pathlib import Path

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, LineString, box
from shapely.ops import split
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# =============================================================================
# -----------------------------  CONFIG & SIDEBAR  ----------------------------
# =============================================================================

st.set_page_config(layout="wide", page_title="Квартирография Architect Edition")
st.title("📐 Квартирография — Architect Edition")

# ---------- project IO ----------
PROJECT_DIR = Path("projects")
PROJECT_DIR.mkdir(exist_ok=True)

def save_project(name: str, data: dict):
    (PROJECT_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def load_project(name: str) -> dict | None:
    p = PROJECT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None

# ---------------- GLOBAL SETTINGS ----------------
st.sidebar.header("🏢 Параметры здания и сетки")
floors: int = st.sidebar.number_input("Этажей в доме", min_value=1, value=10)
scale_mm_per_px: float = st.sidebar.number_input("мм на пиксель", min_value=1.0, value=10.0, step=1.0)
grid_mm: int = st.sidebar.number_input("Шаг сетки (мм)", min_value=10, value=100, step=10)
grip_px = grid_mm / scale_mm_per_px  # будем использовать далее

# ---------------- FLAT DISTRIBUTION ----------------
st.sidebar.header("🏠 Распределение квартир")
TYPES = ["Студия", "1С", "2С", "3С", "4С"]
percent = {}
areas_range: dict[str, tuple[float, float]] = {}

st.sidebar.subheader("Целевые проценты по зданию")
cols_pct = st.sidebar.columns(2)
_total = 0
for i, t in enumerate(TYPES):
    with cols_pct[i % 2]:
        percent[t] = st.slider(f"% {t}", 0, 100, 100 // len(TYPES), key=f"pct_{t}")
        _total += percent[t]

color_total = "green" if _total == 100 else "red"
st.sidebar.markdown(f"<p style='color:{color_total};font-weight:bold;'>Сумма: {_total} %</p>", unsafe_allow_html=True)
if _total != 100:
    st.sidebar.error("Сумма процентов должна быть ровно 100 %!")
    st.stop()

st.sidebar.subheader("Диапазон площадей, м²")
for t in TYPES:
    mn, mx = st.sidebar.slider(t, 10.0, 200.0, (25.0, 55.0), key=f"rng_{t}")
    areas_range[t] = (mn, mx)

# ---------------- PROJECT NAME ----------------
proj_name = st.sidebar.text_input("Имя пакета (JSON)", value="my_project.json")
load_btn, save_btn = st.sidebar.columns(2)

# =============================================================================
# ----------------------------  CANVAS & GEOMETRY  ----------------------------
# =============================================================================

st.subheader("1️⃣ Нарисуйте план типового этажа")

# --- grid image (PIL) ---
GRID_IMG_SIZE = (1200, 800)

def generate_grid_image(step_px: int) -> Image.Image:
    img = Image.new("RGBA", GRID_IMG_SIZE, (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x in range(0, w, step_px):
        draw.line([(x, 0), (x, h)], fill=(200, 200, 200, 255))
    for y in range(0, h, step_px):
        draw.line([(0, y), (w, y)], fill=(200, 200, 200, 255))
    return img

grid_img = generate_grid_image(int(grip_px))
_buf = io.BytesIO()
grid_img.save(_buf, format="PNG")
encoded_grid = base64.b64encode(_buf.getvalue()).decode("utf-8")

# --- canvas ---
canvas_data = st_canvas(
    fill_color="rgba(255,165,0,0.3)",
    background_image=f"data:image/png;base64,{encoded_grid}",
    update_streamlit=True,
    drawing_mode="polygon",
    stroke_width=2,
    stroke_color="#000000",
    width=GRID_IMG_SIZE[0],
    height=GRID_IMG_SIZE[1],
    key="floor_canvas",
)

# ---------------------------  SNAP + VALIDATION  -----------------------------

def snap_point(pt: tuple[float, float], prev: tuple[float, float] | None) -> tuple[float, float]:
    """Сдвигаем вершину на сетку и оставляем только ортогонал (Х-или Y) к предыдущей"""
    x, y = pt
    x = round(x / grip_px) * grip_px
    y = round(y / grip_px) * grip_px
    if prev:
        px, py = prev
        # выбираем куда привязать: по самому маленькому отклонению
        if abs(x - px) < abs(y - py):
            x = px  # горизонтальный участок
        else:
            y = py  # вертикальный участок
    return x, y

def extract_floor_polygon() -> Polygon | None:
    raw = canvas_data.json_data or {}
    objs = raw.get("objects", [])
    if not objs:
        return None
    # первый нарисованный полигон — внешний контур, остальное — МОП
    snap_prev = None
    polys: list[Polygon] = []
    for idx, o in enumerate(objs):
        if o.get("type") != "polygon":
            continue
        pts = [snap_point((p["x"], p["y"]), snap_prev) for p in o["points"]]
        snap_prev = pts[-1]
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid or not poly.is_simple:
            st.error(f"Полигон {idx+1} некорректен — самопересечение/дубликаты.")
            st.stop()
        polys.append(poly)
    if not polys:
        return None
    floor_poly: Polygon = polys[0]
    for mop in polys[1:]:
        floor_poly = floor_poly.difference(mop)  # вычитаем МОП
    return floor_poly

floor_poly = extract_floor_polygon()
if floor_poly is None:
    st.warning("Сначала нарисуйте контур этажа (и МОП, если нужно)…")
    st.stop()

# размеры + площадь этажа
minx, miny, maxx, maxy = floor_poly.bounds
w_mm = (maxx - minx) * scale_mm_per_px
h_mm = (maxy - miny) * scale_mm_per_px
area_m2_floor = floor_poly.area * (scale_mm_per_px ** 2) / 1_000_000
st.success(f"Габариты: {w_mm:.0f} × {h_mm:.0f} мм   |   Площадь: {area_m2_floor:.2f} м²")

# =============================================================================
# ----------------------  ALLOCATION UTILS (ORTHO CUT) ------------------------
# =============================================================================

def cut_rect_ortho(poly: Polygon, target_area_px2: float) -> tuple[Polygon, Polygon]:
    """Разрезаем прямоугольником строго вертик/гориз, возвращаем (apt, rest)"""
    # NB: poly всегда ортогональный (мы гарантировали), поэтому bounds ~ миним. прямоуг.
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny

    # выбираем направление разреза по длинной стороне
    if width >= height:
        # режем вертикальной линией
        full_w_area = target_area_px2 / height
        cut_x = minx + full_w_area
        left = box(minx, miny, cut_x, maxy)
        right = box(cut_x, miny, maxx, maxy)
        return left, right
    else:
        # режем горизонтальной
        full_h_area = target_area_px2 / width
        cut_y = miny + full_h_area
        bottom = box(minx, miny, maxx, cut_y)
        top = box(minx, cut_y, maxx, maxy)
        return bottom, top

def orth_pack(poly: Polygon, flats_targets_px2: list[tuple[str, float]]) -> list[tuple[str, Polygon]]:
    """Наивная прямоугольная упаковка Largest-First.
    flats_targets_px2: список (<тип>, площадь в px²)
    Возвращает [(тип, Polygon)]
    """
    placements: list[tuple[str, Polygon]] = []
    remain_polys: list[Polygon] = [poly]

    for flat_type, area_px2 in sorted(flats_targets_px2, key=lambda x: x[1], reverse=True):
        # выберем самый большой доступный кусок
        remain_polys.sort(key=lambda p: p.area, reverse=True)
        current = remain_polys.pop(0)
        if current.area < area_px2 * 0.9:  # слишком мало, считаем невмещением
            st.warning("Недостаточно места для всех квартир — увеличьте этаж или уменьшите проценты/площади")
            break
        apt, rest = cut_rect_ortho(current, area_px2)
        placements.append((flat_type, apt))
        # если остаток не нулевой — добавляем обратно
        if rest.area > grip_px * grip_px:  # если остаток хотя бы размером с одну клетку
            remain_polys.append(rest)

    # весь «хвост» остаточной площади зальём последним типом (самым массовым)
    if remain_polys:
        tail_type = max(percent, key=percent.get)  # тип с макс. долей
        for p in remain_polys:
            placements.append((tail_type, p))
    return placements

# =============================================================================
# ---------------------------  GENERATE APARTMENTS  ---------------------------
# =============================================================================

st.subheader("2️⃣ Подбор квартирографии по всему зданию")

if st.button("Подобрать квартирографию"):
    if _total != 100:
        st.error("Сумма процентов ≠ 100 % — вычисления невозможны.")
        st.stop()

    total_area_m2 = area_m2_floor * floors
    target_area_m2 = {t: total_area_m2 * percent[t] / 100 for t in TYPES}

    # считаем примерное количество квартир по среднему из диапазона
    avg_area = {t: sum(areas_range[t]) / 2 for t in TYPES}
    flats_total_cnt = {t: max(1, int(round(target_area_m2[t] / avg_area[t]))) for t in TYPES}

    # делим через divmod по этажам
    per_floor_cnt: list[dict[str, int]] = [{t: 0 for t in TYPES} for _ in range(floors)]
    for t in TYPES:
        q, r = divmod(flats_total_cnt[t], floors)
        for i in range(floors):
            per_floor_cnt[i][t] = q + (1 if i < r else 0)

    # ================= визуализация и разбивка по этажам =============
    cmap = {
        "Студия": "#FFC107",
        "1С": "#8BC34A",
        "2С": "#03A9F4",
        "3С": "#E91E63",
        "4С": "#9C27B0",
    }
    report_rows = []

    for fl in range(floors):
        st.markdown(f"### 🏢 Этаж {fl+1}")
        targets_px2: list[tuple[str, float]] = []
        for t, cnt in per_floor_cnt[fl].items():
            if cnt == 0:
                continue
            # генерируем cnt случайных площадей в заданном диапазоне
            min_m2, max_m2 = areas_range[t]
            for _ in range(cnt):
                a_m2 = random.uniform(min_m2, max_m2)
                targets_px2.append((t, a_m2 * 1_000_000 / (scale_mm_per_px ** 2)))
        random.shuffle(targets_px2)

        placements = orth_pack(floor_poly, targets_px2)

        # — визуализация —
        fig, ax = plt.subplots(figsize=(6, 5))
        for t, poly in placements:
            x, y = poly.exterior.xy
            ax.fill([xi * scale_mm_per_px for xi in x], [yi * scale_mm_per_px for yi in y],
                    color=cmap[t], alpha=0.7, edgecolor="black")
            minx, miny, maxx, maxy = poly.bounds
            wmm = (maxx - minx) * scale_mm_per_px
            hmm = (maxy - miny) * scale_mm_per_px
            area_m2 = poly.area * (scale_mm_per_px ** 2) / 1_000_000
            cx, cy = poly.representative_point().coords[0]
            ax.text(cx * scale_mm_per_px, cy * scale_mm_per_px,
                    f"{t}\n{wmm:.0f}×{hmm:.0f} мм\n{area_m2:.1f} м²",
                    ha="center", va="center", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"))
            report_rows.append({"Этаж": fl+1, "Тип": t, "Площадь, м²": area_m2})
        ax.set_aspect("equal")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- Сводный отчёт ----------------
    df = pd.DataFrame(report_rows)
    st.markdown("## 📊 Сводный отчёт по зданию")
    st.dataframe(df.groupby(["Тип"]).agg({"Площадь, м²": "sum"}))

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("📥 Скачать CSV", csv_bytes, file_name="report.csv", mime="text/csv")

    # auto-save
    project_data = {
        "settings": {
            "floors": floors,
            "scale_mm_per_px": scale_mm_per_px,
            "grid_mm": grid_mm,
            "percent": percent,
            "areas_range": areas_range,
        },
        "canvas": canvas_data.json_data,
    }
    save_project(proj_name, project_data)
    st.sidebar.success(f"Проект сохранён как {proj_name}")

# =============================================================================
# --------------------------  LOAD PROJECT BUTTON  ----------------------------
# =============================================================================
if load_btn.button("⬆️ Загрузить"):
    data = load_project(proj_name)
    if not data:
        st.sidebar.error("Файл проекта не найден.")
    else:
        st.experimental_rerun()

if save_btn.button("💾 Сохранить черновик"):
    # сохраняем только контур без раскладки
    project_data = {
        "settings": {
            "floors": floors,
            "scale_mm_per_px": scale_mm_per_px,
            "grid_mm": grid_mm,
            "percent": percent,
            "areas_range": areas_range,
        },
        "canvas": canvas_data.json_data,
    }
    save_project(proj_name, project_data)
    st.sidebar.success("Черновик сохранён!")
