# -*- coding: utf-8 -*-
"""
📐 Квартирография Architect Edition (с учётом МОПов и сеткой 10 см)
Streamlit-программа для интерактивной разметки этажей с визуальной сеткой, исключением МОПов и генерацией квартир по %.
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import random
from PIL import Image, ImageDraw
import base64

# ==== Константы ====
SCALE_CM_PER_CELL = 10   # 10 см = 1 клетка
CELL_SIZE_PX = 20        # размер клетки на экране в пикселях

# ==== Конфигурация страницы ====
st.set_page_config(page_title="Квартирография Architect Edition", layout="wide")
st.title("📐 Квартирография — Architect Edition")

# ==== Ввод параметров здания ====
st.sidebar.header("🏢 Параметры здания")
floors = st.sidebar.number_input("Этажей в доме", min_value=1, value=23)
floor_width_m = st.sidebar.number_input("Ширина этажа (м)", min_value=1.0, value=10.5)
floor_length_m = st.sidebar.number_input("Длина этажа (м)", min_value=1.0, value=72.0)

# ==== Расчёт канвы ====
grid_cols = int((floor_length_m * 100) / SCALE_CM_PER_CELL)
grid_rows = int((floor_width_m * 100) / SCALE_CM_PER_CELL)
canvas_width = grid_cols * CELL_SIZE_PX
canvas_height = grid_rows * CELL_SIZE_PX

# ==== Генерация изображения сетки ====
def generate_grid_image(width_px, height_px, cell_size):
    img = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(img)
    for x in range(0, width_px, cell_size):
        draw.line([(x, 0), (x, height_px)], fill="#DDD")
    for y in range(0, height_px, cell_size):
        draw.line([(0, y), (width_px, y)], fill="#DDD")
    return img

grid_img = generate_grid_image(canvas_width, canvas_height, CELL_SIZE_PX)
buf = BytesIO()
grid_img.save(buf, format="PNG")
grid_img_b64 = base64.b64encode(buf.getvalue()).decode()

# ==== Параметры квартир ====
st.sidebar.header("🏘 Распределение типов квартир (%)")
percent_distribution = {
    "Студия": st.sidebar.slider("Студии", 0.0, 100.0, 1.4),
    "1-комн": st.sidebar.slider("1-комн", 0.0, 100.0, 34.6),
    "2-комн": st.sidebar.slider("2-комн", 0.0, 100.0, 41.8),
    "3-комн": st.sidebar.slider("3-комн", 0.0, 100.0, 21.8),
}

# ==== Полотно для рисования ====
st.subheader("1️⃣ Нарисуйте МОПы (нежилые зоны)")
st.caption("Полигональные области, которые будут исключены из планировки квартир")

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",  # красные прозрачные МОПы
    stroke_width=2,
    stroke_color="#FF0000",
    background_image=f"data:image/png;base64,{grid_img_b64}",
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="polygon",
    key="canvas_mop"
)

# ==== Обработка МОПов ====
def extract_mop_polygons(data_json):
    polygons = []
    if not data_json or "objects" not in data_json:
        return []
    for obj in data_json["objects"]:
        if obj.get("type") == "polygon":
            path = obj.get("path")
            if path:
                poly = Polygon([(x, y) for x, y in path])
                if poly.is_valid:
                    polygons.append(poly)
    return polygons

mop_polys = extract_mop_polygons(canvas_result.json_data)
usable_poly = Polygon([(0, 0), (canvas_width, 0), (canvas_width, canvas_height), (0, canvas_height)])
if mop_polys:
    mop_union = unary_union(mop_polys)
    usable_poly = usable_poly.difference(mop_union)

# ==== Генерация квартир ====
st.subheader("2️⃣ Генерация квартир")
st.caption("Алгоритмическая генерация квартир с учётом заданных процентов и ограничений")

apt_constraints = {
    "Студия": {"min": 28, "max": 31, "width": 7.2},
    "1-комн": {"min": 32, "max": 46, "width": 7.2},
    "2-комн": {"min": 47, "max": 65, "width": 7.2},
    "3-комн": {"min": 66, "max": 99, "width": 7.2},
}

TOTAL_AREA_M2 = (floor_width_m * floor_length_m - (len(mop_polys) * 15)) * floors  # оценка МОПов в 15 м2 каждый

@st.cache_data
def generate_apartments():
    apartments = []
    for apt_type, percent in percent_distribution.items():
        target_area = TOTAL_AREA_M2 * (percent / 100)
        used = 0
        while used < target_area:
            s = random.randint(apt_constraints[apt_type]["min"], apt_constraints[apt_type]["max"])
            w = apt_constraints[apt_type]["width"]
            l = round(s / w, 2)
            if used + s > target_area:
                break
            apartments.append({"Тип": apt_type, "Площадь": s, "Ширина": w, "Длина": l})
            used += s
    return apartments

if st.button("🚀 Сгенерировать квартиры"):
    plan = generate_apartments()
    df = pd.DataFrame(plan)
    st.dataframe(df)

    st.subheader("📊 Распределение по типам")
    fig, ax = plt.subplots()
    df.groupby("Тип")["Площадь"].sum().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    out = BytesIO()
    df.to_excel(out, index=False)
    st.download_button("📥 Скачать Excel", data=out.getvalue(), file_name="plan.xlsx")
