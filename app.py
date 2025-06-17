canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=2,
    stroke_color="#FF0000",
    background_image=Image.fromarray(grid_img),
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
