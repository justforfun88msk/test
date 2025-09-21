# app.py

import streamlit as st
import sklearn
import warnings
import pandas as pd # ИСПРАВЛЕНО: Добавлен недостающий импорт pandas

# Импорты из нашего проекта
import ui_steps
import ui_config
import ml_core
from utils import get_session_id

warnings.filterwarnings("ignore")

# --- Начальная настройка страницы ---
st.set_page_config(page_title="Sminex ML", layout="wide", page_icon="🤖")
st.markdown(f"<style>{ui_config.APP_CSS}</style>", unsafe_allow_html=True)

# --- Инициализация состояния сессии ---
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0 # 0 - Домашняя страница

# --- Заголовок и подзаголовок ---
st.markdown(
    f"""
    <div style="display: flex; align-items: baseline; justify-content: space-between;">
        <h1 style="margin-bottom:0;">
            <b>Sminex ML</b>
            <span style="color:gray; font-size:0.4em; margin-left: 10px;">
                AutoML {ui_config.APP_VERSION}
            </span>
        </h1>
        <span style="font-size: 0.8em; color: #666;">by Charikov</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Загрузите данные → Настройте задачу → Получите лучшую модель → Сделайте прогноз → Найдите оптимальные параметры")

# --- Боковое меню (Sidebar) для навигации ---
with st.sidebar:
    st.title("Навигация")
    steps = {
        "🏠 Домашняя страница": 0,
        "📁 1. Загрузка данных": 1,
        "🎯 2. Настройка задачи": 2,
        "🤖 3. Обучение моделей": 3,
        "📊 4. Анализ модели": 4,
        "🔮 5. Прогноз": 5,
        "⚙️ 6. Калькулятор": 6,
    }
    
    # Проверка, можно ли перейти на шаг
    max_unlocked_step = 0
    if 'train_df' in st.session_state: max_unlocked_step = 1
    if 'target' in st.session_state: max_unlocked_step = 2
    if 'leaderboard' in st.session_state: max_unlocked_step = 3
    if 'fitted_pipe' in st.session_state: max_unlocked_step = 6 # Все шаги доступны

    for step_name, step_num in steps.items():
        is_disabled = step_num > max_unlocked_step and step_num > 0
        if st.button(step_name, key=f"sidebar_step_{step_num}", use_container_width=True, disabled=is_disabled):
            st.session_state.wizard_step = step_num
            st.rerun()

    st.markdown("---")
    st.caption(f"Версия: {ui_config.APP_VERSION}")
    st.caption(f"sklearn: {sklearn.__version__}")
    st.caption(f"ID сессии: {get_session_id()[:8]}")

# --- Липкая панель с текущим состоянием ---
with st.container():
    st.markdown('<div class="sticky-bar">', unsafe_allow_html=True)
    ds_shape = st.session_state.get("train_df", pd.DataFrame()).shape
    ds_info = f"{ds_shape[0]}×{ds_shape[1]}" if ds_shape[0] > 0 else "не загружен"
    
    badges = [
        f'<span class="badge">Датасет: <b>{ds_info}</b></span>',
        f'<span class="badge">Target: <b>{st.session_state.get("target", "—")}</b></span>',
        f'<span class="badge">Тип: <b>{st.session_state.get("task_type", "—")}</b></span>',
        f'<span class="badge">Модель: <b>{st.session_state.get("active_model_name", "—")}</b></span>',
    ]
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# --- Маршрутизатор (Router) для отображения нужного шага ---
current_step = st.session_state.wizard_step

if current_step == 0:
    ui_steps.render_step0_home()
elif current_step == 1:
    ui_steps.render_step1_upload()
elif current_step == 2:
    ui_steps.render_step2_setup()
elif current_step == 3:
    ui_steps.render_step3_training()
elif current_step == 4:
    ui_steps.render_step4_analysis()
elif current_step == 5:
    ui_steps.render_step5_predict()
elif current_step == 6:
    ui_steps.render_step6_calculator()
else:
    st.error("Произошла ошибка навигации. Возврат на главную страницу.")
    st.session_state.wizard_step = 0
    st.rerun()
