# app.py - Auto ML Sminex v0.25 by Charikov

import streamlit as st
import sklearn
import warnings
import pandas as pd
import os
import traceback
import logging
import sys
from datetime import datetime
import gc

import ui_steps
import ui_config
import ml_core
from utils import get_session_id, get_random_tip

# ============ ЛОГИРОВАНИЕ С РОТАЦИЕЙ ============
log_dir = os.path.join(os.path.expanduser("~"), ".streamlit", "logs_sminex")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"sminex_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ✅ УЛУЧШЕНО: Ротация логов (удаляем старые файлы)
try:
    log_files = sorted([f for f in os.listdir(log_dir) if f.startswith("sminex_ml_")])
    if len(log_files) > 10:  # Храним только последние 10 файлов
        for old_log in log_files[:-10]:
            try:
                os.remove(os.path.join(log_dir, old_log))
            except Exception:
                pass
except Exception:
    pass

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info(f"🚀 Sminex AutoML {ui_config.APP_VERSION} запущен")
logger.info(f"Логи: {log_file}")
logger.info(f"sklearn версия: {sklearn.__version__}")
logger.info(f"Параллелизм: N_JOBS={ml_core.N_JOBS}")
logger.info(f"XGB: {ml_core.XGB_AVAILABLE}, LGBM: {ml_core.LGBM_AVAILABLE}, CatBoost: {ml_core.CATBOOST_AVAILABLE}")
logger.info(f"Optuna: {ml_core.OPTUNA_AVAILABLE}")
logger.info("="*70)

warnings.filterwarnings("ignore")

# ============ STREAMLIT КОНФИГ ============
st.set_page_config(
    page_title="Auto ML Sminex v.025 by Charikov",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Применяем CSS
st.markdown(f"<style>{ui_config.APP_CSS}</style>", unsafe_allow_html=True)

# ============ SESSION STATE ИНИЦИАЛИЗАЦИЯ ============
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0
    logger.info("Инициализация session state")

if "session_id" not in st.session_state:
    st.session_state.session_id = get_session_id()
    logger.info(f"Новая сессия: {st.session_state.session_id[:12]}")

# ✅ ДОБАВЛЕНО: Отслеживание времени последней активности
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()
else:
    st.session_state.last_activity = datetime.now()

# ============ ФУНКЦИИ ============
def get_max_unlocked_step():
    """Максимальный разблокированный шаг с валидацией."""
    # 0 – Главная, 1 – Загрузка данных
    max_step = 1

    df = st.session_state.get("train_df")
    if df is None:
        # Данные не загружены — доступны только Главная и Шаг 1
        return max_step

    # Если данные есть:
    # Шаг 2 (настройка задачи) и Шаг 3 (обучение) разрешаем всегда.
    # Внутри render_step2 / render_step3 уже есть подробные проверки
    # на target, split и т.п., поэтому тут лишний блок не нужен.
    max_step = 3

    # --- Шаг 4: Анализ моделей ---
    leaderboard = st.session_state.get("leaderboard")
    if leaderboard is None or getattr(leaderboard, "empty", False):
        # Модели ещё не обучены / нет результатов — дальше не пускаем
        return max_step
    max_step = 4

    # --- Шаги 5–6: Прогнозирование и оптимизация ---
    fitted_pipe = st.session_state.get("fitted_pipe")
    if fitted_pipe is not None:
        # Как только есть финальная модель — открываем шаги 5 и 6
        max_step = 6

    return max_step

def clear_session():
    """✅ УЛУЧШЕНО: Очистить сессию с явным освобождением памяти."""
    keys_to_keep = ["wizard_step", "session_id", "last_activity"]
    
    # Явно удаляем большие объекты
    large_objects = ['train_df', 'X_train', 'X_test', 'fitted_pipe', 'leaderboard']
    for key in large_objects:
        if key in st.session_state:
            del st.session_state[key]
    
    # Удаляем остальные ключи
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            st.session_state.pop(key, None)
    
    # ✅ ДОБАВЛЕНО: Явный сбор мусора
    gc.collect()
    
    logger.info("Сессия очищена, память освобождена")

# ============ ЗАГОЛОВОК ============
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style="margin:0; color: #101820;">Auto ML Sminex</h1>
            <p style="color:#5f6368; font-size:0.95em; margin-top:6px;">v.025 · by Charikov</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div class='floating-hint'>Лаконичный AutoML без лишнего шума</div>", unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("<h3 style='margin-bottom:4px;'>Навигация</h3>", unsafe_allow_html=True)

    steps = {
        "🏠 Главная": 0,
        "📁 1. Загрузка": 1,
        "🎯 2. Настройка": 2,
        "🤖 3. Обучение": 3,
        "📊 4. Аналитика": 4,
        "🔮 5. Прогноз": 5,
        "⚙️ 6. Оптимизация": 6,
    }

    max_unlocked = get_max_unlocked_step()

    for step_name, step_num in steps.items():
        is_disabled = step_num > max_unlocked and step_num > 0
        is_current = step_num == st.session_state.wizard_step

        button_type = "primary" if is_current else "secondary"

        if st.button(
            step_name,
            key=f"sidebar_{step_num}",
            use_container_width=True,
            disabled=is_disabled,
            type=button_type if not is_disabled else "secondary"
        ):
            st.session_state.wizard_step = step_num
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Статус")
    
    if 'train_df' in st.session_state and st.session_state.train_df is not None:
        df_shape = st.session_state.train_df.shape
        st.metric("📊 Датасет", f"{df_shape[0]:,} × {df_shape[1]}")
    else:
        st.metric("📊 Датасет", "—")
    
    if 'target' in st.session_state:
        target_display = st.session_state.target[:15] + "..." if len(st.session_state.target) > 15 else st.session_state.target
        st.metric("🎯 Цель", target_display)
    else:
        st.metric("🎯 Цель", "—")
    
    if 'task_type' in st.session_state:
        task_emoji = {"binary": "🔵", "multiclass": "🌈", "regression": "📈"}
        task = st.session_state.task_type
        st.metric("🔍 Тип", f"{task_emoji.get(task, '🎯')} {task}")
    else:
        st.metric("🔍 Тип", "—")
    
    if 'active_model_name' in st.session_state:
        model_display = (
            st.session_state.active_model_name[:12] + "..."
            if len(st.session_state.active_model_name) > 12
            else st.session_state.active_model_name
        )
        st.metric("🤖 Модель", model_display)
    else:
        st.metric("🤖 Модель", "—")
    
    st.markdown("---")
    
    st.markdown("---")
    st.markdown("<p style='color:#6e6e73;'>Минимум шума — максимум данных.</p>", unsafe_allow_html=True)

    if st.button("Очистить проект", use_container_width=True):
        clear_session()
        st.rerun()

# ============ STICKY BAR ============
with st.container():
    st.markdown('<div class="sticky-bar">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'train_df' in st.session_state and st.session_state.train_df is not None:
            df_shape = st.session_state.train_df.shape
            st.metric("📊 Датасет", f"{df_shape[0]:,}×{df_shape[1]}")
        else:
            st.metric("📊 Датасет", "—")
    
    with col2:
        target = st.session_state.get("target", "—")
        display_target = target[:15] + "..." if isinstance(target, str) and len(target) > 15 else target
        st.metric("🎯 Target", display_target if target else "—")
    
    with col3:
        task = st.session_state.get("task_type", "—")
        st.metric("🔍 Тип", task if task else "—")
    
    with col4:
        model = st.session_state.get("active_model_name", "—")
        display_model = model[:12] + "..." if isinstance(model, str) and len(model) > 12 else model
        st.metric("🤖 Модель", display_model if model else "—")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============ ROUTER С УЛУЧШЕННОЙ ОБРАБОТКОЙ ОШИБОК ============
try:
    current_step = st.session_state.wizard_step
    logger.info(f"Рендеринг шага: {current_step}")
    
    # ✅ ДОБАВЛЕНО: Валидация состояния перед рендерингом
    max_unlocked = get_max_unlocked_step()
    if current_step > max_unlocked and current_step > 0:
        st.warning(f"⚠️ Шаг {current_step} еще не доступен. Завершите предыдущие шаги.")
        st.session_state.wizard_step = max_unlocked
        if st.button("🔄 Вернуться к последнему доступному шагу"):
            st.rerun()
    else:
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
            st.error("❌ Ошибка навигации: неизвестный шаг")
            if st.button("🔄 На главную"):
                st.session_state.wizard_step = 0
                st.rerun()

except Exception as e:
    logger.error(f"КРИТИЧЕСКАЯ ОШИБКА на шаге {st.session_state.wizard_step}: {str(e)}", exc_info=True)
    
    st.error(f"❌ Критическая ошибка: {type(e).__name__}")
    st.error(f"Сообщение: {str(e)[:500]}")
    
    # ✅ УЛУЧШЕНО: Более информативные сообщения об ошибках
    with st.expander("📋 Детали ошибки (для разработчиков)"):
        st.code(traceback.format_exc())
        
        st.markdown("**Состояние сессии:**")
        st.json({
            "wizard_step": st.session_state.wizard_step,
            "has_train_df": 'train_df' in st.session_state,
            "has_target": 'target' in st.session_state,
            "has_split": 'X_train' in st.session_state,
            "has_leaderboard": 'leaderboard' in st.session_state,
            "has_fitted_pipe": 'fitted_pipe' in st.session_state,
        })
    
    st.markdown("---")
    st.markdown("### 🔧 Возможные решения:")
    st.markdown("""
    1. **Попробуйте вернуться на главную** и начать заново
    2. **Очистите данные проекта** (кнопка в боковом меню)
    3. **Проверьте логи** для подробной информации
    4. **Уменьшите размер датасета** если файл слишком большой
    5. **Попробуйте другой файл** для проверки
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 На главную", use_container_width=True):
            st.session_state.wizard_step = 0
            st.rerun()
    with col2:
        if st.button("🗑️ Очистить данные", use_container_width=True):
            clear_session()
            st.rerun()
    with col3:
        st.caption(f"📁 Логи: {log_dir}")

# ============ FOOTER ============
st.divider()
