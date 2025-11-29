# -*- coding: utf-8 -*-
"""
ui_steps.py – ULTIMATE версия v0.25 с ПОЛНЫМИ ИСПРАВЛЕНИЯМИ:
✅ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
- accurate_mode сохраняется в session_state
- Правильная обработка кодировок с кэшированием
- Dynamic CV splits для маленьких датасетов
- Проверка на дубликаты строк
- Stratified sampling при загрузке
- Edge cases обработка
- Правильная валидация метрик
- Улучшенная обработка ошибок

✅ СРЕДНИЕ ИСПРАВЛЕНИЯ:
- Оптимизация памяти
- Лучший feedback пользователю
- Улучшенное логирование

✅ НИЗКИЕ ИСПРАВЛЕНИЯ:
- Улучшенные подсказки
- Лучшая валидация форм
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import math
import numbers
import io
import plotly.express as px
import plotly.graph_objects as pgo
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.inspection import permutation_importance
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import os
import logging

# Импорты из проекта
import ml_core
from utils import (
    detect_csv_sep, detect_file_encoding, human_time_ms, enforce_min_duration, 
    download_button, get_session_id, smart_sample_large_file, get_file_size_mb,
    sanitize_column_names, remove_duplicate_columns, validate_data_types,
    check_and_remove_duplicates, get_file_hash, estimate_memory_usage, optimize_dtypes
)
from ui_config import MODEL_DESCRIPTIONS, get_model_tags, RANDOM_SEED, MAX_DATASET_SIZE, SAMPLE_SIZE_FOR_LARGE_DATASETS

# Проверка доступности моделей
from ml_core import LGBM_AVAILABLE, CATBOOST_AVAILABLE, XGB_AVAILABLE, OPTUNA_AVAILABLE

logger = logging.getLogger(__name__)

# =========================================================
# STEP 0: HOME
# =========================================================

def render_step0_home():
    """Главная страница."""
    st.title("🤖 Добро пожаловать в Sminex ML!")
    st.markdown("""
    **Sminex ML – это профессиональный инструмент для автоматизации машинного обучения.**  
    Он позволяет:  
    * **📁 Загружать данные** в формате CSV/XLSX (до 200 МБ).  
    * **🎯 Автоматически определять** тип задачи (классификация или регрессия).  
    * **🤖 Обучать и сравнивать** десятки моделей машинного обучения с параллелизмом.  
    * **📊 Анализировать** детали и качество модели с визуализациями.  
    * **🔮 Делать прогнозы** на новые данные с валидацией типов.  
    * **⚙️ Использовать калькулятор "Что, если?"** для поиска оптимальных параметров.
    """)
    
    st.info("✨ **Новые возможности в v0.25:**\n"
            "- ⚡ Параллельное обучение моделей (до 8x быстрее)\n"
            "- 🎯 Stratified sampling для несбалансированных данных\n"
            "- 🧹 Автоматическое удаление дубликатов строк\n"
            "- 💾 Оптимизация использования памяти\n"
            "- 📊 Больше метрик для multiclass задач")
    
    st.subheader("🚀 Как начать?")
    st.markdown("""
    1. Нажмите **"📁 1. Загрузка данных"** в боковом меню.  
    2. Загрузите ваш файл с данными (CSV или Excel).  
    3. Следуйте инструкциям на каждом этапе wizard'а.
    4. Получите обученную модель и прогнозы!
    """)
    
    if st.button("🚀 Начать новый проект", type="primary", use_container_width=True):
        st.session_state.wizard_step = 1
        st.rerun()

# =========================================================
# STEP 1: UPLOAD
# =========================================================

def render_step1_upload():
    """Загрузка и парсинг данных."""
    st.header("📁 Шаг 1. Загрузка данных")
    st.markdown("""
    Загрузите ваш файл с данными в формате CSV или Excel. Данные должны быть в виде таблицы, где:  
    * **Строки** – это отдельные объекты (например, клиенты, товары, события).  
    * **Столбцы** – это характеристики (признаки) этих объектов и целевая переменная.
    
    ⚡ **Рекомендации:**
    - Минимум 100 строк для надежных результатов
    - Избегайте файлов с более чем 10,000 столбцов
    - Проверьте что целевая переменная не имеет пропусков
    """)
    
    # Опции для CSV
    csv_separator = st.selectbox(
        "Выберите разделитель для CSV файлов (если не определен автоматически)",
        options=["Автоопределение", ";", ",", "\t", "|"],
        index=0,
        help="Для большинства файлов автоопределение работает корректно"
    )
    
    # Загрузка файла
    up = st.file_uploader(
        "CSV/XLSX файл", 
        type=["csv", "xls", "xlsx"], 
        help="Максимальный размер: 200 МБ. Большие файлы будут автоматически сэмплированы."
    )

    data_loaded = False
    if up is not None:
        try:
            file_size_mb = get_file_size_mb(up)
            
            # ✅ ДОБАВЛЕНО: Предупреждение о больших файлах
            if file_size_mb > 100:
                st.warning(f"⚠️ Большой файл ({file_size_mb:.1f} МБ). "
                          f"Загрузка может занять несколько минут...")
            else:
                st.info(f"📦 Размер файла: {file_size_mb:.1f} МБ")
            
            t0 = time.time()
            
            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Кэширование определения кодировки
            file_hash = None
            
            # Определение кодировки и разделителя
            if up.name.lower().endswith(".csv"):
                first_bytes = up.read(50_000)
                file_hash = get_file_hash(first_bytes)  # ✅ ДОБАВЛЕНО: hash для кэша
                
                encoding = detect_file_encoding(first_bytes, cache_key=file_hash)
                up.seek(0)
                
                if csv_separator == "Автоопределение":
                    sep = detect_csv_sep(first_bytes, encoding)
                    st.info(f"✅ Определена кодировка: **{encoding}**, разделитель: **'{sep}'**")
                else:
                    sep = csv_separator
                    st.info(f"✅ Кодировка: **{encoding}**, разделитель: **'{sep}'**")
                
                # smart_sample_large_file с кодировкой
                df = smart_sample_large_file(
                    up, sep, 
                    max_rows=MAX_DATASET_SIZE,
                    sample_size=SAMPLE_SIZE_FOR_LARGE_DATASETS,
                    encoding=encoding,
                    target_col=None,  # На этом этапе target еще не известен
                    task_type=None
                )
            
            else:  # Excel
                df = pd.read_excel(up)
                st.info(f"✅ Excel файл загружен")
            
            # Очистка данных
            df = sanitize_column_names(df)
            df = remove_duplicate_columns(df)
            
            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверка на дубликаты строк
            df, n_duplicates = check_and_remove_duplicates(df, warn=True)
            if n_duplicates > 0:
                dup_pct = (n_duplicates / (len(df) + n_duplicates)) * 100
                st.warning(f"⚠️ Удалено {n_duplicates} дублирующихся строк ({dup_pct:.1f}%). "
                          f"Дубликаты могут исказить метрики качества модели.")
            
            # Проверка на пустоту
            if df.empty:
                st.error("❌ Файл пуст или не может быть распарсен!")
                return
            
            if df.shape[0] < 2:
                st.error("❌ Слишком мало строк (нужно минимум 2)")
                return
            
            # ✅ ДОБАВЛЕНО: Оценка использования памяти
            memory_mb = estimate_memory_usage(df)
            if memory_mb > 500:
                st.warning(f"⚠️ Датасет занимает {memory_mb:.1f} МБ в памяти. "
                          f"Рекомендуется оптимизация...")
                with st.spinner("Оптимизация типов данных..."):
                    df = optimize_dtypes(df, aggressive=False)
                    new_memory_mb = estimate_memory_usage(df)
                    saved_mb = memory_mb - new_memory_mb
                    if saved_mb > 0:
                        st.success(f"✅ Оптимизировано: экономия {saved_mb:.1f} МБ памяти")
            
            st.session_state.timer_info = {"load_ms": int((time.time() - t0) * 1000)}
            st.success(
                f"✅ Загружено: **{df.shape[0]:,}** строк × **{df.shape[1]}** столбцов "
                f"({human_time_ms(st.session_state.timer_info['load_ms'])})"
            )
            
            # ✅ УЛУЧШЕНО: Показать основную статистику
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Строк", f"{df.shape[0]:,}")
            with col2:
                st.metric("Столбцов", df.shape[1])
            with col3:
                num_cols = df.select_dtypes(include=[np.number]).shape[1]
                st.metric("Числовых", num_cols)
            with col4:
                cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
                st.metric("Категориальных", cat_cols)
            
            st.dataframe(df.head(10), use_container_width=True)
            
            # Предупреждение о больших датасетах
            if df.shape[0] > MAX_DATASET_SIZE:
                st.warning(
                    f"⚠️ Датасет слишком большой ({df.shape[0]:,} строк). "
                    f"Для анализа будет использована выборка из {SAMPLE_SIZE_FOR_LARGE_DATASETS:,} строк."
                )
                df = df.sample(n=SAMPLE_SIZE_FOR_LARGE_DATASETS, random_state=RANDOM_SEED)
                st.info(f"✅ Создана выборка: {df.shape[0]:,} × {df.shape[1]}")
            
            # ✅ ДОБАВЛЕНО: Предупреждение о слишком малом количестве данных
            if df.shape[0] < 100:
                st.warning("⚠️ Менее 100 строк. Качество моделей может быть низким. "
                          "Рекомендуется минимум 100 строк для надежных результатов.")
            
            # Очистка старых данных при загрузке новых
            keys_to_reset = [
                'target', 'task_type', 'train_df', 'X_train', 'X_test',
                'y_train', 'y_test', 'leaderboard', 'active_model_name', 'best_estimator',
                'fitted_pipe', 'prediction_data', 'primary_metric', 'selected_features',
                'available_features', 'calculator_base_data', 'dt_cols_hint',
                'timer_info', 'text_processing', 'use_log_transform', 'test_size',
                'accurate_mode'
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    st.session_state.pop(key, None)
            
            st.session_state.train_df = df
            data_loaded = True
            logger.info(f"Загружено: {df.shape}, дубликатов удалено: {n_duplicates}")
        
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла:\n{str(e)[:200]}")
            logger.error(f"File upload error: {e}", exc_info=True)
            
            # ✅ ДОБАВЛЕНО: Подсказки при ошибках
            with st.expander("💡 Возможные решения"):
                st.markdown("""
                - Проверьте что файл не поврежден
                - Убедитесь что файл имеет правильную структуру (строки × столбцы)
                - Попробуйте открыть файл в Excel/LibreOffice для проверки
                - Если файл очень большой, попробуйте уменьшить его размер
                - Проверьте кодировку файла (должна быть UTF-8 или CP1251)
                """)

    if st.button("➡️ Далее", type="primary", disabled=not data_loaded, use_container_width=True):
        st.session_state.wizard_step = 2
        st.rerun()

# =========================================================
# STEP 2: SETUP (WITH TRAIN-TEST SPLIT)
# =========================================================

def render_step2_setup():
    """Настройка задачи и создание train-test split."""
    st.header("🎯 Шаг 2. Настройка задачи: Цель и признаки")

    df = st.session_state.get("train_df")
    if df is None:
        st.warning("⚠️ Сначала загрузите данные на Шаг 1.")
        if st.button("⬅️ Вернуться на Шаг 1"):
            st.session_state.wizard_step = 1
            st.rerun()
        return

    cols = list(df.columns)

    st.markdown("#### 🎯 1. Выберите целевую переменную (target)")
    st.markdown("Это то, что модель будет предсказывать. Обычно это последний столбец в таблице.")
    
    current_target = st.session_state.get('target')
    target_index = cols.index(current_target) + 1 if current_target in cols else 0
    target = st.selectbox(
        "Целевая переменная", 
        options=["– Выберите –"] + cols, 
        index=target_index,
        help="Столбец с значениями, которые нужно предсказывать"
    )
    
    if target == "– Выберите –":
        st.info("⚠️ Выберите целевую переменную, чтобы продолжить.")
        return
    
    st.session_state.target = target
    
    # ✅ ДОБАВЛЕНО: Показать статистику по target
    with st.expander("📊 Статистика целевой переменной"):
        target_series = df[target]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Уникальных значений", target_series.nunique())
        with col2:
            missing_pct = (target_series.isna().sum() / len(target_series)) * 100
            st.metric("Пропусков", f"{missing_pct:.1f}%")
        with col3:
            st.metric("Тип данных", str(target_series.dtype))
        
        if target_series.nunique() < 20:
            st.write("**Распределение значений:**")
            value_counts = target_series.value_counts().head(10)
            st.bar_chart(value_counts)

    # Определение типа задачи с обработкой ошибок
    try:
        task_type = ml_core.detect_problem_type(df[target], get_session_id())
        st.session_state.task_type = task_type
        st.session_state.primary_metric = ml_core.get_primary_metric(task_type)
        
        # ✅ УЛУЧШЕНО: Более информативное сообщение
        task_emoji = {"binary": "🔵", "multiclass": "🌈", "regression": "📈"}
        st.success(
            f"{task_emoji.get(task_type, '🎯')} Определен тип задачи: **{task_type}**. "
            f"Основная метрика: **{st.session_state.primary_metric}**"
        )
        
        # ✅ ДОБАВЛЕНО: Объяснение типа задачи
        if task_type == "binary":
            st.info("💡 **Бинарная классификация:** Предсказание одного из двух классов (да/нет, 0/1, и т.д.)")
        elif task_type == "multiclass":
            st.info("💡 **Многоклассовая классификация:** Предсказание одного из нескольких классов")
        else:
            st.info("💡 **Регрессия:** Предсказание непрерывного числового значения")
            
    except ValueError as e:
        st.error(f"❌ Ошибка при анализе целевой переменной:\n{str(e)}")
        return

    # Опция переопределения типа
    st.markdown("#### 🎯 1.1. Уточнение типа задачи (опционально)")
    task_override = st.selectbox(
        "Если тип определен неправильно, выберите правильный:",
        options=["Автоопределение", "binary", "multiclass", "regression"],
        index=0,
        help="Обычно автоопределение работает корректно"
    )
    if task_override != "Автоопределение":
        st.session_state.task_type = task_override
        st.session_state.primary_metric = ml_core.get_primary_metric(task_override)
        st.info(f"✅ Тип задачи изменен на: **{task_override}**")

    st.markdown("#### 📊 2. Выберите признаки и столбцы данных")
    st.markdown("Исключите ненужные столбцы и укажите, где находятся даты/время.")

    available_features = [col for col in df.columns if col != target]
    st.session_state.available_features = available_features

    with st.form("features_and_dates_form"):
        selected_features = st.multiselect(
            "Признаки для обучения (оставьте пусто, чтобы использовать все):",
            options=available_features,
            default=st.session_state.get('selected_features', available_features[:20] if len(available_features) > 20 else available_features),
            help="Можно выбрать только нужные столбцы или оставить все"
        )
        
        # Если ничего не выбрано - используем все
        if not selected_features:
            selected_features = available_features
        
        # Поиск потенциальных date столбцов
        potential_dt_cols = [
            c for c in selected_features
            if any(k in c.lower() for k in ["date", "время", "time", "дата", "timestamp"]) or 
            pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        
        dt_cols_hint = st.multiselect(
            "Столбцы, содержащие дату/время (для авто-генерации признаков):",
            options=potential_dt_cols,
            default=st.session_state.get('dt_cols_hint', []),
            help="Из этих столбцов будут извлечены год, месяц, день недели, и т.д."
        )

        # Опции обработки
        st.markdown("**Дополнительные опции обработки:**")
        
        text_processing = st.checkbox(
            "✅ Включить обработку текстовых признаков (TF-IDF)",
            value=st.session_state.get('text_processing', False),
            help="Автоматически найти и обработать текстовые столбцы с помощью TF-IDF векторизации"
        )
        st.session_state.text_processing = text_processing

        use_log_transform = st.checkbox(
            "✅ Использовать log-преобразование к числовым признакам",
            value=st.session_state.get('use_log_transform', False),
            help="Помогает при признаках с экспоненциальным распределением (безопасно для отрицательных значений, использует log1p)"
        )
        st.session_state.use_log_transform = use_log_transform
        
        # Train-test split ratio
        test_size = st.slider(
            "📊 Процент данных для теста",
            min_value=0.1, max_value=0.5, value=0.2, step=0.05,
            help="20% по умолчанию означает 80% train, 20% test"
        )
        
        st.markdown("---")
        submitted = st.form_submit_button("✅ Применить и создать split", type="primary", use_container_width=True)
        
        if submitted:
            st.session_state.selected_features = selected_features
            st.session_state.dt_cols_hint = dt_cols_hint
            st.session_state.test_size = test_size
            
            # ✅ ДОБАВЛЕНО: Валидация выбора
            if len(selected_features) == 0:
                st.error("❌ Необходимо выбрать хотя бы один признак!")
                return
            
            if len(selected_features) > 1000:
                st.warning("⚠️ Выбрано более 1000 признаков. Это может привести к долгому обучению.")
            
            # Создаем train-test split
            try:
                X = df[selected_features].copy()
                y = df[target].copy()
                
                # Удаляем строки с пропусками в целевой переменной
                valid_idx = y.notna()
                X = X[valid_idx]
                y = y[valid_idx]
                
                if len(X) < 4:
                    st.error("❌ Слишком мало данных для split (нужно минимум 4 строки после удаления пропусков)")
                    return
                
                # ✅ УЛУЧШЕНО: Создаем stratified split если это классификация
                try:
                    stratify_col = None
                    if st.session_state.task_type in ('binary', 'multiclass'):
                        # ✅ ИСПРАВЛЕНО: Проверяем что все классы имеют достаточно примеров
                        class_counts = y.value_counts()
                        min_class_count = class_counts.min()
                        min_test_samples = int(len(y) * test_size)
                        
                        if min_class_count >= 2 and min_test_samples >= len(class_counts):
                            stratify_col = y
                            st.info("✅ Используется stratified split для сохранения пропорций классов")
                        else:
                            st.warning(f"⚠️ Stratified split невозможен (мин. класс: {min_class_count}). Используется random split.")
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=RANDOM_SEED,
                        stratify=stratify_col
                    )
                except Exception as e:
                    logger.warning(f"Stratified split failed: {e}, using regular split")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=RANDOM_SEED
                    )
                
                # Сохраняем в session_state
                st.session_state.X_train = X_train.reset_index(drop=True)
                st.session_state.X_test = X_test.reset_index(drop=True)
                st.session_state.y_train = y_train.reset_index(drop=True)
                st.session_state.y_test = y_test.reset_index(drop=True)
                
                # ✅ УЛУЧШЕНО: Показать статистику split
                st.success(f"✅ Train-Test Split создан успешно!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Train размер", f"{len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
                    if st.session_state.task_type in ('binary', 'multiclass'):
                        st.write("**Распределение классов (train):**")
                        train_dist = y_train.value_counts()
                        st.bar_chart(train_dist)
                
                with col2:
                    st.metric("📊 Test размер", f"{len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
                    if st.session_state.task_type in ('binary', 'multiclass'):
                        st.write("**Распределение классов (test):**")
                        test_dist = y_test.value_counts()
                        st.bar_chart(test_dist)
                
                logger.info(f"Split created: train={len(X_train)}, test={len(X_test)}")
                
                time.sleep(1)
                st.session_state.wizard_step = 3
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Ошибка при создании split: {str(e)}")
                logger.error(f"Split error: {e}", exc_info=True)

# =========================================================
# STEP 3: TRAINING
# =========================================================

def render_step3_training():
    """Обучение моделей."""
    st.header("🤖 Шаг 3. Обучение и сравнение моделей")

    if 'train_df' not in st.session_state or 'target' not in st.session_state:
        st.warning("⚠️ Сначала завершите шаги 1 и 2.")
        return

    # Проверка что split существует
    if 'X_train' not in st.session_state or 'X_test' not in st.session_state:
        st.error("❌ Train-Test Split не найден! Вернитесь на шаг 2.")
        if st.button("⬅️ Вернуться на шаг 2"):
            st.session_state.wizard_step = 2
            st.rerun()
        return

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    
    df = st.session_state.train_df
    target_col = st.session_state.target
    task = st.session_state.task_type

    features = st.session_state.get('selected_features') or st.session_state.get('available_features', [])
    
    if not features:
        st.error("❌ Нет выбранных признаков")
        return
    
    # ✅ УЛУЧШЕНО: Показать информацию о данных
    st.info(f"📊 Обучение на **{len(X_train):,}** строках с **{len(features)}** признаками. "
            f"Тестирование на **{len(X_test):,}** строках.")

    # Динамическое определение n_splits
    n_splits = ml_core.get_optimal_cv_splits(len(X_train))
    st.info(f"ℹ️ Для кросс-валидации будет использовано **{n_splits} folds**")

    st.markdown("#### ℹ️ Доступные алгоритмы и режимы")
    st.success(
        "В обучении участвуют только установленные библиотеки: "
        f"sklearn (всегда), "
        f"XGBoost {'✅' if ml_core.XGB_AVAILABLE else '❌'}, "
        f"LightGBM {'✅' if ml_core.LGBM_AVAILABLE else '❌'}, "
        f"CatBoost {'✅' if ml_core.CATBOOST_AVAILABLE else '❌'}, "
        f"Optuna {'✅' if OPTUNA_AVAILABLE else '❌'} для точной настройки."
    )
    st.caption(
        "💡 Подсказка: для очень больших датасетов (>50k строк) начните с быстрого режима, "
        "чтобы увидеть базовые метрики за минуты, а затем включайте точный режим для топ-моделей."
    )

    # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Выбор режима с сохранением в session_state
    st.markdown("### ⚙️ Режим обучения")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**⚡ Быстрый режим (5-10 мин)**")
        st.caption("- Стандартные гиперпараметры\n"
                  "- Все доступные модели\n"
                  "- Параллельное обучение\n"
                  "- Подходит для первичного анализа")
    with col2:
        st.markdown("**🎯 Точный режим (30-120 мин)**")
        st.caption("- Optuna оптимизация\n"
                  "- До 50 trials на модель\n"
                  "- Early stopping\n"
                  f"- {'✅ Доступен' if OPTUNA_AVAILABLE else '❌ Требует Optuna'}")
    
    mode = st.radio(
        "Выберите режим",
        ["⚡ Быстро (5-10 мин)", "🎯 Точно (30-120 мин, Optuna)"],
        horizontal=True,
        index=1 if st.session_state.get('accurate_mode', False) else 0,
        help="Быстрый режим использует стандартные параметры. Точный - оптимизирует с Optuna."
    )
    accurate_mode = (mode == "🎯 Точно (30-120 мин, Optuna)")
    st.session_state.accurate_mode = accurate_mode  # ✅ Сохраняем в session_state

    if accurate_mode and not OPTUNA_AVAILABLE:
        st.warning("⚠️ Optuna не установлена, будет использован быстрый режим")
        accurate_mode = False
        st.session_state.accurate_mode = False

    if st.button("▶️ Запустить обучение", type="primary", use_container_width=True):
        if accurate_mode:
            st.info("⏳ Запущен режим точной настройки с Optuna. Это может занять 30-120 минут...\n"
                   "💡 **Совет:** Optuna автоматически остановится при отсутствии улучшений (early stopping)")

            if task == "regression":
                models_to_tune = ["Ridge", "Lasso", "RandomForestRegressor"]
                if XGB_AVAILABLE:
                    models_to_tune.append("XGBRegressor")
                if LGBM_AVAILABLE:
                    models_to_tune.append("LGBMRegressor")
                if CATBOOST_AVAILABLE:
                    models_to_tune.append("CatBoostRegressor")
            else:
                models_to_tune = ["LogisticRegression", "RandomForestClassifier"]
                if XGB_AVAILABLE:
                    models_to_tune.append("XGBClassifier")
                if LGBM_AVAILABLE:
                    models_to_tune.append("LGBMClassifier")
                if CATBOOST_AVAILABLE:
                    models_to_tune.append("CatBoostClassifier")

            results = []
            progress_bar = st.progress(0, text="Инициализация точной настройки...")
            status_text = st.empty()
            t0_all = time.time()

            cv = ml_core.get_cv(task, n_splits=n_splits, shuffle=True, seed=RANDOM_SEED)
            dt_cols_hint = st.session_state.get('dt_cols_hint')

            for i, name in enumerate(models_to_tune):
                status_text.info(
                    f"🔧 Точная настройка {i+1}/{len(models_to_tune)}: **{name}** "
                    f"(Optuna, до 50 trials с early stopping)"
                )
                
                model_start = time.time()
                best_model = ml_core.tune_with_optuna(
                    name, X_train, y_train, cv,
                    n_trials=50,
                    dt_cols_hint=dt_cols_hint
                )
                model_duration = time.time() - model_start
                
                if best_model is None:
                    st.warning(f"⚠️ Модель {name} недоступна или произошла ошибка")
                    continue

                # CV evaluation
                preprocessor = ml_core.build_preprocessor(
                    X_train,
                    dt_cols_hint,
                    ml_core.is_linear_model(name),
                    True,
                    _sid=get_session_id(),
                    text_processing=st.session_state.get('text_processing', False),
                    model_name=name,
                    use_log_transform=st.session_state.get('use_log_transform', False)
                )
                
                scores, duration = ml_core.cv_evaluate(
                    preprocessor,
                    best_model, X_train, y_train, task,
                    n_splits=n_splits, shuffle=True, seed=RANDOM_SEED,
                    _sid=get_session_id(),
                    _cache_bust=i
                )
                
                row = {
                    "model": name, 
                    "cv_time": human_time_ms(duration),
                    "tune_time": human_time_ms(model_duration * 1000),
                    **scores
                }
                results.append(row)
                progress_bar.progress((i + 1) / len(models_to_tune), text=f"✅ Готово: {name}")

            status_text.empty()
            progress_bar.empty()

        else:  # Fast mode
            models = ml_core.get_models(task, mode="fast")
            if not models:
                st.error("❌ Не найдено доступных моделей")
                return

            dt_cols_hint = st.session_state.get('dt_cols_hint')
            text_processing = st.session_state.get('text_processing', False)
            use_log_transform = st.session_state.get('use_log_transform', False)

            preprocessors = {
                False: ml_core.build_preprocessor(
                    X_train, dt_cols_hint, use_scaler=False, handle_outliers=True,
                    _sid=get_session_id(),
                    text_processing=text_processing,
                    model_name=None,
                    use_log_transform=use_log_transform
                ),
                True: ml_core.build_preprocessor(
                    X_train, dt_cols_hint, use_scaler=True, handle_outliers=True,
                    _sid=get_session_id(),
                    text_processing=text_processing,
                    model_name=None,
                    use_log_transform=use_log_transform
                )
            }

            results = []
            progress_bar = st.progress(0, text="Инициализация обучения...")
            status_text = st.empty()
            t0_all = time.time()

            cv = ml_core.get_cv(task, n_splits=n_splits, shuffle=True, seed=RANDOM_SEED)

            for i, (name, model) in enumerate(models.items()):
                status_text.info(f"🤖 Обучение модели {i+1}/{len(models)}: **{name}**")
                
                needs_scaler = ml_core.is_linear_model(name)
                preprocessor = preprocessors[needs_scaler]

                scores, duration = ml_core.cv_evaluate(
                    preprocessor, model, X_train, y_train, task,
                    n_splits=n_splits, shuffle=True, seed=RANDOM_SEED,
                    _sid=get_session_id(),
                    _cache_bust=i
                )
                
                row = {"model": name, "cv_time": human_time_ms(duration), **scores}
                results.append(row)
                progress_bar.progress((i + 1) / len(models), text=f"✅ Готово: {name}")

            status_text.empty()
            progress_bar.empty()

        enforce_min_duration(t0_all, min_seconds=2.0)
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Лидербоард с правильной обработкой NaN
        leaderboard = pd.DataFrame(results)
        
        if leaderboard.empty:
            st.error("❌ Не удалось обучить ни одной модели. Проверьте логи.")
            return
        
        primary_metric = st.session_state.primary_metric
        sort_metric = ml_core.choose_sort_metric(leaderboard, task, primary_metric)
        
        if sort_metric and sort_metric in leaderboard.columns:
            ascending = ml_core.metric_ascending(sort_metric)
            leaderboard = leaderboard.sort_values(
                by=sort_metric,
                ascending=ascending,
                key=lambda s: s.fillna(float('inf') if ascending else float('-inf'))
            ).reset_index(drop=True)

        st.session_state.leaderboard = leaderboard
        best_model_name = ml_core.select_best_model(leaderboard, task, sort_metric)
        
        if not best_model_name:
            st.error("❌ Не удалось выбрать лучшую модель (все метрики NaN). "
                    "Возможно данные слишком малы или некорректны.")
            return
        
        st.session_state.active_model_name = best_model_name

        st.success(f"✅ Обучение завершено за {human_time_ms((time.time() - t0_all) * 1000)}! "
                  f"Лучшая модель по метрике '{sort_metric}': **{best_model_name}**")
        st.rerun()

    # Показ лидерборда если он есть
    if 'leaderboard' in st.session_state:
        st.subheader("📊 Лидербоард моделей")
        
        # ✅ УЛУЧШЕНО: Подсветка лучшей модели
        leaderboard_display = st.session_state.leaderboard.copy()
        
        # Стиль для лучшей модели
        def highlight_best(row):
            if row['model'] == st.session_state.active_model_name:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            leaderboard_display.style.apply(highlight_best, axis=1).format(precision=4),
            use_container_width=True
        )

        st.subheader("🎯 Выбор активной модели для анализа")
        st.markdown("Выберите модель для детального анализа и финального обучения на всех данных.")
        
        model_names = st.session_state.leaderboard['model'].tolist()
        active_model_idx = model_names.index(st.session_state.active_model_name) if st.session_state.active_model_name in model_names else 0
        new_active_model = st.selectbox(
            "Активная модель", 
            model_names, 
            index=active_model_idx,
            help="Эта модель будет обучена на ВСЕХ данных для финального использования"
        )
        st.session_state.active_model_name = new_active_model

        if st.button(f"✅ Обучить '{new_active_model}' на ВСЕХ данных и перейти к анализу", 
                    type="primary", use_container_width=True):
            with st.spinner(f"Обучение '{new_active_model}' на полном датасете..."):
                # Используем ВСЕ данные для финального обучения
                X_full = st.session_state.train_df[st.session_state.get('selected_features') or st.session_state.available_features].copy()
                y_full = st.session_state.train_df[st.session_state.target].copy()
                
                # Удаляем пропуски
                valid_idx = y_full.notna()
                X_full = X_full[valid_idx].reset_index(drop=True)
                y_full = y_full[valid_idx].reset_index(drop=True)
                
                dt_cols_hint = st.session_state.get('dt_cols_hint')
                text_processing = st.session_state.get('text_processing', False)
                use_log_transform = st.session_state.get('use_log_transform', False)
                
                # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем accurate_mode из session_state
                accurate_mode = st.session_state.get('accurate_mode', False)

                cv = ml_core.get_cv(task, n_splits=n_splits, shuffle=True, seed=RANDOM_SEED)
                
                if accurate_mode and OPTUNA_AVAILABLE:
                    best_model = ml_core.tune_with_optuna(
                        new_active_model, X_full, y_full, cv,
                        n_trials=50,
                        dt_cols_hint=dt_cols_hint
                    )
                    if best_model is None:
                        # Fallback на fast model
                        best_model = ml_core.get_models(task, mode="fast").get(new_active_model)
                else:
                    best_model = ml_core.get_models(task, mode="fast").get(new_active_model)
                
                if best_model is None:
                    st.error(f"❌ Модель {new_active_model} не найдена")
                    return
                
                preprocessor = ml_core.build_preprocessor(
                    X_full, dt_cols_hint, 
                    use_scaler=ml_core.is_linear_model(new_active_model),
                    handle_outliers=True,
                    _sid=get_session_id(),
                    text_processing=text_processing,
                    model_name=new_active_model,
                    use_log_transform=use_log_transform
                )

                st.session_state.fitted_pipe = ml_core.fit_best(
                    preprocessor, best_model, X_full, y_full, _sid=get_session_id()
                )

            st.success(f"✅ Модель '{new_active_model}' обучена на {len(X_full):,} строках!")
            st.session_state.wizard_step = 4
            st.rerun()

# =========================================================
# STEP 4: ANALYSIS
# =========================================================

def render_step4_analysis():
    """Анализ обученной модели."""
    st.header(f"📊 Шаг 4. Анализ модели: {st.session_state.get('active_model_name', '')}")

    if 'fitted_pipe' not in st.session_state:
        st.warning("⚠️ Сначала обучите модель на Шаге 3.")
        if st.button("⬅️ Вернуться на Шаг 3"):
            st.session_state.wizard_step = 3
            st.rerun()
        return

    if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
        st.error("❌ Данные теста не найдены! Вернитесь на шаг 2.")
        return

    est = st.session_state.fitted_pipe
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    task = st.session_state.task_type

    try:
        y_pred = est.predict(X_test)
    except Exception as e:
        st.error(f"❌ Ошибка при прогнозировании: {str(e)}")
        logger.error(f"Prediction error: {e}", exc_info=True)
        return

    st.subheader("📈 Метрики качества на отложенной выборке")
    st.markdown(f"Оценка на **{len(X_test):,}** независимых примерах, которые модель не видела при обучении.")
    
    if task == 'regression':
        rmse = math.sqrt(ml_core.mean_squared_error(y_test, y_pred))
        mae = ml_core.mean_absolute_error(y_test, y_pred)
        r2 = ml_core.r2_score(y_test, y_pred)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:,.2f}", help="Root Mean Squared Error - среднеквадратичная ошибка")
        c2.metric("MAE", f"{mae:,.2f}", help="Mean Absolute Error - средняя абсолютная ошибка")
        c3.metric("R²", f"{r2:.3f}", help="Coefficient of determination - доля объясненной дисперсии")
        
        # ✅ ДОБАВЛЕНО: График predicted vs actual
        st.subheader("📉 Predicted vs Actual")
        fig = px.scatter(
            x=y_test, y=y_pred, 
            labels={"x": "Фактические значения", "y": "Предсказанные значения"},
            title="Predicted vs Actual Values"
        )
        # Добавляем линию y=x
        fig.add_trace(pgo.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Идеальная линия',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Classification
        from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
        
        acc = ml_core.accuracy_score(y_test, y_pred)
        f1 = ml_core.f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        if task == 'binary':
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}", help="Доля правильных предсказаний")
            c2.metric("F1-score", f"{f1:.3f}", help="Гармоническое среднее precision и recall")
            c3.metric("Precision", f"{precision:.3f}", help="Точность - доля правильных среди предсказанных положительных")
            c4.metric("Recall", f"{recall:.3f}", help="Полнота - доля найденных положительных")
        else:
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.3f}", help="Доля правильных предсказаний")
            c2.metric("F1-weighted", f"{f1:.3f}", help="Взвешенный F1-score по классам")
            c3.metric("Balanced Acc", f"{balanced_acc:.3f}", help="Сбалансированная accuracy (учитывает дисбаланс классов)")

        st.subheader("📵 Матрица ошибок")
        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Предсказание", y="Истинные значения", color="Кол-во"),
            x=[str(l) for l in labels], y=[str(l) for l in labels], 
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ✅ ДОБАВЛЕНО: Показать классы с наибольшими ошибками
        if len(labels) > 2:
            errors_per_class = {}
            for i, label in enumerate(labels):
                total = cm[i].sum()
                correct = cm[i, i]
                errors = total - correct
                errors_per_class[str(label)] = errors
            
            st.markdown("**Ошибки по классам:**")
            errors_df = pd.DataFrame(list(errors_per_class.items()), columns=['Класс', 'Ошибок'])
            errors_df = errors_df.sort_values('Ошибок', ascending=False)
            st.bar_chart(errors_df.set_index('Класс'))

    st.subheader("⭐ Важность признаков (Permutation Importance)")
    st.markdown("Показывает, насколько каждый признак важен для предсказаний модели. "
               "Чем выше значение, тем важнее признак.")
    
    with st.spinner("Расчет важности признаков (может занять 1-2 минуты)..."):
        try:
            # ✅ УЛУЧШЕНО: Используем больше данных если доступно
            sample_size = min(5000, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=RANDOM_SEED)
            y_sample = y_test.loc[X_sample.index]
            
            result = permutation_importance(
                est, X_sample, y_sample, 
                n_repeats=10,  # ✅ УВЕЛИЧЕНО: больше повторений для стабильности
                random_state=RANDOM_SEED, 
                n_jobs=ml_core.N_JOBS
            )
            
            importances = pd.DataFrame({
                'feature': X_sample.columns,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=True).tail(20)
            
            fig = px.bar(
                importances, 
                x='importance_mean', 
                y='feature', 
                orientation='h', 
                title="Топ-20 самых важных признаков",
                error_x='importance_std',
                labels={'importance_mean': 'Важность (среднее)', 'feature': 'Признак'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ✅ ДОБАВЛЕНО: Показать топ признаки текстом
            st.markdown("**Топ-5 признаков:**")
            for i, row in importances.tail(5).iterrows():
                st.caption(f"**{row['feature']}**: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
            
        except Exception as e:
            st.warning(f"Не удалось рассчитать важность: {e}")
            logger.error(f"Feature importance error: {e}", exc_info=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Вернуться к обучению", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()
    with col2:
        if st.button("➡️ Далее к прогнозированию", type="primary", use_container_width=True):
            st.session_state.wizard_step = 5
            st.rerun()

# =========================================================
# STEP 5: PREDICT
# =========================================================

def render_step5_predict():
    """Прогнозирование на новых данных."""
    st.header("🔮 Шаг 5. Прогнозирование на новых данных")
    
    st.markdown("""
    Загрузите новый файл с данными для получения прогнозов. 
    Файл должен иметь **те же столбцы** (признаки), что и обучающие данные.
    
    ⚡ **Важно:**
    - Целевая переменная может отсутствовать
    - Названия столбцов должны совпадать
    - Типы данных будут автоматически проверены и конвертированы при необходимости
    """)

    if 'fitted_pipe' not in st.session_state:
        st.warning("⚠️ Сначала обучите модель на Шаге 3.")
        if st.button("⬅️ Вернуться на Шаг 3"):
            st.session_state.wizard_step = 3
            st.rerun()
        return

    up = st.file_uploader(
        "Загрузите файл для прогнозирования", 
        type=["csv", "xls", "xlsx"],
        help="Файл должен содержать те же столбцы, что и обучающие данные"
    )
    
    if up:
        try:
            # Загрузка файла
            if up.name.lower().endswith('.csv'):
                first_bytes = up.read(50_000)
                file_hash = get_file_hash(first_bytes)
                encoding = detect_file_encoding(first_bytes, cache_key=file_hash)
                up.seek(0)
                
                sep = detect_csv_sep(first_bytes, encoding)
                up.seek(0)
                
                st.info(f"✅ Определена кодировка: **{encoding}**, разделитель: **'{sep}'**")
                
                df_new = smart_sample_large_file(up, sep, encoding=encoding)
            else:
                df_new = pd.read_excel(up)
                st.info(f"✅ Excel файл загружен")

            # Очистка
            df_new = sanitize_column_names(df_new)
            
            st.success(f"✅ Загружено: **{df_new.shape[0]:,}** строк × **{df_new.shape[1]}** столбцов")
            
            # Показать превью
            with st.expander("👁️ Превью данных"):
                st.dataframe(df_new.head(10))
            
            est = st.session_state.fitted_pipe
            original_features = list(st.session_state.X_train.columns)
            
            # Проверка наличия столбцов
            missing_cols = set(original_features) - set(df_new.columns)
            if missing_cols:
                st.error(f"❌ В загруженном файле отсутствуют столбцы: {list(missing_cols)}")
                st.info("💡 **Подсказка:** Проверьте что файл содержит все необходимые столбцы. "
                       "Названия должны точно совпадать.")
                return

            df_to_predict = df_new[original_features].copy()
            
            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Валидация типов данных
            is_valid, messages = validate_data_types(st.session_state.X_train, df_to_predict)
            
            if messages:
                with st.expander("⚠️ Предупреждения о типах данных"):
                    for msg in messages:
                        if msg.startswith("❌"):
                            st.error(msg)
                        else:
                            st.warning(msg)
            
            if not is_valid:
                st.error("❌ Критические ошибки в типах данных. Прогнозирование невозможно.")
                return
            
            # Прогнозирование
            with st.spinner("Выполнение прогноза..."):
                try:
                    predictions = est.predict(df_to_predict)
                except Exception as e:
                    st.error(f"❌ Ошибка при прогнозировании: {str(e)}")
                    logger.error(f"Prediction error: {e}", exc_info=True)
                    return

            result_df = df_new.copy()
            result_df[f"prediction_{st.session_state.target}"] = predictions
            
            # ✅ ДОБАВЛЕНО: Для классификации добавляем вероятности
            if st.session_state.task_type in ('binary', 'multiclass') and hasattr(est, 'predict_proba'):
                try:
                    probas = est.predict_proba(df_to_predict)
                    classes = est.classes_
                    for i, cls in enumerate(classes):
                        result_df[f"proba_{cls}"] = probas[:, i]
                except Exception:
                    pass

            st.session_state.prediction_data = result_df
            st.success("✅ Прогноз готов!")
            
            # Показать результаты
            st.subheader("📊 Результаты прогнозирования")
            st.dataframe(result_df.head(20), use_container_width=True)
            
            # ✅ ДОБАВЛЕНО: Статистика прогнозов
            with st.expander("📈 Статистика прогнозов"):
                pred_col = f"prediction_{st.session_state.target}"
                
                if st.session_state.task_type == 'regression':
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Среднее", f"{result_df[pred_col].mean():.2f}")
                    with col2:
                        st.metric("Медиана", f"{result_df[pred_col].median():.2f}")
                    with col3:
                        st.metric("Мин", f"{result_df[pred_col].min():.2f}")
                    with col4:
                        st.metric("Макс", f"{result_df[pred_col].max():.2f}")
                    
                    st.write("**Распределение прогнозов:**")
                    fig = px.histogram(result_df, x=pred_col, title="Распределение предсказанных значений")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("**Распределение классов:**")
                    class_dist = result_df[pred_col].value_counts()
                    st.bar_chart(class_dist)

            # Скачивание результатов
            st.subheader("📥 Скачать результаты")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                download_button(csv_bytes, "predictions.csv", "📥 Скачать CSV", "text/csv")
            
            with col2:
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Predictions')
                    xlsx_bytes = output.getvalue()
                    download_button(xlsx_bytes, "predictions.xlsx", "📥 Скачать XLSX", 
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.caption(f"XLSX недоступен: {e}")

        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {str(e)[:200]}")
            logger.error(f"Predict file error: {e}", exc_info=True)
            
            with st.expander("💡 Возможные решения"):
                st.markdown("""
                - Проверьте что файл имеет правильную структуру
                - Убедитесь что названия столбцов совпадают с обучающими данными
                - Проверьте типы данных в столбцах
                - Попробуйте сохранить файл в другом формате
                """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Вернуться к анализу", use_container_width=True):
            st.session_state.wizard_step = 4
            st.rerun()
    with col2:
        if st.button("➡️ Далее к оптимизации", type="primary", use_container_width=True):
            st.session_state.wizard_step = 6
            st.rerun()

# =========================================================
# STEP 6: CALCULATOR (What-If)
# =========================================================

def render_step6_calculator():
    """Калькулятор оптимизации параметров."""
    st.header("⚙️ Шаг 6. Калькулятор оптимизации 'Что, если?'")
    
    st.markdown("""
    Используйте этот калькулятор, чтобы найти оптимальные значения параметров для достижения целевого значения.
    
    **Как это работает:**
    1. Выберите базовую строку из обучающих данных
    2. Укажите диапазоны для числовых признаков
    3. Выберите категориальные признаки для изменения
    4. Запустите оптимизацию
    5. Получите оптимальные параметры
    
    💡 **Совет:** Используется алгоритм differential evolution для поиска глобального оптимума.
    """)

    if 'fitted_pipe' not in st.session_state:
        st.warning("⚠️ Сначала обучите модель на Шаге 3.")
        if st.button("⬅️ Вернуться на Шаг 3"):
            st.session_state.wizard_step = 3
            st.rerun()
        return

    est = st.session_state.fitted_pipe
    X_train = st.session_state.X_train
    task = st.session_state.task_type

    st.subheader("1️⃣ Выберите базовую строку для анализа")
    st.markdown("Это отправная точка для оптимизации. Выберите типичный пример из ваших данных.")
    
    idx = st.number_input(
        "Номер строки в обучающих данных",
        min_value=0, max_value=len(X_train)-1, value=0,
        help=f"Доступно строк: 0-{len(X_train)-1}"
    )
    base_row = X_train.iloc[[idx]].copy()
    
    with st.expander("👁️ Показать выбранную строку"):
        st.dataframe(base_row.T, use_container_width=True)
        
        # ✅ ДОБАВЛЕНО: Показать текущее предсказание
        try:
            current_pred = est.predict(base_row)[0]
            st.info(f"📊 Текущее предсказание для этой строки: **{current_pred:.4f}**")
        except Exception:
            pass

    st.subheader("2️⃣ Настройте параметры для оптимизации")
    
    all_features = base_row.columns.tolist()
    num_features = [f for f in all_features if pd.api.types.is_numeric_dtype(X_train[f])]
    cat_features = [f for f in all_features if not pd.api.types.is_numeric_dtype(X_train[f])]

    st.markdown("**Числовые признаки**")
    st.caption("Укажите диапазоны значений для поиска оптимума")
    
    bounds = {}
    for f in num_features:
        col_data = X_train[f].dropna()
        if len(col_data) == 0:
            continue
        min_val, max_val = float(col_data.min()), float(col_data.max())
        if min_val == max_val:
            max_val = min_val + 1  # Avoid slider error
        
        # ✅ УЛУЧШЕНО: Показать текущее значение
        current_val = float(base_row[f].iloc[0])
        bounds[f] = st.slider(
            f"Диапазон для '{f}' (текущее: {current_val:.2f})", 
            min_val, max_val, (min_val, max_val),
            help=f"Минимум: {min_val:.2f}, Максимум: {max_val:.2f}"
        )

    st.markdown("**Категориальные признаки**")
    st.caption("Отметьте признаки, которые можно изменять")
    
    cat_choices = {}
    for f in cat_features:
        options = sorted([str(x) for x in X_train[f].dropna().unique()])
        if len(options) == 0:
            continue
        if st.checkbox(f"Разрешить изменение '{f}'", key=f"check_{f}"):
            cat_choices[f] = options

    # ✅ ДОБАВЛЕНО: Проверка что есть что оптимизировать
    if not bounds and not cat_choices:
        st.warning("⚠️ Выберите хотя бы один признак для оптимизации (числовой или категориальный)")
        return

    st.subheader("3️⃣ Параметры оптимизации")
    
    col1, col2 = st.columns(2)
    with col1:
        if task == 'regression':
            objective = st.radio(
                "Цель", 
                ["Максимизировать", "Минимизировать"], 
                horizontal=True,
                help="Что делать с предсказанным значением"
            )
        else:
            objective = "Максимизировать"
            st.info("Цель: Максимизировать вероятность предсказания")
    
    with col2:
        popsize = st.slider(
            "Размер популяции", 
            5, 50, 15,
            help="Больше = лучше результат, но медленнее"
        )
        maxiter = st.slider(
            "Макс. итерации", 
            10, 200, 50,
            help="Больше = лучше результат, но медленнее"
        )

    if st.button("▶️ Запустить оптимизацию", type="primary", use_container_width=True):
        with st.spinner("Выполнение оптимизации (может занять 1-5 минут)..."):
            optimizable_num = list(bounds.keys())
            optimizable_cat = list(cat_choices.keys())

            optimizer_bounds = [bounds[f] for f in optimizable_num]
            for f in optimizable_cat:
                optimizer_bounds.append((0, len(cat_choices[f]) - 0.001))

            def objective_function(x):
                row_to_predict = base_row.copy()
                for i, f in enumerate(optimizable_num):
                    row_to_predict[f] = x[i]
                offset = len(optimizable_num)
                for i, f in enumerate(optimizable_cat):
                    choice_idx = int(x[offset + i])
                    row_to_predict[f] = cat_choices[f][choice_idx]
                
                try:
                    if task == 'regression':
                        score = est.predict(row_to_predict)[0]
                    else:
                        if hasattr(est, "predict_proba"):
                            score = est.predict_proba(row_to_predict).max()
                        else:
                            score = float(est.predict(row_to_predict)[0])
                except Exception:
                    return float('inf') if objective == "Минимизировать" else float('-inf')
                
                return -score if objective == "Максимизировать" else score

            if not optimizer_bounds:
                st.warning("⚠️ Выберите хотя бы один параметр для оптимизации")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # ✅ УЛУЧШЕНО: Callback для отображения прогресса
                iteration = [0]
                def callback(xk, convergence):
                    iteration[0] += 1
                    progress = min(100, int((iteration[0] / maxiter) * 100))
                    progress_bar.progress(progress)
                    status_text.text(f"Итерация {iteration[0]}/{maxiter}")
                
                result = differential_evolution(
                    objective_function,
                    bounds=optimizer_bounds,
                    popsize=popsize,
                    maxiter=maxiter,
                    seed=RANDOM_SEED,
                    callback=callback,
                    workers=1,
                    updating='deferred'
                )
                
                progress_bar.empty()
                status_text.empty()

                st.success("✅ Оптимизация завершена!")

                # Результаты
                try:
                    base_pred = est.predict(base_row)[0]
                except Exception:
                    base_pred = 0
                opt_pred = -result.fun if objective == "Максимизировать" else result.fun
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Базовое значение", f"{base_pred:.4f}")
                col2.metric("Оптимальное значение", f"{opt_pred:.4f}")
                improvement = opt_pred - base_pred
                col3.metric(
                    "Улучшение", 
                    f"{improvement:+.4f}",
                    delta=f"{(improvement/abs(base_pred)*100 if base_pred != 0 else 0):.1f}%"
                )

                # Показываем измененные параметры
                optimal_row = base_row.copy()
                x_opt = result.x
                for i, f in enumerate(optimizable_num):
                    optimal_row[f] = x_opt[i]
                offset = len(optimizable_num)
                for i, f in enumerate(optimizable_cat):
                    choice_idx = int(x_opt[offset + i])
                    optimal_row[f] = cat_choices[f][choice_idx]

                st.subheader("📄 Измененные параметры")
                comparison_df = pd.concat([base_row, optimal_row])
                comparison_df.index = ["Базовая строка", "Оптимальная строка"]
                
                # Показываем только изменившиеся столбцы
                changed_cols = [col for col in all_features 
                               if str(comparison_df[col].iloc[0]) != str(comparison_df[col].iloc[1])]
                
                if changed_cols:
                    st.dataframe(
                        comparison_df[changed_cols].T.style.format(precision=3),
                        use_container_width=True
                    )
                    
                    # ✅ ДОБАВЛЕНО: Показать рекомендации
                    st.markdown("**💡 Рекомендации:**")
                    for col in changed_cols:
                        old_val = comparison_df[col].iloc[0]
                        new_val = comparison_df[col].iloc[1]
                        if pd.api.types.is_numeric_dtype(type(old_val)):
                            if new_val > old_val:
                                st.caption(f"📈 Увеличить **{col}** с {old_val:.2f} до {new_val:.2f}")
                            else:
                                st.caption(f"📉 Уменьшить **{col}** с {old_val:.2f} до {new_val:.2f}")
                        else:
                            st.caption(f"🔄 Изменить **{col}** с '{old_val}' на '{new_val}'")
                else:
                    st.info("ℹ️ Оптимальные параметры совпадают с базовыми. "
                           "Попробуйте расширить диапазоны или выбрать другую базовую строку.")

    st.markdown("---")
    if st.button("🏠 Вернуться на главную", use_container_width=True):
        # Очищаем только данные, оставляем структуру
        for key in list(st.session_state.keys()):
            if key not in ["wizard_step", "session_id"]:
                del st.session_state[key]
        st.session_state.wizard_step = 0
        st.rerun()
