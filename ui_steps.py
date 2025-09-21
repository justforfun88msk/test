# ui_steps.py (ФИНАЛЬНАЯ ВЕРСИЯ v6 - с XLSX)

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

# Импорты из нашего проекта
import ml_core
from utils import detect_csv_sep, human_time_ms, enforce_min_duration, download_button
from ui_config import MODEL_DESCRIPTIONS, get_model_tags, RANDOM_SEED

# ... (Код функций render_step0_home по render_step4_analysis остается без изменений) ...

def render_step0_home():
    """Отрисовывает домашнюю страницу."""
    st.title("Добро пожаловать в Sminex ML!")
    st.markdown("""
    **Sminex ML — это инструмент для автоматизации машинного обучения.**
    Он позволяет:
    * **Загружать** данные в формате CSV/XLSX.
    * **Автоматически определять** тип задачи (классификация или регрессия).
    * **Обучать и сравнивать** десятки моделей машинного обучения.
    * **Получать детальный отчет** о качестве модели.
    * **Делать прогнозы** на новых данных.
    * **Использовать уникальный калькулятор "Что, если?"** для поиска оптимальных параметров.
    """)
    st.subheader("Как начать?")
    st.markdown("""
    1.  Нажмите **"📁 1. Загрузка данных"** в боковом меню.
    2.  Загрузите ваш файл с данными.
    3.  Следуйте инструкциям на каждом шаге.
    """)
    if st.button("🚀 Начать новый проект", type="primary"):
        st.session_state.wizard_step = 1
        st.rerun()

def render_step1_upload():
    """Отрисовывает UI для загрузки данных."""
    st.header("Шаг 1. Загрузка данных")
    st.markdown("""
    Загрузите ваш файл с данными в формате CSV или Excel. Данные должны быть в виде таблицы, где:
    * **Строки** — это отдельные объекты (например, клиенты, товары, сделки).
    * **Столбцы** — это характеристики (признаки) этих объектов и целевая переменная.
    """)
    up = st.file_uploader("CSV/XLSX файл", type=["csv", "xls", "xlsx"], help="Максимальный размер файла: 200 МБ")
    
    data_loaded = False
    if up is not None:
        try:
            t0 = time.time()
            if up.name.lower().endswith(".csv"):
                first_bytes = up.read(50_000)
                up.seek(0)
                sep = detect_csv_sep(first_bytes)
                df = pd.read_csv(up, sep=sep)
                st.info(f"✅ Определён разделитель CSV: **'{sep}'**")
            else:
                xls = pd.ExcelFile(up)
                sheet = st.selectbox("Выберите лист Excel", options=xls.sheet_names)
                df = pd.read_excel(xls, sheet_name=sheet)
            
            st.session_state.timer_info = {"load_ms": int((time.time() - t0) * 1000)}
            st.success(f"✅ Загружено: {df.shape[0]} строк × {df.shape[1]} столбцов за {human_time_ms(st.session_state.timer_info['load_ms'])}")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Сброс состояния при новой загрузке
            keys_to_reset = [
                'target', 'task_type', 'train_X', 'train_y', 'X_train', 'X_test',
                'y_train', 'y_test', 'leaderboard', 'active_model_name', 'best_estimator',
                'fitted_pipe', 'prediction_data', 'primary_metric', 'selected_features',
                'available_features', 'calculator_base_data', 'dt_cols_hint'
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    st.session_state.pop(key, None)
            
            st.session_state.train_df = df
            data_loaded = True
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {e}")
    
    if st.button("Далее ➜", type="primary", disabled=not data_loaded):
        st.session_state.wizard_step = 2
        st.rerun()


def render_step2_setup():
    """Отрисовывает UI для настройки задачи."""
    st.header("Шаг 2. Настройка задачи: Цель и признаки")
    df = st.session_state.get("train_df")
    if df is None:
        st.warning("⚠️ Сначала загрузите датасет на шаге 1.")
        if st.button("⬅ Назад на Шаг 1"):
            st.session_state.wizard_step = 1
            st.rerun()
        return

    cols = list(df.columns)
    
    st.markdown("#### 🎯 1. Выберите целевой столбец (target)")
    st.markdown("Это то, что модель будет предсказывать.")
    
    current_target = st.session_state.get('target')
    target_index = cols.index(current_target) + 1 if current_target in cols else 0
    target = st.selectbox("Целевой столбец", options=["—"] + cols, index=target_index)

    if target == "—":
        st.info("Выберите целевой столбец, чтобы продолжить.")
        return
        
    st.session_state.target = target
    try:
        task_type = ml_core.detect_problem_type(df[target])
        st.session_state.task_type = task_type
        st.session_state.primary_metric = ml_core.get_primary_metric(task_type)
        st.success(f"Определён тип задачи: **{task_type}**. Основная метрика: **{st.session_state.primary_metric}**")
    except ValueError as e:
        st.error(f"❌ Ошибка анализа целевой переменной: {e}")
        return

    st.markdown("#### 📋 2. Выберите признаки и столбцы с датами")
    st.markdown("Исключите ненужные столбцы и укажите, где находятся даты для автоматического создания признаков (год, месяц и т.д.).")

    available_features = [col for col in df.columns if col != target]
    st.session_state.available_features = available_features
    
    with st.form("features_and_dates_form"):
        selected_features = st.multiselect(
            "Признаки для обучения (оставьте пустым, чтобы выбрать все):",
            options=available_features,
            default=st.session_state.get('selected_features', [])
        )
        
        features_to_check_for_dates = selected_features or available_features
        potential_dt_cols = [
            c for c in features_to_check_for_dates 
            if any(k in c.lower() for k in ["date", "дат", "врем", "time"]) or pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        
        dt_cols_hint = st.multiselect(
            "Столбцы, содержащие дату/время (для авто-генерации признаков):",
            options=potential_dt_cols,
            default=st.session_state.get('dt_cols_hint', [])
        )

        submitted = st.form_submit_button("✅ Применить и перейти к обучению", type="primary")
        if submitted:
            st.session_state.selected_features = selected_features
            st.session_state.dt_cols_hint = dt_cols_hint
            st.session_state.wizard_step = 3
            st.rerun()

def render_step3_training():
    """Отрисовывает UI для обучения моделей."""
    st.header("Шаг 3. Обучение и сравнение моделей")
    
    if 'train_df' not in st.session_state or 'target' not in st.session_state:
        st.warning("⚠️ Пожалуйста, завершите шаги 1 и 2.")
        return

    df = st.session_state.train_df
    target_col = st.session_state.target
    task = st.session_state.task_type
    
    features = st.session_state.get('selected_features') or st.session_state.get('available_features', [])
    X = df[features]
    y = df[target_col].dropna()
    X = X.loc[y.index]

    if 'X_train' not in st.session_state:
         stratify = y if task in ('binary', 'multiclass') else None
         st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
             X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
         )

    if st.button("🚀 Запустить обучение", type="primary"):
        models = ml_core.get_models(task)
        if not models:
            st.error("❌ Не найдено доступных моделей. Установите библиотеки: xgboost, lightgbm, catboost.")
            return

        dt_cols_hint = st.session_state.get('dt_cols_hint')
        pre_unscaled = ml_core.build_preprocessor(X, dt_cols_hint, use_scaler=False, handle_outliers=True)
        pre_scaled = ml_core.build_preprocessor(X, dt_cols_hint, use_scaler=True, handle_outliers=True)
        
        results = []
        progress_bar = st.progress(0, "Инициализация обучения...")
        status_text = st.empty()
        
        t0_all = time.time()
        for i, (name, model) in enumerate(models.items()):
            status_text.info(f"⏳ Обучение модели {i+1}/{len(models)}: {name}")
            preprocessor = pre_scaled if ml_core.is_linear_model(name) else pre_unscaled
            
            scores, duration = ml_core.cv_evaluate(preprocessor, model, X, y, task, n_splits=5, shuffle=True, seed=RANDOM_SEED)
            
            row = {"model": name, "cv_time": human_time_ms(duration), **scores}
            results.append(row)
            progress_bar.progress((i + 1) / len(models), f"Завершено: {name}")

        status_text.empty()
        progress_bar.empty()
        enforce_min_duration(t0_all, min_seconds=4.0)

        leaderboard = pd.DataFrame(results)
        primary_metric = st.session_state.primary_metric
        sort_metric = ml_core.choose_sort_metric(leaderboard, task, primary_metric)
        
        if sort_metric:
            ascending = ml_core.metric_ascending(sort_metric)
            leaderboard = leaderboard.sort_values(by=sort_metric, ascending=ascending).reset_index(drop=True)
            
        st.session_state.leaderboard = leaderboard
        best_model_name = ml_core.select_best_model(leaderboard, task, sort_metric)
        st.session_state.active_model_name = best_model_name
        st.success(f"✅ Обучение завершено! Лучшая модель по метрике '{sort_metric}': **{best_model_name}**")
        st.rerun()

    if 'leaderboard' in st.session_state:
        st.subheader("🏆 Лидерборд моделей")
        st.dataframe(st.session_state.leaderboard.style.format(precision=4), use_container_width=True)
        
        st.subheader("🃏 Выбор активной модели для анализа")
        model_names = st.session_state.leaderboard['model'].tolist()
        active_model_idx = model_names.index(st.session_state.active_model_name) if st.session_state.active_model_name in model_names else 0
        
        new_active_model = st.selectbox("Активная модель", model_names, index=active_model_idx)
        st.session_state.active_model_name = new_active_model

        if st.button(f"✅ Обучить '{new_active_model}' на всех данных и перейти к анализу", type="primary"):
            with st.spinner(f"Обучение '{new_active_model}' на полном датасете..."):
                model_instance = ml_core.get_models(task)[new_active_model]
                preprocessor = ml_core.build_preprocessor(
                    st.session_state.X_train, st.session_state.get('dt_cols_hint'), 
                    use_scaler=ml_core.is_linear_model(new_active_model), handle_outliers=True
                )
                st.session_state.fitted_pipe = ml_core.fit_best(preprocessor, model_instance, X, y)
            
            st.success(f"Модель '{new_active_model}' обучена!")
            st.session_state.wizard_step = 4
            st.rerun()

def render_step4_analysis():
    """Отрисовывает UI для анализа модели."""
    st.header(f"Шаг 4. Анализ модели: {st.session_state.get('active_model_name', '')}")

    if 'fitted_pipe' not in st.session_state:
        st.warning("⚠️ Сначала обучите модель на шаге 3.")
        return

    est = st.session_state.fitted_pipe
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    task = st.session_state.task_type
    y_pred = est.predict(X_test)

    if task == 'regression':
        rmse = math.sqrt(ml_core.mean_squared_error(y_test, y_pred))
        mae = ml_core.mean_absolute_error(y_test, y_pred)
        r2 = ml_core.r2_score(y_test, y_pred)
        st.subheader("📈 Метрики качества на отложенной выборке")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:,.2f}")
        c2.metric("MAE", f"{mae:,.2f}")
        c3.metric("R²", f"{r2:.3f}")
    else: # Classification
        st.subheader("📈 Метрики качества на отложенной выборке")
        acc = ml_core.accuracy_score(y_test, y_pred)
        f1 = ml_core.f1_score(y_test, y_pred, average='weighted')
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("F1-weighted", f"{f1:.3f}")
        
        st.subheader("🧩 Матрица ошибок")
        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Предсказания", y="Истинные значения", color="Кол-во"),
                        x=labels, y=labels, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Важность признаков (Permutation Importance)")
    with st.spinner("Расчет важности признаков..."):
        result = permutation_importance(
            est, X_test, y_test, n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1
        )
        importances = pd.DataFrame({
            'feature': X_test.columns,
            'importance_mean': result.importances_mean
        }).sort_values('importance_mean', ascending=True).tail(20)
        
        fig = px.bar(importances, x='importance_mean', y='feature', orientation='h', title="Топ-20 самых важных признаков")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Далее к Прогнозу ➜", type="primary"):
        st.session_state.wizard_step = 5
        st.rerun()

def render_step5_predict():
    """Отрисовывает UI для прогноза на новых данных."""
    st.header("Шаг 5. Прогноз на новых данных")
    if 'fitted_pipe' not in st.session_state:
        st.warning("⚠️ Сначала обучите модель на шаге 3.")
        return

    up = st.file_uploader("Загрузите файл для прогноза", type=["csv", "xls", "xlsx"])

    if up:
        try:
            # Определяем, какой файл загружен
            if up.name.lower().endswith('.csv'):
                df_new = pd.read_csv(up)
            else:
                df_new = pd.read_excel(up)

            est = st.session_state.fitted_pipe
            original_features = st.session_state.X_train.columns
            
            # Проверяем недостающие колонки
            missing_cols = set(original_features) - set(df_new.columns)
            if missing_cols:
                st.error(f"В загруженном файле отсутствуют необходимые столбцы: {list(missing_cols)}")
                return
            
            # Гарантируем правильный порядок колонок
            df_to_predict = df_new[original_features]

            with st.spinner("Выполняется предсказание..."):
                predictions = est.predict(df_to_predict)

            result_df = df_new.copy()
            result_df[f"prediction_{st.session_state.target}"] = predictions
            st.session_state.prediction_data = result_df
            
            st.success("✅ Прогноз готов!")
            st.dataframe(result_df.head(), use_container_width=True)
            
            # --- ИСПРАВЛЕНО: Возвращена логика для скачивания XLSX ---
            st.subheader("📥 Скачать результаты")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_bytes = result_df.to_csv(index=False, decimal=',', sep=';').encode('utf-8-sig')
                download_button(csv_bytes, "predictions.csv", "Скачать в формате CSV")

            with col2:
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Predictions')
                    xlsx_bytes = output.getvalue()
                    download_button(xlsx_bytes, "predictions.xlsx", "Скачать в формате XLSX")
                except ImportError:
                    st.caption("Для скачивания в XLSX установите: pip install xlsxwriter")
                except Exception as e:
                    st.caption(f"Ошибка при создании XLSX: {e}")

        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {e}")

    if st.button("Далее к Калькулятору ➜", type="primary"):
        st.session_state.wizard_step = 6
        st.rerun()

def benchmark_and_estimate_time(est, base_row, bounds, cat_choices, popsize, maxiter):
    """Проводит микро-бенчмарк и оценивает общее время оптимизации."""
    with st.spinner("Провожу быстрый тест производительности..."):
        timings = []
        n_samples = 20
        for _ in range(n_samples):
            row_to_predict = base_row.copy()
            for f, (lo, hi) in bounds.items():
                row_to_predict[f] = np.random.uniform(lo, hi)
            for f, choices in cat_choices.items():
                row_to_predict[f] = np.random.choice(choices)
            
            start_time = time.perf_counter()
            _ = est.predict(row_to_predict)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
        
        avg_time_per_eval_ms = (sum(timings) / n_samples) * 1000
        total_evals = popsize * maxiter
        total_time_sec = (total_evals * avg_time_per_eval_ms) / 1000

        if total_time_sec < 2:
            estimate_str = "менее пары секунд."
        elif total_time_sec < 60:
            estimate_str = f"примерно {int(total_time_sec)} секунд."
        else:
            minutes = total_time_sec / 60
            estimate_str = f"примерно {minutes:.1f} минут."

        st.session_state.time_estimation = f"ℹ️ Примерное время расчета: **{estimate_str}**"

def render_step6_calculator():
    """Отрисовывает UI для калькулятора 'Что, если?' с продвинутой оптимизацией."""
    st.header("Шаг 6. Калькулятор оптимизации 'Что, если?'")
    
    if 'fitted_pipe' not in st.session_state:
        st.warning("⚠️ Сначала обучите модель.")
        return

    est = st.session_state.fitted_pipe
    X_train = st.session_state.X_train
    task = st.session_state.task_type

    st.subheader("1. Выберите базовую строку для анализа")
    idx = st.number_input("Номер строки в обучающих данных", min_value=0, max_value=len(X_train)-1, value=0)
    base_row = X_train.iloc[[idx]]
    st.dataframe(base_row)

    st.subheader("2. Настройте параметры для оптимизации")
    
    all_features = base_row.columns.tolist()
    num_features = [f for f in all_features if pd.api.types.is_numeric_dtype(X_train[f])]
    cat_features = [f for f in all_features if not pd.api.types.is_numeric_dtype(X_train[f])]

    st.markdown("**Числовые признаки**")
    bounds = {}
    for f in num_features:
        min_val, max_val = float(X_train[f].min()), float(X_train[f].max())
        bounds[f] = st.slider(f"Диапазон для '{f}'", min_val, max_val, (min_val, max_val))

    st.markdown("**Категориальные признаки** (отметьте те, что можно менять)")
    cat_choices = {}
    for f in cat_features:
        options = sorted(list(X_train[f].dropna().unique()))
        if st.checkbox(f"Разрешить изменение '{f}'", key=f"check_{f}"):
            cat_choices[f] = options

    if task == 'regression':
        objective = st.radio("Цель", ["Максимизировать", "Минимизировать"], horizontal=True)
    else:
        objective = "Максимизировать"
    
    popsize_help_text = """
    Это как количество "разведчиков", которых алгоритм отправляет на поиски лучшего решения.
    - **Маленькое значение (5-10):** Быстрый, но менее тщательный поиск. Может найти "хорошее", но не "лучшее" решение.
    - **Большое значение (20+):** Более медленный, но более надежный и глубокий поиск. Повышает шансы найти наилучшую комбинацию.
    """
    popsize = st.slider("Размер популяции (больше = качественнее, но дольше)", 5, 50, 15, help=popsize_help_text)
    maxiter = st.slider("Макс. итераций (больше = качественнее, но дольше)", 10, 200, 50)

    if 'time_estimation' in st.session_state:
        st.info(st.session_state.time_estimation)

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("⏱️ Оценить время"):
            benchmark_and_estimate_time(est, base_row, bounds, cat_choices, popsize, maxiter)
            st.rerun()
    with col2:
        if st.button("🚀 Запустить оптимизацию", type="primary"):
            with st.spinner("Идет подбор оптимальных параметров... Это может занять несколько минут."):
                if 'time_estimation' in st.session_state:
                    del st.session_state['time_estimation']
                
                optimizable_num = list(bounds.keys())
                optimizable_cat = list(cat_choices.keys())
                
                optimizer_bounds = [bounds[f] for f in optimizable_num]
                for f in optimizable_cat:
                    optimizer_bounds.append((0, len(cat_choices[f]) - 1e-9))

                def objective_function(x):
                    row_to_predict = base_row.copy()
                    for i, f in enumerate(optimizable_num):
                        row_to_predict[f] = x[i]
                    offset = len(optimizable_num)
                    for i, f in enumerate(optimizable_cat):
                        choice_idx = int(x[offset + i])
                        row_to_predict[f] = cat_choices[f][choice_idx]

                    if task == 'regression':
                        score = est.predict(row_to_predict)[0]
                    else:
                        score = est.predict_proba(row_to_predict)[0, 1]
                    
                    return -score if objective == "Максимизировать" else score

                result = differential_evolution(
                    objective_function, bounds=optimizer_bounds,
                    popsize=popsize, maxiter=maxiter, seed=RANDOM_SEED
                )

                st.success("✅ Оптимизация завершена!")
                
                if task == 'regression':
                    base_pred = est.predict(base_row)[0]
                else:
                    base_pred = est.predict_proba(base_row)[0, 1]
                
                opt_pred = -result.fun if objective == "Максимизировать" else result.fun

                st.subheader("📈 Результат")
                c1, c2, c3 = st.columns(3)
                c1.metric("Базовый прогноз", f"{base_pred:.4f}")
                c2.metric("Оптимальный прогноз", f"{opt_pred:.4f}")
                c3.metric("Прирост", f"{opt_pred - base_pred:.4f}", delta=f"{opt_pred - base_pred:.4f}")
                
                optimal_row = base_row.copy()
                x_opt = result.x
                for i, f in enumerate(optimizable_num):
                    optimal_row[f] = x_opt[i]
                offset = len(optimizable_num)
                for i, f in enumerate(optimizable_cat):
                    choice_idx = int(x_opt[offset + i])
                    optimal_row[f] = cat_choices[f][choice_idx]
                
                st.subheader("🔧 Измененные параметры")
                comparison_df = pd.concat([base_row, optimal_row])
                comparison_df.index = ["Базовая строка", "Оптимальная строка"]
                
                changed_cols = [col for col in all_features if str(comparison_df[col].iloc[0]) != str(comparison_df[col].iloc[1])]
                
                if changed_cols:
                    st.dataframe(comparison_df[changed_cols].T.style.format(precision=3))
                else:
                    st.info("Оптимальные параметры совпадают с базовыми. Попробуйте изменить диапазоны.")
    
    st.markdown("---")
    if st.button("🏠 Начать новый анализ (на главную)"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
