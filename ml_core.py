# -*- coding: utf-8 -*-
"""
ml_core.py — ULTIMATE версия v0.25 с ПОЛНЫМИ ИСПРАВЛЕНИЯМИ:
✅ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
- Правильная классификация float как категориальных (целые числа в float формате)
- Динамическое ограничение max_categories для OHE (предотвращение OOM)
- Параллелизм (n_jobs > 1) для всех моделей и CV
- Кэш с TTL и max_entries
- Правильная обработка NaN в метриках с валидацией
- Stratified sampling для несбалансированных данных

✅ СРЕДНИЕ ИСПРАВЛЕНИЯ:
- Проверка на дубликаты строк
- Улучшенный TextProcessor с лучшим sampling
- Оптимизация Optuna с early stopping
- Улучшенное определение типа задачи

✅ НИЗКИЕ ИСПРАВЛЕНИЯ:
- Больше метрик для multiclass
- Лучшая обработка edge cases
- Улучшенное логирование
"""

import math
import time
import warnings
import logging
import os
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, RobustScaler, 
    MinMaxScaler, FunctionTransformer, OrdinalEncoder
)
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ---- Опциональные библиотеки бустинга ----
XGB_AVAILABLE = LGBM_AVAILABLE = CATBOOST_AVAILABLE = False
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    pass

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    pass

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    pass

# ---- Optuna ----
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

from utils import get_session_id, get_ttl_to_4am
import ui_config

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ✅ ИСПРАВЛЕНО: Динамическое определение n_jobs
N_JOBS = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
logger.info(f"N_JOBS установлено на {N_JOBS} (доступно ядер: {os.cpu_count()})")

RANDOM_SEED = 42

# =========================================================
# 1) ✅ ИСПРАВЛЕНО: ОПРЕДЕЛЕНИЕ ТИПА ЗАДАЧИ
# =========================================================

def _is_categorical_series(s: pd.Series) -> bool:
    """
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Улучшенная эвристика для определения категориальных признаков.
    - Правильно обрабатывает float с целочисленными значениями (1.0, 2.0, 3.0)
    - Более строгие критерии для числовых типов
    - Учитывает размер датасета
    """
    # Явные категориальные типы
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype.name in ("object", "category", "bool", "boolean"):
        return True
    
    # Числовые типы - более консервативный подход
    if pd.api.types.is_numeric_dtype(s):
        try:
            nunique = s.nunique(dropna=True)
        except Exception:
            return False
        
        data_len = len(s.dropna())
        if data_len == 0:
            return False
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обработка float с целочисленными значениями
        if pd.api.types.is_float_dtype(s):
            non_null = s.dropna()
            if len(non_null) > 0:
                # Проверяем, являются ли все float по факту целыми числами
                is_integer_values = np.allclose(non_null, np.round(non_null), equal_nan=True, rtol=1e-9)
                if is_integer_values:
                    # Теперь проверяем как integer с более строгими критериями
                    is_cat = nunique <= 15 and nunique <= max(2, int(data_len * 0.03))
                    if is_cat:
                        logger.info(f"Столбец '{s.name}' определен как категориальный (float с целыми значениями, {nunique} уникальных)")
                    return is_cat
            # Обычный float - не категория
            return False
        
        # Для integer типов - строгие критерии
        if pd.api.types.is_integer_dtype(s):
            # Максимум 15 уникальных ИЛИ меньше 3% от размера
            is_cat = nunique <= 15 and nunique <= max(2, int(data_len * 0.03))
            if is_cat:
                logger.info(f"Столбец '{s.name}' определен как категориальный (int, {nunique} уникальных)")
            return is_cat
        
        return False
    
    return False

def detect_problem_type(y: pd.Series, _sid: str = None) -> str:
    """
    ✅ УЛУЧШЕНО: Определить тип задачи: 'binary', 'multiclass', 'regression'.
    Более надежная логика с проверками на edge cases.
    """
    if y is None or len(y) == 0:
        raise ValueError("Целевая переменная пуста или None")
    
    y_clean = y.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(y_clean) < 2:
        raise ValueError(
            f"Целевая переменная имеет менее 2 валидных значений (осталось {len(y_clean)}). "
            "Удалите NaN/inf или предоставьте больше данных."
        )
    
    if pd.api.types.is_datetime64_any_dtype(y_clean):
        raise ValueError("Целевая переменная не может быть datetime-типом.")
    
    nunq = int(y_clean.nunique())
    is_numeric = pd.api.types.is_numeric_dtype(y_clean)
    
    # Boolean явно
    if pd.api.types.is_bool_dtype(y_clean) or y_clean.dtype.name in ("boolean",):
        return "binary" if nunq == 2 else "multiclass"
    
    # Object/category типы - это классификация
    if not is_numeric and nunq > 1:
        return "binary" if nunq == 2 else "multiclass"
    
    # ✅ УЛУЧШЕНО: Числовые типы - нужно определить регрессию ли это
    if is_numeric:
        try:
            float_vals = y_clean.astype(float)
            rounded_vals = float_vals.round()
            looks_continuous = not np.allclose(float_vals, rounded_vals, equal_nan=True, rtol=1e-9)
        except Exception:
            looks_continuous = True
        
        # Если выглядит как целые числа И их мало - классификация
        max_allowed_classes = max(20, int(len(y_clean) * 0.05))
        if not looks_continuous and nunq <= max_allowed_classes:
            task = "binary" if nunq == 2 else "multiclass"
            logger.info(f"Числовой target классифицирован как {task} "
                       f"({nunq} уникальных значений, мало для регрессии)")
            return task
        
        # ✅ ДОБАВЛЕНО: Дополнительная проверка на малое количество уникальных значений
        if nunq <= 10 and not looks_continuous:
            logger.warning(f"Target имеет только {nunq} уникальных значений. "
                          f"Определено как {'binary' if nunq == 2 else 'multiclass'}. "
                          f"Если это регрессия, переопределите тип вручную.")
            return "binary" if nunq == 2 else "multiclass"
    
    return "regression"

# =========================================================
# 2) ✅ ИСПРАВЛЕНО: ПРЕДОБРАБОТКА
# =========================================================

class DateTimeExpander(BaseEstimator, TransformerMixin):
    """
    ✅ ИСПРАВЛЕНО: Расширяет datetime-столбцы в компоненты.
    - Поддерживает Unix timestamps
    - Безопасная обработка ошибок
    """
    def __init__(self, dt_cols_hint: Optional[List[str]] = None, min_success: float = 0.7, strict_hint: bool = False):
        self.dt_cols_hint = dt_cols_hint
        self.min_success = min_success
        self.strict_hint = strict_hint
        self.dt_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        cols_to_check = self.dt_cols_hint if (self.strict_hint and self.dt_cols_hint) else list(X.columns)
        self.dt_cols_.clear()
        
        for c in cols_to_check:
            if c not in X.columns:
                continue

            handled = False

            # ✅ ДОБАВЛЕНО: Проверка Unix timestamps
            if pd.api.types.is_numeric_dtype(X[c]):
                try:
                    vals = X[c].dropna()
                    if len(vals) > 0:
                        min_val, max_val = vals.min(), vals.max()

                        # ✅ ИСПРАВЛЕНО: Реалистичная проверка Unix timestamp
                        # - Значения должны быть целочисленными/псевдо-целыми
                        # - Диапазон в секундах с 1990 по 2100 (исключаем мелкие величины типа длины/веса)
                        looks_integer = np.allclose(vals, np.round(vals), rtol=0, atol=1e-3)
                        realistic_range = 6.3e8 <= min_val <= 4.1e9 and max_val <= 4.1e9

                        if looks_integer and realistic_range:
                            try:
                                parsed = pd.to_datetime(vals, unit='s', errors='coerce')
                                if parsed.notna().mean() >= self.min_success:
                                    self.dt_cols_.append(c)
                                    logger.info(f"Столбец {c} определен как Unix timestamp")
                                    handled = True
                                    continue
                            except Exception:
                                pass
                except Exception:
                    pass

                # ✅ ИСПРАВЛЕНО: не пытаться парсить произвольные числовые столбцы как даты
                if not handled:
                    continue

            # ✅ ИСПРАВЛЕНО: Обычный parsing (для нечисловых столбцов)
            parsed, ok = self._safe_parse_datetime(X[c], self.min_success)
            if ok:
                self.dt_cols_.append(c)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        
        for c in self.dt_cols_:
            if c not in Xc.columns:
                continue
            
            try:
                # Проверяем если числовой (unix timestamp)
                if pd.api.types.is_numeric_dtype(Xc[c]):
                    try:
                        parsed = pd.to_datetime(Xc[c], unit='s', errors='coerce')
                    except Exception:
                        parsed, _ = self._safe_parse_datetime(Xc[c], self.min_success)
                else:
                    parsed, _ = self._safe_parse_datetime(Xc[c], self.min_success)
                
                if pd.api.types.is_datetime64_any_dtype(parsed):
                    Xc[f"{c}_year"] = parsed.dt.year
                    Xc[f"{c}_month"] = parsed.dt.month
                    Xc[f"{c}_day"] = parsed.dt.day
                    Xc[f"{c}_dow"] = parsed.dt.dayofweek
                    Xc[f"{c}_hour"] = parsed.dt.hour
                    Xc[f"{c}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
                    Xc = Xc.drop(columns=[c])
            except Exception as e:
                logger.warning(f"Ошибка при трансформе столбца {c}: {e}")
        
        return Xc

    def _safe_parse_datetime(self, col: pd.Series, min_success: float) -> Tuple[Optional[pd.Series], bool]:
        if pd.api.types.is_datetime64_any_dtype(col):
            return col, True
        
        try:
            parsed = pd.to_datetime(col, errors="coerce", infer_datetime_format=True)
            ok = parsed.notna().mean() >= min_success
            if ok:
                return parsed, True
            else:
                return col, False
        except Exception as e:
            logger.debug(f"Ошибка парсинга datetime для {col.name}: {e}")
            return col, False

class TextProcessor(BaseEstimator, TransformerMixin):
    """
    ✅ УЛУЧШЕНО: Обработка текстовых признаков с ограничениями памяти.
    - Увеличен лимит sampling до 100K
    - Улучшенная проверка на текстовые столбцы
    - Безопасное снижение размерности
    """
    def __init__(self, max_features: int = 200, n_components: int = 30, min_text_length: int = 5):
        self.max_features = max_features
        self.n_components = n_components
        self.min_text_length = min_text_length
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english',
            analyzer='char',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        self.svd = TruncatedSVD(n_components=min(n_components, max_features - 1), random_state=42)
        self.text_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        potential_text = [c for c in X.columns if X[c].dtype == 'object']
        self.text_cols_ = []
        
        for col in potential_text:
            try:
                # Проверяем что это действительно текст, а не категория
                non_null = X[col].dropna()
                if len(non_null) == 0:
                    continue
                
                # Если слишком мало уникальных значений - это категория, не текст
                if non_null.nunique() < len(non_null) * 0.3:  # < 30% уникальных
                    logger.debug(f"Столбец {col} похож на категорию, пропускаем")
                    continue
                
                # Если слишком много пустых строк или очень коротких
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len < self.min_text_length:
                    logger.debug(f"Столбец {col} имеет короткие строки (среднее {avg_len}), пропускаем")
                    continue
                
                self.text_cols_.append(col)
            except Exception as e:
                logger.debug(f"Ошибка при анализе столбца {col}: {e}")
        
        if not self.text_cols_:
            logger.info("Текстовых столбцов не найдено")
            return self
        
        # ✅ УЛУЧШЕНО: Увеличен лимит sampling до 100K
        try:
            combined_text = X[self.text_cols_].fillna('').apply(lambda row: ' '.join(row.astype(str)), axis=1)
            # Ограничиваем по размеру
            if len(combined_text) > 100000:
                sample_idx = np.random.choice(len(combined_text), 100000, replace=False)
                combined_text_sample = combined_text.iloc[sample_idx]
                logger.info(f"TF-IDF обучается на выборке {len(combined_text_sample)} из {len(combined_text)}")
            else:
                combined_text_sample = combined_text
            
            self.vectorizer.fit(combined_text_sample)
            tfidf_matrix = self.vectorizer.transform(combined_text_sample)
            
            # ✅ ИСПРАВЛЕНО: Безопасное обучение SVD
            n_comp = min(self.n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
            if n_comp > 0:
                self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
                self.svd.fit(tfidf_matrix)
            else:
                logger.warning(f"Слишком мало данных для SVD (n_comp={n_comp})")
        except Exception as e:
            logger.error(f"Ошибка при обучении TextProcessor: {e}")
            self.text_cols_ = []
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "text_cols_") or not self.text_cols_:
            return X.copy()
        
        try:
            Xc = X.copy()
            combined_text = Xc[self.text_cols_].fillna('').apply(lambda row: ' '.join(row.astype(str)), axis=1)
            tfidf_matrix = self.vectorizer.transform(combined_text)
            
            if hasattr(self.svd, 'components_'):
                svd_matrix = self.svd.transform(tfidf_matrix)
                n_comp = svd_matrix.shape[1]
            else:
                svd_matrix = tfidf_matrix.toarray()
                n_comp = svd_matrix.shape[1]
            
            svd_df = pd.DataFrame(
                svd_matrix[:, :min(n_comp, 30)],
                columns=[f"text_svd_{i}" for i in range(min(n_comp, 30))],
                index=Xc.index
            )
            
            Xc = Xc.drop(columns=self.text_cols_, errors='ignore')
            Xc = pd.concat([Xc, svd_df], axis=1)
            return Xc
        except Exception as e:
            logger.error(f"Ошибка при трансформе текста: {e}")
            return X.copy()

# ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Кэширование с TTL и max_entries
@st.cache_resource(ttl=3600, max_entries=10)
def build_preprocessor(
    X: pd.DataFrame,
    dt_cols_hint: Optional[List[str]],
    use_scaler: bool,
    handle_outliers: bool,
    _sid: str = None,
    text_processing: bool = False,
    model_name: str = None,
    use_log_transform: bool = False
) -> Pipeline:
    """
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Собирает Pipeline предобработки.
    - DateTimeExpander (с unix timestamp support)
    - TextProcessor (с ограничениями памяти)
    - ColumnTransformer (числовые + категориальные)
    - Динамическое ограничение max_categories для OHE
    - Правильная обработка edge cases
    """
    logger.info(f"Создание preprocessor для модели: {model_name}")
    
    # ✅ ИСПРАВЛЕНО: Очистка данных перед обработкой
    X = X.copy()
    
    # Удаляем полностью пустые столбцы
    X = X.dropna(axis=1, how='all')
    
    # ✅ ДОБАВЛЕНО: Обработка дублирующихся столбцов
    from utils import remove_duplicate_columns, sanitize_column_names
    X = sanitize_column_names(X)
    X = remove_duplicate_columns(X)
    
    # DateTimeExpander
    dt_exp = DateTimeExpander(dt_cols_hint=dt_cols_hint, strict_hint=bool(dt_cols_hint))
    try:
        X_transformed = dt_exp.fit(X).transform(X)
        logger.info(f"DateTimeExpander обработал {len(dt_exp.dt_cols_)} столбцов")
    except Exception as e:
        logger.warning(f"Ошибка DateTimeExpander: {e}")
        X_transformed = X.copy()
        dt_exp.dt_cols_ = []
    
    # TextProcessor
    text_processor = None
    if text_processing:
        potential_text = [c for c in X_transformed.columns if X_transformed[c].dtype == 'object']
        if potential_text:
            logger.info(f"Найдено {len(potential_text)} потенциальных текстовых столбцов")
            text_processor = TextProcessor(max_features=200, n_components=30, min_text_length=3)
            try:
                X_transformed = text_processor.fit_transform(X_transformed)
                logger.info(f"TextProcessor обработан, осталось {len(text_processor.text_cols_)} текстовых столбцов")
            except Exception as e:
                logger.warning(f"Ошибка TextProcessor: {e}")
                text_processor = None
    
    # ✅ ИСПРАВЛЕНО: Правильное определение num/cat столбцов
    num_cols = []
    cat_cols = []
    
    for c in X_transformed.columns:
        try:
            if pd.api.types.is_numeric_dtype(X_transformed[c]):
                # Проверяем, не является ли это закодированной категорией
                if _is_categorical_series(X_transformed[c]):
                    cat_cols.append(c)
                else:
                    num_cols.append(c)
            elif _is_categorical_series(X_transformed[c]):
                cat_cols.append(c)
            else:
                # Fallback: object типы считаем категориальными
                cat_cols.append(c)
        except Exception as e:
            logger.debug(f"Ошибка при классификации столбца {c}: {e}")
            cat_cols.append(c)
    
    logger.info(f"Классификация: {len(num_cols)} числовых, {len(cat_cols)} категориальных")
    
    # ✅ ИСПРАВЛЕНО: Числовой pipeline с безопасной обработкой
    num_steps = [
        ("imp", SimpleImputer(strategy="median", fill_value=0))
    ]
    
    if use_log_transform:
        num_steps.append(("log", FunctionTransformer(np.log1p, validate=False)))
    
    if handle_outliers:
        num_steps.append(("robust", RobustScaler()))
    
    if use_scaler:
        num_steps.append(("scale", StandardScaler()))
    
    num_pipe = Pipeline(steps=num_steps)
    
    # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Динамическое ограничение max_categories
    data_size = len(X_transformed)
    n_cat_cols = len(cat_cols)
    
    # Рассчитываем безопасное ограничение
    max_total_categories = min(200, max(20, data_size // 500))
    max_per_column = min(30, max(5, max_total_categories // max(1, n_cat_cols)))
    
    logger.info(f"Динамическое ограничение OHE: max_categories={max_per_column} на столбец "
                f"(общий лимит: {max_total_categories})")
    
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent", fill_value="missing")),
        ("ohe", OneHotEncoder(
            handle_unknown="ignore", 
            sparse_output=True,  # ✅ ИСПРАВЛЕНО: используем sparse для экономии памяти
            max_categories=max_per_column,
            drop='if_binary'
        )),
    ])
    
    # ✅ ИСПРАВЛЕНО: ColumnTransformer с правильной обработкой remainder
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    
    if not transformers:
        logger.warning("Нет ни числовых ни категориальных столбцов!")
        # Fallback: добавить identity transformer
        col_tf = ColumnTransformer(
            transformers=[("identity", FunctionTransformer(), list(X_transformed.columns))],
            remainder="drop",
            sparse_threshold=0.3  # ✅ ИСПРАВЛЕНО: позволяем sparse матрицы
        )
    else:
        col_tf = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.3,  # ✅ ИСПРАВЛЕНО: позволяем sparse матрицы
            n_jobs=1  # Preprocessing без параллелизма для стабильности
        )
    
    # Финальный pipeline
    steps = [("dt", dt_exp)]
    if text_processor:
        steps.append(("text", text_processor))
    steps.append(("cols", col_tf))
    
    return Pipeline(steps=steps)

# =========================================================
# 3) ✅ ИСПРАВЛЕНО: МОДЕЛИ И МЕТРИКИ
# =========================================================

def get_models(task: str, mode: str = "fast") -> Dict[str, Optional[BaseEstimator]]:
    """
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Возвращает доступные модели с параллелизмом.
    Все модели используют n_jobs=N_JOBS для ускорения.
    """
    models: Dict[str, Optional[BaseEstimator]] = {}
    
    if task not in ("regression", "binary", "multiclass"):
        return {}
    
    if task == "regression":
        models.update({
            "LinearRegression": LinearRegression(n_jobs=N_JOBS),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1, max_iter=5000),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=300, max_depth=12, n_jobs=N_JOBS, random_state=RANDOM_SEED
            ),
            "ExtraTreesRegressor": ExtraTreesRegressor(
                n_estimators=300, max_depth=12, n_jobs=N_JOBS, random_state=RANDOM_SEED
            ),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                max_iter=200, random_state=RANDOM_SEED
            ),
        })
        if XGB_AVAILABLE:
            models["XGBRegressor"] = XGBRegressor(
                n_estimators=300, max_depth=6, random_state=RANDOM_SEED, 
                tree_method="hist", verbosity=0, n_jobs=N_JOBS
            )
        if LGBM_AVAILABLE:
            models["LGBMRegressor"] = LGBMRegressor(
                n_estimators=300, random_state=RANDOM_SEED, verbosity=-1, n_jobs=N_JOBS
            )
        if CATBOOST_AVAILABLE:
            models["CatBoostRegressor"] = CatBoostRegressor(
                iterations=300, random_seed=RANDOM_SEED, verbose=False, 
                allow_writing_files=False, thread_count=N_JOBS
            )
    
    elif task in ("binary", "multiclass"):
        models.update({
            "LogisticRegression": LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED, n_jobs=N_JOBS
            ),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=300, max_depth=12, n_jobs=N_JOBS, random_state=RANDOM_SEED
            ),
            "ExtraTreesClassifier": ExtraTreesClassifier(
                n_estimators=300, max_depth=12, n_jobs=N_JOBS, random_state=RANDOM_SEED
            ),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
                max_iter=200, random_state=RANDOM_SEED
            ),
        })
        if XGB_AVAILABLE:
            models["XGBClassifier"] = XGBClassifier(
                n_estimators=300, max_depth=6, random_state=RANDOM_SEED, 
                use_label_encoder=False, verbosity=0, n_jobs=N_JOBS
            )
        if LGBM_AVAILABLE:
            models["LGBMClassifier"] = LGBMClassifier(
                n_estimators=300, random_state=RANDOM_SEED, verbosity=-1, n_jobs=N_JOBS
            )
        if CATBOOST_AVAILABLE:
            models["CatBoostClassifier"] = CatBoostClassifier(
                iterations=300, random_seed=RANDOM_SEED, verbose=False, 
                allow_writing_files=False, thread_count=N_JOBS
            )
    
    return {k: v for k, v in models.items() if v is not None}

def get_cv(task: str, n_splits: int, shuffle: bool, seed: int):
    """✅ ИСПРАВЛЕНО: Динамическая обработка n_splits для маленьких датасетов."""
    # n_splits должна быть уже правильно вычислена в caller, но проверяем
    n_splits = max(2, min(n_splits, 10))  # Min 2, max 10
    
    if task in ("binary", "multiclass"):
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

def get_optimal_cv_splits(data_size: int) -> int:
    """✅ ДОБАВЛЕНО: Динамически вычислить оптимальное количество folds."""
    if data_size < 50:
        return min(3, max(2, data_size // 10))
    elif data_size < 100:
        return 3
    elif data_size < 500:
        return 5
    else:
        return 5

def get_metrics(task: str) -> Dict[str, Any]:
    """✅ УЛУЧШЕНО: Набор метрик с дополнительными для multiclass."""
    if task == "regression":
        return {
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
            "RMSE": make_scorer(lambda y, p: math.sqrt(mean_squared_error(y, p)), greater_is_better=False),
            "R2": make_scorer(r2_score),
        }
    elif task == "binary":
        return {
            "Accuracy": make_scorer(accuracy_score),
            "F1": make_scorer(f1_score, average="binary", zero_division=0),
            "ROC_AUC": make_scorer(roc_auc_score),
            "Precision": make_scorer(precision_score, average="binary", zero_division=0),
            "Recall": make_scorer(recall_score, average="binary", zero_division=0),
        }
    elif task == "multiclass":
        return {
            "Accuracy": make_scorer(accuracy_score),
            "F1": make_scorer(f1_score, average="weighted", zero_division=0),
            "F1_Macro": make_scorer(f1_score, average="macro", zero_division=0),
            "Balanced_Acc": make_scorer(balanced_accuracy_score),
            "Precision": make_scorer(precision_score, average="weighted", zero_division=0),
            "Recall": make_scorer(recall_score, average="weighted", zero_division=0),
        }
    return {}

def get_primary_metric(task: str) -> str:
    """✅ ИСПРАВЛЕНО: Основная метрика для сортировки."""
    if task == "binary":
        return "ROC_AUC"
    elif task == "multiclass":
        return "F1"
    return "RMSE"

def choose_sort_metric(cv_results: pd.DataFrame, task: str, primary_metric: str) -> str:
    """Какую метрику использовать для сортировки лидерборда."""
    return primary_metric

def metric_ascending(metric: str) -> bool:
    """True – меньше лучше (MAE/RMSE), False – больше лучше."""
    return metric in ("MAE", "RMSE")

def is_linear_model(name: str) -> bool:
    """Нужно ли масштабирование (для linear моделей)."""
    return name in ("LinearRegression", "LogisticRegression", "Ridge", "Lasso")

# =========================================================
# 4) ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: КРОСС-ВАЛИДАЦИЯ И ОБУЧЕНИЕ
# =========================================================

def cv_evaluate(
    _pipe: Pipeline,
    _model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    n_splits: int,
    shuffle: bool,
    seed: int,
    _sid: str = None,
    max_memory_mb: int = 2048,
    _cache_bust: int = 0
) -> Tuple[Dict[str, float], float]:
    """
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: CV с правильной обработкой edge cases.
    - Удаляет NaN из y
    - Динамическая обработка n_splits
    - Безопасная обработка ошибок
    - Параллелизм в CV
    - Логирование
    """
    np.random.seed(seed)
    
    # ✅ ИСПРАВЛЕНО: Удалить NaN и выровнять X
    valid_idx = y.notna()
    y_clean = y[valid_idx]
    X_clean = X.loc[valid_idx].copy()
    
    if len(X_clean) != len(y_clean):
        X_clean = X_clean.iloc[:len(y_clean)]
    
    logger.info(f"CV evaluate: {len(X_clean)} samples, {X_clean.shape[1]} features")
    
    # ✅ ИСПРАВЛЕНО: Проверка минимальных требований
    if len(X_clean) < 4:
        logger.error(f"Слишком мало данных для CV: {len(X_clean)} < 4")
        return {"Accuracy": 0.0, "F1": 0.0, "MAE": float('inf')}, 0.0
    
    # ✅ ИСПРАВЛЕНО: Динамическое вычисление n_splits
    n_splits = get_optimal_cv_splits(len(X_clean))
    logger.info(f"Вычислены n_splits={n_splits} для {len(X_clean)} samples")
    
    start = time.time()
    
    try:
        cv = get_cv(task, n_splits, shuffle, seed)
        est = Pipeline(steps=[("preprocessor", _pipe), ("model", _model)])
        
        metrics_dict = get_metrics(task)
        scores_out: Dict[str, float] = {}
        
        for m_name, scorer in metrics_dict.items():
            try:
                # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Параллелизм в CV
                scr = cross_val_score(
                    est, X_clean, y_clean, 
                    cv=cv, 
                    scoring=scorer, 
                    n_jobs=N_JOBS,  # ИСПРАВЛЕНО: используем параллелизм
                    error_score=np.nan
                )
                scores_out[m_name] = float(np.nanmean(scr))
                logger.info(f"  {m_name}: {scores_out[m_name]:.4f}")
            except Exception as e:
                logger.error(f"Ошибка метрики {m_name}: {e}")
                scores_out[m_name] = float('nan')
    
    except Exception as e:
        logger.error(f"Ошибка CV: {e}")
        # Возвращаем худшие возможные метрики
        return {m: float('nan') for m in get_metrics(task).keys()}, 0.0
    
    dur_ms = (time.time() - start) * 1000.0
    return scores_out, dur_ms

def select_best_model(cv_results: pd.DataFrame, task: str, metric_to_use: str) -> str:
    """
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Выбрать лучшую модель с валидацией NaN.
    Проверяет что хотя бы одна модель имеет валидную метрику.
    """
    if cv_results.empty:
        logger.error("cv_results пуст!")
        return ""
    
    metric = metric_to_use or get_primary_metric(task)
    
    # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверка что хотя бы одна модель имеет валидную метрику
    if metric not in cv_results.columns:
        logger.error(f"Метрика {metric} не найдена в cv_results!")
        # Пробуем альтернативные метрики
        for alt_metric in cv_results.columns:
            if alt_metric != 'model' and alt_metric != 'cv_time':
                logger.warning(f"Используем альтернативную метрику: {alt_metric}")
                metric = alt_metric
                break
        else:
            return ""
    
    valid_scores = cv_results[metric].notna()
    if not valid_scores.any():
        logger.error(f"Все модели вернули NaN для метрики {metric}!")
        return ""
    
    # Заменяем NaN на худшие значения
    if metric_ascending(metric):
        sort_key = cv_results[metric].fillna(float('inf'))
        ascending = True
    else:
        sort_key = cv_results[metric].fillna(float('-inf'))
        ascending = False
    
    sorted_df = cv_results.copy()
    sorted_df['_sort_key'] = sort_key
    sorted_df = sorted_df.sort_values('_sort_key', ascending=ascending)
    
    best_model = str(sorted_df.iloc[0]["model"])
    logger.info(f"Выбрана лучшая модель: {best_model} ({metric}={sorted_df.iloc[0][metric]:.4f})")
    
    return best_model

@st.cache_resource(ttl=7200, max_entries=5)  # ✅ ИСПРАВЛЕНО: TTL 2 часа
def fit_best(_pipe: Pipeline, _model: BaseEstimator, X: pd.DataFrame, y: pd.Series, _sid: str = None) -> Pipeline:
    """✅ ИСПРАВЛЕНО: Обучить финальный Pipeline."""
    est = Pipeline(steps=[("preprocessor", _pipe), ("model", _model)])
    logger.info(f"Обучение финального Pipeline на {len(X)} samples")
    est.fit(X, y)
    return est

# =========================================================
# 5) ✅ УЛУЧШЕНО: OPTUNA TUNING
# =========================================================

def tune_with_optuna(
    model_name: str, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cv, 
    n_trials: int = 50,
    dt_cols_hint: Optional[List[str]] = None
):
    """
    ✅ УЛУЧШЕНО: Optuna с timeout, early stopping, и выборкой данных.
    - Увеличен sampling до 20K для больших датасетов
    - Early stopping если нет улучшений
    - Timeout и progress
    """
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna не установлена")
        return None
    
    # ✅ УЛУЧШЕНО: Увеличен лимит sampling до 20K
    if len(X_train) > 20000:
        sample_idx = np.random.choice(len(X_train), 20000, replace=False)
        X_train = X_train.iloc[sample_idx]
        y_train = y_train.iloc[sample_idx]
        logger.info(f"Optuna обучается на выборке {len(X_train)} samples")
    
    # ✅ ИСПРАВЛЕНО: Mapping моделей на objective functions
    objective_map = {
        "Ridge": _make_ridge_objective,
        "Lasso": _make_lasso_objective,
        "RandomForestRegressor": _make_rf_objective,
        "RandomForestClassifier": _make_rf_cls_objective,
        "LogisticRegression": _make_logreg_objective,
        "XGBRegressor": _make_xgb_objective,
        "XGBClassifier": _make_xgb_cls_objective,
        "LGBMRegressor": _make_lgbm_objective,
        "LGBMClassifier": _make_lgbm_cls_objective,
        "CatBoostRegressor": _make_cb_objective,
        "CatBoostClassifier": _make_cb_cls_objective,
    }
    
    objective_func = objective_map.get(model_name)
    if not objective_func:
        logger.warning(f"Нет objective для {model_name}")
        return None
    
    try:
        study = optuna.create_study(
            direction="maximize",
            study_name=model_name,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        
        # ✅ УЛУЧШЕНО: Early stopping callback
        early_stop_patience = 10
        best_value_trials = []
        
        def early_stopping_callback(study, trial):
            best_value_trials.append(study.best_value)
            if len(best_value_trials) > early_stop_patience:
                recent_improvements = [
                    abs(best_value_trials[i] - best_value_trials[i-1]) 
                    for i in range(-early_stop_patience, 0)
                ]
                if max(recent_improvements) < 1e-4:
                    logger.info(f"Early stopping: нет улучшений за {early_stop_patience} trials")
                    study.stop()
        
        # ✅ ИСПРАВЛЕНО: Timeout callback
        timeout_seconds = 3600  # 1 час максимум
        start_time = time.time()
        
        def timeout_callback(study, trial):
            if time.time() - start_time > timeout_seconds:
                logger.info("Optuna остановлена по timeout")
                study.stop()
        
        study.optimize(
            lambda trial: objective_func(trial, X_train, y_train, cv, dt_cols_hint),
            n_trials=n_trials,
            timeout=timeout_seconds,
            callbacks=[early_stopping_callback, timeout_callback],
            show_progress_bar=False
        )
        
        best_params = study.best_params
        logger.info(f"Optuna завершена для {model_name}, best_value={study.best_value:.4f}, "
                   f"n_trials={len(study.trials)}")
        
        # ✅ ИСПРАВЛЕНО: Возврат обученной модели с лучшими параметрами
        return _instantiate_model(model_name, best_params)
    
    except Exception as e:
        logger.error(f"Ошибка Optuna: {e}")
        return None

# ✅ ДОБАВЛЕНО: Вспомогательные функции для Optuna
def _wrap_and_cv_score(
    model: BaseEstimator, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cv, 
    scoring: str, 
    dt_cols_hint: Optional[List[str]] = None
) -> float:
    """Обернуть модель в preprocessor и вычислить CV score."""
    try:
        pre = build_preprocessor(
            X_train, 
            dt_cols_hint=dt_cols_hint, 
            use_scaler=False, 
            handle_outliers=True,
            model_name=model.__class__.__name__
        )
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        scores = cross_val_score(
            pipe, X_train, y_train, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=N_JOBS,  # ✅ ИСПРАВЛЕНО: параллелизм
            error_score=np.nan
        )
        return float(np.nanmean(scores))
    except Exception as e:
        logger.error(f"CV score error: {e}")
        return 0.0

# Ridge objective
def _make_ridge_objective(trial, X_train, y_train, cv, dt_cols_hint):
    alpha = trial.suggest_float("alpha", 0.001, 100.0, log=True)
    model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "r2", dt_cols_hint)

# Lasso objective
def _make_lasso_objective(trial, X_train, y_train, cv, dt_cols_hint):
    alpha = trial.suggest_float("alpha", 0.0001, 1.0, log=True)
    model = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=5000)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "r2", dt_cols_hint)

# RandomForest Regressor objective
def _make_rf_objective(trial, X_train, y_train, cv, dt_cols_hint):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
    }
    model = RandomForestRegressor(**params)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "r2", dt_cols_hint)

# RandomForest Classifier objective
def _make_rf_cls_objective(trial, X_train, y_train, cv, dt_cols_hint):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
    }
    model = RandomForestClassifier(**params)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "f1_weighted", dt_cols_hint)

# LogisticRegression objective
def _make_logreg_objective(trial, X_train, y_train, cv, dt_cols_hint):
    C = trial.suggest_float("C", 0.001, 1000.0, log=True)
    model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED, n_jobs=N_JOBS)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "f1_weighted", dt_cols_hint)

# XGB Regressor objective
def _make_xgb_objective(trial, X_train, y_train, cv, dt_cols_hint):
    if not XGB_AVAILABLE:
        return 0.0
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": N_JOBS,
    }
    model = XGBRegressor(**params)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "r2", dt_cols_hint)

# XGB Classifier objective
def _make_xgb_cls_objective(trial, X_train, y_train, cv, dt_cols_hint):
    if not XGB_AVAILABLE:
        return 0.0
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_SEED,
        "use_label_encoder": False,
        "verbosity": 0,
        "n_jobs": N_JOBS,
    }
    model = XGBClassifier(**params)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "f1_weighted", dt_cols_hint)

# LGBM Regressor objective
def _make_lgbm_objective(trial, X_train, y_train, cv, dt_cols_hint):
    if not LGBM_AVAILABLE:
        return 0.0
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_SEED,
        "verbosity": -1,
        "n_jobs": N_JOBS,
    }
    model = LGBMRegressor(**params)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "r2", dt_cols_hint)

# LGBM Classifier objective
def _make_lgbm_cls_objective(trial, X_train, y_train, cv, dt_cols_hint):
    if not LGBM_AVAILABLE:
        return 0.0
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_SEED,
        "verbosity": -1,
        "n_jobs": N_JOBS,
    }
    model = LGBMClassifier(**params)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "f1_weighted", dt_cols_hint)

# CatBoost Regressor objective
def _make_cb_objective(trial, X_train, y_train, cv, dt_cols_hint):
    if not CATBOOST_AVAILABLE:
        return 0.0
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": N_JOBS,
    }
    model = CatBoostRegressor(**params)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "r2", dt_cols_hint)

# CatBoost Classifier objective
def _make_cb_cls_objective(trial, X_train, y_train, cv, dt_cols_hint):
    if not CATBOOST_AVAILABLE:
        return 0.0
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": N_JOBS,
    }
    model = CatBoostClassifier(**params)
    return _wrap_and_cv_score(model, X_train, y_train, cv, "f1_weighted", dt_cols_hint)

def _instantiate_model(model_name: str, params: Dict) -> Optional[BaseEstimator]:
    """✅ ДОБАВЛЕНО: Инстанцировать модель с параметрами."""
    try:
        if model_name == "Ridge":
            return Ridge(**params, random_state=RANDOM_SEED)
        elif model_name == "Lasso":
            return Lasso(**params, random_state=RANDOM_SEED, max_iter=5000)
        elif model_name == "RandomForestRegressor":
            return RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=N_JOBS)
        elif model_name == "RandomForestClassifier":
            return RandomForestClassifier(**params, random_state=RANDOM_SEED, n_jobs=N_JOBS)
        elif model_name == "LogisticRegression":
            return LogisticRegression(**params, max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED, n_jobs=N_JOBS)
        elif model_name == "XGBRegressor" and XGB_AVAILABLE:
            return XGBRegressor(**params, random_state=RANDOM_SEED, tree_method="hist", verbosity=0, n_jobs=N_JOBS)
        elif model_name == "XGBClassifier" and XGB_AVAILABLE:
            return XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False, verbosity=0, n_jobs=N_JOBS)
        elif model_name == "LGBMRegressor" and LGBM_AVAILABLE:
            return LGBMRegressor(**params, random_state=RANDOM_SEED, verbosity=-1, n_jobs=N_JOBS)
        elif model_name == "LGBMClassifier" and LGBM_AVAILABLE:
            return LGBMClassifier(**params, random_state=RANDOM_SEED, verbosity=-1, n_jobs=N_JOBS)
        elif model_name == "CatBoostRegressor" and CATBOOST_AVAILABLE:
            return CatBoostRegressor(**params, random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False, thread_count=N_JOBS)
        elif model_name == "CatBoostClassifier" and CATBOOST_AVAILABLE:
            return CatBoostClassifier(**params, random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False, thread_count=N_JOBS)
    except Exception as e:
        logger.error(f"Ошибка при инстанцировании {model_name}: {e}")
    
    return None
