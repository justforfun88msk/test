# ml_core.py

import math
import time
import warnings
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
# ИСПРАВЛЕНО: Добавлен TransformerMixin в импорт
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, roc_auc_score, precision_recall_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from utils import get_session_id, get_ttl_to_4am
from ui_config import RANDOM_SEED, get_model_tags

# --- Проверка доступности опциональных библиотек ---
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from dateutil import parser as dateutil_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

warnings.filterwarnings("ignore")

# --- Логика определения типов данных ---

def is_categorical_series(s: pd.Series) -> bool:
    """Определяет, является ли столбец категориальным."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype.name in ('object', 'category', 'bool'):
        return True
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        nunique = s.astype('float').round(6).nunique(dropna=True)
        return nunique <= 15 and nunique <= len(s) * 0.05
    return False

@st.cache_data(ttl=get_ttl_to_4am(), show_spinner=False)
def detect_problem_type(y: pd.Series, _sid: str = None) -> str:
    """Автоматически определяет тип задачи (регрессия, бинарная/мультиклассовая классификация)."""
    if y is None:
        return 'unknown'
    y_clean = y.replace([np.inf, -np.inf], np.nan).dropna()
    if len(y_clean) < 2:
        raise ValueError("Таргет имеет менее 2 валидных значений после удаления NaN/inf — обучать нельзя.")
    
    nunq = int(y_clean.nunique())
    is_numeric = pd.api.types.is_numeric_dtype(y_clean)
    is_float_like = False
    
    if is_numeric:
        try:
            float_vals = y_clean.astype('float')
            rounded_vals = float_vals.round()
            is_float_like = not np.allclose(float_vals, rounded_vals, equal_nan=True)
        except (ValueError, TypeError):
            is_float_like = False

    if pd.api.types.is_bool_dtype(y_clean) or (not is_numeric and nunq > 1):
        return "binary" if nunq == 2 else "multiclass"
    elif not is_float_like and nunq <= max(20, int(len(y_clean) * 0.05)):
        return "binary" if nunq == 2 else "multiclass"
    else:
        return "regression"

# --- Логика Предобработки ---

class DateTimeExpander(BaseEstimator, TransformerMixin):
    """Трансформер для разбора и создания признаков из дат."""
    def __init__(self, dt_cols_hint: Optional[List[str]] = None, min_success: float = 0.7, strict_hint: bool = False):
        self.dt_cols_hint = dt_cols_hint
        self.min_success = min_success
        self.strict_hint = strict_hint
        self.dt_cols_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        cols_to_check = self.dt_cols_hint if self.strict_hint and self.dt_cols_hint else X.columns
        for c in cols_to_check:
            if c in X.columns:
                _, parsed_ok = self._safe_parse_datetime(X[c], self.min_success)
                if parsed_ok:
                    self.dt_cols_.append(c)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for c in self.dt_cols_:
            if c in X_copy.columns:
                parsed, _ = self._safe_parse_datetime(X_copy[c], self.min_success)
                if pd.api.types.is_datetime64_any_dtype(parsed):
                    X_copy[f'{c}_year'] = parsed.dt.year
                    X_copy[f'{c}_month'] = parsed.dt.month
                    X_copy[f'{c}_day'] = parsed.dt.day
                    X_copy[f'{c}_dayofweek'] = parsed.dt.dayofweek
                    X_copy[f'{c}_hour'] = parsed.dt.hour
                    X_copy = X_copy.drop(columns=[c])
        return X_copy

    def _safe_parse_datetime(self, col: pd.Series, min_success: float) -> Tuple[pd.Series, bool]:
        """Умный парсер дат, который пробует разные форматы, включая Excel."""
        if pd.api.types.is_datetime64_any_dtype(col):
            return col, True
        
        # Попытка с dateutil, если доступен
        if DATEUTIL_AVAILABLE:
            try:
                # errors='coerce' здесь не работает, нужен try/except внутри apply
                parsed_dates = []
                for item in col:
                    try:
                        parsed_dates.append(dateutil_parser.parse(str(item)) if pd.notna(item) else pd.NaT)
                    except (ValueError, TypeError):
                        parsed_dates.append(pd.NaT)
                parsed = pd.Series(parsed_dates, index=col.index)

                if parsed.notna().mean() >= min_success:
                    return pd.to_datetime(parsed), True
            except Exception:
                pass # Фоллбэк к pandas
        
        # Фоллбэк к pandas to_datetime
        try:
            parsed = pd.to_datetime(col, errors='coerce')
            if parsed.notna().mean() >= min_success:
                return parsed, True
        except Exception:
            return col, False
        
        return col, False

@st.cache_resource(ttl=get_ttl_to_4am(), show_spinner=False)
def build_preprocessor(X: pd.DataFrame, dt_cols_hint: Optional[List[str]], use_scaler: bool, handle_outliers: bool, _sid: str = None) -> Pipeline:
    """Строит пайплайн предобработки на основе данных."""
    
    dt_expander = DateTimeExpander(dt_cols_hint=dt_cols_hint, strict_hint=True if dt_cols_hint else False)
    # Используем fit_transform здесь, чтобы получить преобразованные колонки для дальнейшего анализа
    try:
        X_transformed = dt_expander.fit(X).transform(X)
    except Exception:
        # Если обработка дат падает, просто используем оригинальный X
        X_transformed = X.copy()
        dt_expander.dt_cols_ = []

    num_cols = [c for c in X_transformed.columns if not is_categorical_series(X_transformed[c])]
    cat_cols = [c for c in X_transformed.columns if is_categorical_series(X_transformed[c])]

    num_steps = [("imp", IterativeImputer(random_state=RANDOM_SEED, max_iter=10, initial_strategy="median", skip_complete=True))]
    if handle_outliers:
        num_steps.append(("outlier", RobustScaler()))
    if use_scaler:
        num_steps.append(("scale", StandardScaler()))
    
    num_pipe = Pipeline(steps=num_steps)
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=100)),
    ])
    
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
        
    column_tf = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.0)
    
    return Pipeline(steps=[
        ("dt", dt_expander),
        ("columns", column_tf),
    ])

# --- Логика Моделей и Оценки ---

def get_models(task: str) -> Dict[str, BaseEstimator]:
    """Возвращает словарь моделей для указанной задачи."""
    models = {}
    seed = RANDOM_SEED
    if task == 'regression':
        models.update({
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed),
            "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=seed),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=seed),
        })
        if XGB_AVAILABLE:
            models["XGBRegressor"] = XGBRegressor(random_state=seed, tree_method="hist")
        if LGBM_AVAILABLE:
            models["LGBMRegressor"] = LGBMRegressor(random_state=seed, verbose=-1)
        if CATBOOST_AVAILABLE:
            models["CatBoostRegressor"] = CatBoostRegressor(verbose=False, random_seed=seed, allow_writing_files=False)
    else: # classification
        models.update({
            "LogisticRegression": LogisticRegression(max_iter=200, solver="saga", random_state=seed),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
            "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=seed),
        })
        if XGB_AVAILABLE:
            models["XGBClassifier"] = XGBClassifier(random_state=seed, tree_method="hist", eval_metric="logloss")
        if LGBM_AVAILABLE:
            models["LGBMClassifier"] = LGBMClassifier(random_state=seed, verbose=-1)
        if CATBOOST_AVAILABLE:
            models["CatBoostClassifier"] = CatBoostClassifier(verbose=False, random_seed=seed, allow_writing_files=False)
    
    model_tags = get_model_tags(XGB_AVAILABLE, LGBM_AVAILABLE, CATBOOST_AVAILABLE)
    return {k: v for k, v in models.items() if model_tags.get(k) != "не доступна"}

def get_cv(task: str, n_splits: int, shuffle: bool, seed: int):
    """Возвращает объект кросс-валидатора."""
    if task in ('binary', 'multiclass'):
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

def get_metrics(task: str):
    """Возвращает словарь метрик для задачи."""
    if task in ("binary", "multiclass"):
        metrics = {
            "accuracy": make_scorer(accuracy_score),
            "f1_weighted": make_scorer(f1_score, average='weighted'),
        }
        if task == "binary":
            metrics["roc_auc"] = make_scorer(roc_auc_score, needs_proba=True)
        return metrics
    else:
        return {
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
            "RMSE": make_scorer(lambda y, p: math.sqrt(mean_squared_error(y, p)), greater_is_better=False),
            "R2": make_scorer(r2_score),
        }

def get_primary_metric(task: str) -> str:
    """Возвращает основную метрику для сортировки."""
    return {"binary": "PR-AUC", "multiclass": "f1_weighted", "regression": "RMSE"}.get(task)

def choose_sort_metric(cv_results: pd.DataFrame, task: str, primary_metric: str) -> str:
    """Выбирает метрику для сортировки, с фоллбэком если основная не посчиталась."""
    candidates_map = {
        "binary": [primary_metric, "roc_auc", "f1_weighted", "accuracy"],
        "multiclass": [primary_metric, "f1_weighted", "accuracy"],
        "regression": [primary_metric, "RMSE", "MAE", "R2"],
    }
    for m in candidates_map.get(task, [primary_metric]):
        if m in cv_results.columns and cv_results[m].notna().any():
            return m
    return ""

def metric_ascending(metric: str) -> bool:
    """Определяет, нужно ли сортировать метрику по возрастанию."""
    return metric in ("MAE", "RMSE")

def is_linear_model(name: str) -> bool:
    """Проверяет, является ли модель линейной."""
    return name in ("LinearRegression", "LogisticRegression")

@st.cache_data(ttl=get_ttl_to_4am(), show_spinner=False)
def cv_evaluate(_pipe: Pipeline, _model, X: pd.DataFrame, y: pd.Series, task: str,
                n_splits: int, shuffle: bool, seed: int, _sid: str = None) -> Tuple[Dict, float]:
    """Проводит кросс-валидацию модели и возвращает метрики и время выполнения."""
    start = time.time()
    cv = get_cv(task, n_splits, shuffle, seed)
    est_with_model = Pipeline(steps=_pipe.steps + [("model", _model)])
    metrics_dict = get_metrics(task)
    scores_dict = {}
    
    for metric_name, scorer_func in metrics_dict.items():
        try:
            scores = cross_val_score(est_with_model, X, y, cv=cv, scoring=scorer_func, n_jobs=-1)
            scores_dict[metric_name] = float(np.mean(scores))
        except Exception:
            scores_dict[metric_name] = np.nan
            
    if task == 'regression':
        for metric in ["MAE", "RMSE"]:
            if metric in scores_dict and np.isfinite(scores_dict[metric]):
                scores_dict[metric] = -scores_dict[metric] # sklearn convention
    
    if task == "binary":
        try:
            from sklearn.model_selection import cross_val_predict
            probas = cross_val_predict(est_with_model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            precision, recall, _ = precision_recall_curve(y, probas)
            scores_dict["PR-AUC"] = auc(recall, precision)
        except Exception:
            scores_dict["PR-AUC"] = np.nan
            
    dur_ms = (time.time() - start) * 1000
    return scores_dict, dur_ms

def select_best_model(cv_results: pd.DataFrame, task: str, metric_to_use: str) -> str:
    """Выбирает лучшую модель из таблицы результатов CV."""
    if cv_results.empty: return ""
    
    metric_to_use = choose_sort_metric(cv_results, task, metric_to_use)
    if not metric_to_use: return cv_results.iloc[0]["model"]
    
    ascending = metric_ascending(metric_to_use)
    sorted_df = cv_results.sort_values(
        by=metric_to_use,
        ascending=ascending,
        key=lambda x: x.fillna(np.inf if ascending else -np.inf)
    )
    return sorted_df.iloc[0]["model"]

@st.cache_resource(ttl=get_ttl_to_4am(), show_spinner=False)
def fit_best(_pipe: Pipeline, _model, X: pd.DataFrame, y: pd.Series, _sid: str = None) -> Pipeline:
    """Обучает пайплайн с моделью на всех данных."""
    est = Pipeline(steps=_pipe.steps + [("model", _model)])
    est.fit(X, y)
    return est
