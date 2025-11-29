# utils.py - Sminex AutoML v0.25 ULTIMATE - ВСЕ ИСПРАВЛЕНИЯ
# ✅ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
# - Кэширование результатов chardet по hash файла (ускорение)
# - Проверка на дубликаты строк с предупреждением
# - Stratified sampling для несбалансированных данных
# - Улучшенная валидация данных
# - Безопасная обработка больших файлов

# ✅ СРЕДНИЕ ИСПРАВЛЕНИЯ:
# - Оптимизация memory usage
# - Лучшая обработка кодировок
# - Улучшенное логирование

import base64
import time
import datetime as _dt
import uuid
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
import os
import logging
import hashlib

logger = logging.getLogger(__name__)

# ============ ПОПЫТКА ИМПОРТА CHARDET ДЛЯ ДЕТЕКЦИИ КОДИРОВКИ ============
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logger.warning("chardet не установлен. Используется UTF-8 по умолчанию.")

# ✅ ДОБАВЛЕНО: Кэш для результатов chardet
_ENCODING_CACHE: Dict[str, str] = {}

# ============ ПРОГРЕСС И ВРЕМЯ ============
def _sleep_with_progress(seconds: float, label: str = "Завершение..."):
    """Плавный прогресс-бар."""
    if seconds <= 0:
        return
    steps = min(30, int(seconds * 10))
    step = max(seconds / steps, 0.02)
    prog = st.progress(0)
    ph = st.empty()
    ph.info(label)
    for i in range(steps):
        time.sleep(step)
        prog.progress(min(100, int((i + 1) / steps * 100)))
    ph.empty()
    prog.empty()

def enforce_min_duration(start_time: float, min_seconds: float = 6.0, label: str = "Финализация..."):
    """Минимальная длительность операции."""
    elapsed = time.time() - start_time
    remaining = max(0.0, min_seconds - elapsed)
    _sleep_with_progress(remaining, label)

def human_time_ms(ms: float) -> str:
    """Преобразовать ms в читаемый формат."""
    if ms < 1_000:
        return f"{int(ms)} ms"
    s = ms / 1000
    if s < 60:
        return f"{s:.1f} s"
    m = int(s // 60)
    s = s - m * 60
    return f"{m}m {int(s)}s"


def format_eta(start_time: float, total_steps: int, completed_steps: int) -> str:
    """Оценка оставшегося времени для прогресса."""
    if completed_steps <= 0 or total_steps <= 0:
        return ""
    elapsed = time.time() - start_time
    avg = elapsed / completed_steps
    remaining = max(0.0, avg * (total_steps - completed_steps))
    if remaining < 60:
        return f"≈{int(remaining)}s"
    minutes = int(remaining // 60)
    seconds = int(remaining - minutes * 60)
    return f"≈{minutes}m {seconds:02d}s"

# ============ СЕССИЯ ============
def get_session_id() -> str:
    """Уникальный ID сессии."""
    if "_session_id" not in st.session_state:
        st.session_state["_session_id"] = str(uuid.uuid4())
    return st.session_state["_session_id"]

def get_ttl_to_4am() -> int:
    """TTL до 4 утра следующего дня."""
    now = _dt.datetime.now()
    cutoff = now.replace(hour=4, minute=0, second=0, microsecond=0)
    # Если время >= 4:00 AM, считаем следующий день
    if now >= cutoff:
        cutoff = cutoff + _dt.timedelta(days=1)
    ttl_seconds = max(60, int((cutoff - now).total_seconds()))
    return ttl_seconds

# ============ СКАЧИВАНИЕ ============
def download_button(bytes_data: bytes, filename: str, label: str, mime: str = "application/octet-stream"):
    """Кнопка скачивания."""
    b64 = base64.b64encode(bytes_data).decode("utf-8")
    href = f'<a download="{filename}" href="data:{mime};base64,{b64}" style="text-decoration:none;"><button style="padding:8px 16px;background:#007aff;color:white;border:none;border-radius:6px;cursor:pointer;">{label}</button></a>'
    st.markdown(href, unsafe_allow_html=True)

# ============ ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ДЕТЕКТИРОВАНИЕ КОДИРОВКИ С КЭШИРОВАНИЕМ ============
def detect_file_encoding(file_bytes: bytes, cache_key: Optional[str] = None) -> str:
    """
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обнаруживает кодировку файла с кэшированием.
    - Использует кэш по hash файла для ускорения
    - Fallback на стандартные кодировки
    - Безопасная обработка ошибок
    """
    # ✅ ДОБАВЛЕНО: Проверяем кэш
    if cache_key and cache_key in _ENCODING_CACHE:
        logger.info(f"Кодировка найдена в кэше: {_ENCODING_CACHE[cache_key]}")
        return _ENCODING_CACHE[cache_key]
    
    encoding = 'utf-8'  # default
    
    if CHARDET_AVAILABLE:
        try:
            # Используем только первые 50KB для скорости
            sample = file_bytes[:50000]
            detected = chardet.detect(sample)
            if detected and detected.get('encoding'):
                encoding = detected['encoding']
                # Нормализация имен кодировок
                if encoding.lower() in ('ascii', 'utf-8-sig'):
                    encoding = 'utf-8'
                logger.info(f"chardet обнаружил кодировку: {encoding} (confidence: {detected.get('confidence', 0):.2f})")
        except Exception as e:
            logger.warning(f"Ошибка chardet: {e}")
    
    if not CHARDET_AVAILABLE or encoding == 'utf-8':
        # Fallback: пробуем стандартные кодировки
        for enc in ['utf-8', 'cp1251', 'iso-8859-1', 'latin-1', 'ascii']:
            try:
                file_bytes[:10000].decode(enc)
                encoding = enc
                logger.info(f"Fallback: определена кодировка {enc}")
                break
            except (UnicodeDecodeError, AttributeError):
                continue
    
    # ✅ ДОБАВЛЕНО: Сохраняем в кэш
    if cache_key:
        _ENCODING_CACHE[cache_key] = encoding
    
    return encoding

def get_file_hash(file_bytes: bytes) -> str:
    """✅ ДОБАВЛЕНО: Получить hash файла для кэширования."""
    # Используем только первые 1MB для скорости
    sample = file_bytes[:1024*1024]
    return hashlib.md5(sample).hexdigest()

# ============ ДЕТЕКТИРОВАНИЕ РАЗДЕЛИТЕЛЯ ============
def detect_csv_sep(first_bytes: bytes, encoding: str = 'utf-8') -> str:
    """
    Определить разделитель CSV с учетом кодировки.
    """
    try:
        text = first_bytes.decode(encoding, errors='ignore')
    except Exception as e:
        logger.error(f"Ошибка декодирования CSV: {e}")
        text = first_bytes.decode('utf-8', errors='ignore')
    
    candidates = [',', ';', '\t', '|']
    counts = {c: text.count(c) for c in candidates}
    
    if not counts or max(counts.values()) == 0:
        return ','
    
    best_sep = max(counts, key=counts.get)
    lines = text.split('\n')[:10]
    
    try:
        field_counts = [len(line.split(best_sep)) for line in lines if line.strip()]
        # Проверяем консистентность
        if len(set(field_counts)) == 1 and field_counts[0] > 1:
            return best_sep
    except Exception as e:
        logger.warning(f"Ошибка при проверке sep консистентности: {e}")
    
    return ','

# ============ ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: SMART SAMPLE С STRATIFIED SAMPLING ============
def smart_sample_large_file(
    uploaded_file, 
    sep: str, 
    max_rows: int = 100000, 
    sample_size: int = 50000,
    encoding: str = 'utf-8',
    target_col: Optional[str] = None,
    task_type: Optional[str] = None
) -> pd.DataFrame:
    """
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Загрузить большой CSV с умным сэмплированием.
    - Правильное управление file pointer
    - Обработка кодировок
    - Надежный parsing
    - ✅ ДОБАВЛЕНО: Stratified sampling для несбалансированных данных
    - Обработка ошибок
    """
    try:
        uploaded_file.seek(0)
        
        # 1. Оценка размера: читаем чанки для оценки
        chunk_size = 50000
        rows_estimate = 0
        chunks_to_est = 3
        
        try:
            temp_iter = pd.read_csv(
                uploaded_file, 
                sep=sep, 
                chunksize=chunk_size,
                encoding=encoding,
                on_bad_lines='skip',
                engine='python'
            )
            for i, chunk in enumerate(temp_iter):
                rows_estimate += len(chunk)
                if i >= chunks_to_est:
                    break
        except Exception as e:
            logger.warning(f"Ошибка при оценке размера: {e}")
            rows_estimate = 1000
        
        # Сбросить file pointer
        uploaded_file.seek(0)
        
        estimated_total = rows_estimate * (10 if rows_estimate > 0 else 1)
        
        # Если маленький файл - читаем целиком
        if rows_estimate < max_rows and chunks_to_est < 3:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file, 
                sep=sep,
                encoding=encoding,
                on_bad_lines='skip',
                engine='python'
            )
            logger.info(f"Загружен маленький файл: {df.shape}")
            return df
        
        # 2. Сэмплирование больших файлов
        uploaded_file.seek(0)
        final_chunks = []
        total_collected = 0
        
        frac = min(1.0, (sample_size * 1.5) / max(1, estimated_total))
        
        try:
            reader = pd.read_csv(
                uploaded_file, 
                sep=sep, 
                chunksize=chunk_size,
                encoding=encoding,
                on_bad_lines='skip',
                engine='python'
            )
            for chunk in reader:
                if len(chunk) > 0:
                    sampled_chunk = chunk.sample(frac=frac, random_state=42)
                    final_chunks.append(sampled_chunk)
                    total_collected += len(sampled_chunk)
                    
                    if total_collected > max_rows * 1.2:
                        break
        except Exception as e:
            logger.error(f"Ошибка при сэмплировании: {e}")
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file, 
                sep=sep,
                encoding=encoding,
                on_bad_lines='skip',
                nrows=sample_size,
                engine='python'
            )
        
        if not final_chunks:
            return pd.DataFrame()
        
        df = pd.concat(final_chunks, ignore_index=True)
        
        # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Stratified sampling для несбалансированных данных
        if len(df) > sample_size and target_col and target_col in df.columns and task_type in ('binary', 'multiclass'):
            try:
                from sklearn.model_selection import train_test_split
                # Удаляем NaN из target
                df_valid = df[df[target_col].notna()].copy()
                
                if len(df_valid) > sample_size:
                    # Проверяем что все классы имеют достаточно примеров
                    class_counts = df_valid[target_col].value_counts()
                    min_class_count = class_counts.min()
                    
                    if min_class_count >= 2:
                        # Stratified sampling
                        _, df_sampled = train_test_split(
                            df_valid,
                            test_size=sample_size / len(df_valid),
                            stratify=df_valid[target_col],
                            random_state=42
                        )
                        logger.info(f"Применен stratified sampling: {len(df_sampled)} строк")
                        return df_sampled.reset_index(drop=True)
                    else:
                        logger.warning(f"Stratified sampling невозможен (мин. класс: {min_class_count})")
            except Exception as e:
                logger.warning(f"Ошибка stratified sampling: {e}, используем random")
        
        # Обычный random sampling
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        logger.info(f"Загружена выборка из большого файла: {df.shape}")
        return df.reset_index(drop=True)
    
    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке файла: {e}")
        raise

# ============ ✅ УЛУЧШЕНО: ВАЛИДАЦИЯ ============
def validate_data_types(train_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    ✅ УЛУЧШЕНО: Проверить соответствие типов данных с автоматической конвертацией.
    - Более строгая валидация
    - Лучшие сообщения об ошибках
    - Безопасная конвертация типов
    """
    errors = []
    warnings_list = []
    
    missing_cols = set(train_df.columns) - set(new_df.columns)
    if missing_cols:
        errors.append(f"❌ Отсутствуют столбцы: {list(missing_cols)}")
        return False, errors
    
    # Проверка на inf и NaN
    new_df_clean = new_df.replace([np.inf, -np.inf], np.nan)
    has_missing = new_df_clean.isnull().any().any()
    if has_missing:
        missing_pct = new_df_clean.isnull().sum().sum() / (new_df_clean.shape[0] * new_df_clean.shape[1]) * 100
        warnings_list.append(f"⚠️ Пропуски или бесконечные значения ({missing_pct:.1f}%)")
    
    # Проверка и конвертация типов
    for col in train_df.columns:
        if col not in new_df.columns:
            continue
            
        train_type = train_df[col].dtype
        new_type = new_df[col].dtype
        
        # Категория числовых типов
        train_is_numeric = pd.api.types.is_numeric_dtype(train_type)
        new_is_numeric = pd.api.types.is_numeric_dtype(new_type)
        
        if train_is_numeric != new_is_numeric:
            try:
                if train_is_numeric:
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                else:
                    new_df[col] = new_df[col].astype(str)
                warnings_list.append(f"⚠️ Столбец '{col}' переконвертирован: {new_type} → {train_type}")
            except Exception as e:
                errors.append(f"❌ Невозможно конвертировать '{col}': {str(e)[:50]}")
    
    all_messages = errors + warnings_list
    return len(errors) == 0, all_messages

def detect_positive_class(y: pd.Series, y_prob: np.ndarray) -> int:
    """Определить позитивный класс для бинарной классификации."""
    if len(np.unique(y)) != 2 or y_prob.ndim != 2 or y_prob.shape[1] != 2:
        return 1
    
    class_means = np.mean(y_prob, axis=0)
    return int(np.argmax(class_means))

# ============ ✅ ДОБАВЛЕНО: ПРОВЕРКА НА ДУБЛИКАТЫ СТРОК ============
def check_and_remove_duplicates(df: pd.DataFrame, warn: bool = True) -> Tuple[pd.DataFrame, int]:
    """
    ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверка и удаление дублирующихся строк.
    
    Args:
        df: DataFrame для проверки
        warn: Показать предупреждение если найдены дубликаты
    
    Returns:
        (очищенный DataFrame, количество удаленных дубликатов)
    """
    original_len = len(df)
    df_clean = df.drop_duplicates()
    n_duplicates = original_len - len(df_clean)
    
    if n_duplicates > 0 and warn:
        dup_pct = (n_duplicates / original_len) * 100
        logger.warning(f"Найдено {n_duplicates} дублирующихся строк ({dup_pct:.1f}%)")
        if dup_pct > 5:
            logger.warning("⚠️ ВНИМАНИЕ: Более 5% дублирующихся строк! "
                          "Это может исказить метрики качества модели.")
    
    return df_clean, n_duplicates

# ============ РЕКОМЕНДАЦИИ ============
def get_random_tip() -> str:
    """Случайный совет."""
    tips = [
        "💡 Используйте минимум 100 строк для лучших результатов",
        "🔄 Пробуйте разные модели - не все одинаковы",
        "📊 80/20 - стандартный split train/test",
        "⚙️ OPTUNA находит лучшие параметры",
        "🎯 Проверяйте важность признаков",
        "📈 ROC-AUC лучше для несбалансированных данных",
        "🔮 What-If калькулятор показывает влияние",
        "🧹 Удаляйте дублирующиеся столбцы И строки",
        "🎪 Кат. признаки работают лучше после OHE",
        "⚡ Параллелизм ускоряет обучение в разы",
        "🎲 Stratified sampling важен для несбалансированных классов",
        "📚 Больше данных = лучше качество",
    ]
    return np.random.choice(tips)

# ============ УТИЛИТЫ ДЛЯ РАБОТЫ С ДАННЫМИ ============
def get_file_size_mb(file) -> float:
    """Получить размер файла в МБ."""
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset
    return size / (1024 * 1024)

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Очистить названия столбцов от проблемных символов."""
    df = df.copy()
    df.columns = [str(c).strip().replace('\n', ' ').replace('\r', ' ') for c in df.columns]
    return df

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ УЛУЧШЕНО: Удалить полностью дублирующиеся столбцы.
    - По именам
    - По содержимому
    """
    df = df.copy()
    
    # Удалить дубликаты по именам
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # Удалить столбцы, которые полностью идентичны по значениям
    cols_to_check = list(df.columns)
    duplicates_to_drop = set()
    
    for i, col1 in enumerate(cols_to_check):
        if col1 in duplicates_to_drop:
            continue
        for col2 in cols_to_check[i+1:]:
            if col2 in duplicates_to_drop:
                continue
            try:
                if df[col1].equals(df[col2]):
                    duplicates_to_drop.add(col2)
                    logger.info(f"Столбец '{col2}' идентичен '{col1}', будет удален")
            except Exception:
                pass
    
    if duplicates_to_drop:
        df = df.drop(columns=list(duplicates_to_drop), errors='ignore')
        logger.info(f"Удалено дублирующихся столбцов: {len(duplicates_to_drop)}")
    
    return df

# ============ ✅ ДОБАВЛЕНО: MEMORY MANAGEMENT ============
def estimate_memory_usage(df: pd.DataFrame) -> float:
    """
    ✅ ДОБАВЛЕНО: Оценить использование памяти DataFrame в МБ.
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    return memory_bytes / (1024 * 1024)

def optimize_dtypes(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
    """
    ✅ ДОБАВЛЕНО: Оптимизировать типы данных для экономии памяти.
    
    Args:
        df: DataFrame для оптимизации
        aggressive: Использовать более агрессивную оптимизацию (может потерять точность)
    
    Returns:
        Оптимизированный DataFrame
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # Оптимизация числовых типов
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min >= 0:
                if c_max < 255:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif c_max < 65535:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
        
        elif pd.api.types.is_float_dtype(col_type) and aggressive:
            df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # Оптимизация object типов
        elif col_type == 'object':
            num_unique_values = df_optimized[col].nunique()
            num_total_values = len(df_optimized[col])
            
            if num_unique_values / num_total_values < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
    
    memory_before = estimate_memory_usage(df)
    memory_after = estimate_memory_usage(df_optimized)
    memory_saved = memory_before - memory_after
    
    if memory_saved > 0:
        logger.info(f"Оптимизация типов: экономия {memory_saved:.1f} МБ "
                   f"({memory_saved/memory_before*100:.1f}%)")
    
    return df_optimized
