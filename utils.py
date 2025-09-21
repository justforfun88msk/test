# utils.py

import base64
import time
import datetime as _dt
import uuid
import math
import streamlit as st

def _sleep_with_progress(seconds: float, label: str = "Завершение..."):
    """Плавный прогресс-бар на остаток времени."""
    if seconds <= 0:
        return
    steps = 30
    step = max(seconds / steps, 0.02)
    prog = st.progress(0)
    ph = st.empty()
    ph.info(label)
    for i in range(steps):
        time.sleep(step)
        prog.progress(int((i + 1) / steps * 100))
    ph.empty()
    prog.empty()

def enforce_min_duration(start_time: float, min_seconds: float = 6.0, label: str = "Финализация результатов..."):
    """Гарантирует минимум min_seconds с начала шага, не мешая реальной длительности."""
    elapsed = time.time() - start_time
    remaining = max(0.0, min_seconds - elapsed)
    _sleep_with_progress(remaining, label)

def get_session_id() -> str:
    """Возвращает уникальный ID для текущей сессии Streamlit."""
    if "_session_id" not in st.session_state:
        st.session_state["_session_id"] = str(uuid.uuid4())
    return st.session_state["_session_id"]

def get_ttl_to_4am() -> int:
    """Вычисляет TTL в секундах до 4 утра для сброса кэша."""
    now = _dt.datetime.now()
    cutoff = now.replace(hour=4, minute=0, second=0, microsecond=0)
    if now >= cutoff:
        cutoff = cutoff + _dt.timedelta(days=1)
    return max(60, int((cutoff - now).total_seconds()))

def download_button(bytes_data: bytes, filename: str, label: str, mime: str = "application/octet-stream"):
    """
    Создает кастомную кнопку для скачивания файла.
    ИСПРАВЛЕНО: переменная `data` заменена на `bytes_data`.
    """
    b64 = base64.b64encode(bytes_data).decode("utf-8")
    href = f'<a download="{filename}" href="data:{mime};base64,{b64}" style="text-decoration: none;">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)


def detect_csv_sep(first_bytes: bytes) -> str:
    """Определяет наиболее вероятный разделитель в CSV файле."""
    text = first_bytes.decode('utf-8', errors='ignore')
    candidates = [',', ';', '\t', '|']
    counts = {c: text.count(c) for c in candidates}
    return max(counts, key=counts.get) if counts else ','

def human_time_ms(ms: float) -> str:
    """Преобразует миллисекунды в человекочитаемый формат (ms, s, m s)."""
    if ms < 1_000:
        return f"{int(ms)} ms"
    s = ms / 1000
    if s < 60:
        return f"{s:.1f} s"
    m = int(s // 60)
    s = s - m * 60
    return f"{m}m {s:.0f}s"
