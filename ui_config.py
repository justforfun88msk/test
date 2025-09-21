# ui_config.py

# --- Константы для всего проекта ---
APP_VERSION = "v0.22 rc"
RANDOM_SEED = 42

# --- Стили CSS для приложения ---
APP_CSS = """
body {
    background-color: #f5f5f7;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1d1d1f;
    transition: background-color 0.3s ease, color 0.3s ease;
}
body.dark-theme {
    background-color: #121212;
    color: #e0e0e0;
}
.stButton > button {
    background-color: #333333; /* Стальной серый */
    color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
}
.stButton > button:hover {
    background-color: #007aff; /* Акцентный синий при наведении */
    box-shadow: 0 4px 8px rgba(0, 122, 255, 0.2);
}
.stButton > button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
}
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
    transition: background-color 0.3s ease, color 0.3s ease, border-right 0.3s ease, box-shadow 0.3s ease;
}
body.dark-theme section[data-testid="stSidebar"] {
    background-color: #1e1e1e;
    border-right: 1px solid #333333;
    color: #e0e0e0;
}
.stExpander {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    background-color: #ffffff;
    transition: background-color 0.3s ease;
}
body.dark-theme .stExpander {
    background-color: #2a2a2a;
    box-shadow: 0 1px 3px rgba(255, 255, 255, 0.1);
}
.stMarkdown { line-height: 1.6; }
.stSelectbox, .stSlider, .stNumberInput, .stMultiSelect {
    border-radius: 8px;
    background-color: #ffffff;
    padding: 8px;
    transition: background-color 0.3s ease;
}
body.dark-theme .stSelectbox, body.dark-theme .stSlider, body.dark-theme .stNumberInput, body.dark-theme .stMultiSelect {
    background-color: #2a2a2a;
    color: #e0e0e0;
}
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
.stDataFrame table {
    font-size: 14px;
}
.stDataFrame th, .stDataFrame td {
    padding: 4px 8px !important;
}
/* Стили для бейджей в липкой панели */
.sticky-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background-color: #ffffff;
    padding: 10px 0;
    border-bottom: 1px solid #e0e0e0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: background-color 0.3s ease, border-bottom 0.3s ease, box-shadow 0.3s ease;
}
body.dark-theme .sticky-bar {
    background-color: #1e1e1e;
    border-bottom: 1px solid #333333;
    box-shadow: 0 2px 4px rgba(255,255,255,0.05);
}
.badge {
    display: inline-block;
    padding: 4px 8px;
    margin-right: 8px;
    background-color: #f0f0f0;
    border-radius: 12px;
    font-size: 12px;
    color: #333;
    transition: background-color 0.3s ease, color 0.3s ease;
}
body.dark-theme .badge {
    background-color: #333333;
    color: #e0e0e0;
}
.badge b {
    color: #007aff;
}
.step-ok { background-color: #e8f5e9; color: #2e7d32; }
body.dark-theme .step-ok { background-color: #1b5e20; color: #c8e6c9; }
.step-warn { background-color: #fff8e1; color: #f57f17; }
body.dark-theme .step-warn { background-color: #ff6f00; color: #fff3e0; }
.step-current { background-color: #e3f2fd; color: #1565c0; font-weight: bold; }
body.dark-theme .step-current { background-color: #0d47a1; color: #bbdefb; }
/* Стили для калькулятора */
.highlight-row {
    background-color: #e3f2fd !important; /* Light blue for base row */
}
.highlight-opt-row {
    background-color: #ffebee !important; /* Light red for opt row */
}
body.dark-theme .highlight-row {
    background-color: #0d47a1 !important;
    color: #bbdefb !important;
}
body.dark-theme .highlight-opt-row {
    background-color: #b71c1c !important;
    color: #ffcdd2 !important;
}
/* Стили для карточек моделей */
.model-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.3s ease, transform 0.2s ease;
}
.model-card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
body.dark-theme .model-card {
    background-color: #2a2a2a;
    border-color: #444444;
    box-shadow: 0 2px 4px rgba(255,255,255,0.05);
}
.model-card.selected {
    border: 2px solid #007aff;
    background-color: #e3f2fd;
}
body.dark-theme .model-card.selected {
    border-color: #4da6ff;
    background-color: #0d47a1;
}
.model-card h3 {
    margin-top: 0;
    margin-bottom: 8px;
}
.model-card p {
    margin: 4px 0;
    font-size: 14px;
}
.model-card .metric {
    font-size: 24px;
    font-weight: bold;
    color: #007aff;
}
body.dark-theme .model-card .metric {
    color: #4da6ff;
}
/* Стили для метрик */
.stMetric {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
body.dark-theme .stMetric {
    background-color: #2a2a2a;
    box-shadow: 0 1px 3px rgba(255,255,255,0.1);
}
"""

# --- Описания и теги моделей ---
MODEL_DESCRIPTIONS = {
    "LogisticRegression": "Базовые линейные модели, быстрые и интерпретируемые. Хороши для простых зависимостей, чувствительны к масштабу признаков.",
    "LinearRegression": "Базовые линейные модели, быстрые и интерпретируемые. Хороши для простых зависимостей, чувствительны к масштабу признаков.",
    "RandomForestClassifier": "Ансамбль деревьев без сильной настройки. Хорошо работает на «таблицах», устойчив к шуму, чуть медленнее на больших данных.",
    "RandomForestRegressor": "Ансамбль деревьев без сильной настройки. Хорошо работает на «таблицах», устойчив к шуму, чуть медленнее на больших данных.",
    "ExtraTreesClassifier": "Как RF, но быстрее за счёт сильной рандомизации. Часто не хуже по качеству, хороший «быстрый бэйзлайн».",
    "ExtraTreesRegressor": "Как RF, но быстрее за счёт сильной рандомизации. Часто не хуже по качеству, хороший «быстрый бэйзлайн».",
    "HistGradientBoostingClassifier": "Встроенный бустинг sklearn. Быстрый, без внешних зависимостей, качество близко к LightGBM.",
    "HistGradientBoostingRegressor": "Встроенный бустинг sklearn. Быстрый, без внешних зависимостей, качество близко к LightGBM.",
    "XGBClassifier": "Классический бустинг, высокое качество, много гиперпараметров. Чуть тяжелее по ресурсам.",
    "XGBRegressor": "Классический бустинг, высокое качество, много гиперпараметров. Чуть тяжелее по ресурсам.",
    "LGBMClassifier": "Очень быстрый и качественный бустинг, хорошо масштабируется. Требует установленного пакета.",
    "LGBMRegressor": "Очень быстрый и качественный бустинг, хорошо масштабируется. Требует установленного пакета.",
    "CatBoostClassifier": "Отлично работает с категориями «как есть», меньше возни с OHE. Быстрый старт, иногда лучше на малых выборках.",
    "CatBoostRegressor": "Отлично работает с категориями «как есть», меньше возни с OHE. Быстрый старт, иногда лучше на малых выборках.",
}

# --- Функция для динамической генерации тегов ---
def get_model_tags(xgb_available, lgbm_available, catboost_available):
    return {
        "LogisticRegression": "встроенный",
        "LinearRegression": "встроенный",
        "RandomForestClassifier": "встроенный",
        "RandomForestRegressor": "встроенный",
        "ExtraTreesClassifier": "встроенный",
        "ExtraTreesRegressor": "встроенный",
        "HistGradientBoostingClassifier": "встроенный",
        "HistGradientBoostingRegressor": "встроенный",
        "XGBClassifier": "нужна библиотека" if xgb_available else "не доступна",
        "XGBRegressor": "нужна библиотека" if xgb_available else "не доступна",
        "LGBMClassifier": "нужна библиотека" if lgbm_available else "не доступна",
        "LGBMRegressor": "нужна библиотека" if lgbm_available else "не доступна",
        "CatBoostClassifier": "нужна библиотека" if catboost_available else "не доступна",
        "CatBoostRegressor": "нужна библиотека" if catboost_available else "не доступна",
    }
