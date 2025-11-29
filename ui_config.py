# ui_config.py - Sminex AutoML v0.25 ULTIMATE - Конфигурация UI
# ✅ ВСЕ ИСПРАВЛЕНИЯ: Оптимизированные параметры

APP_VERSION = "v0.25 ULTIMATE"
RANDOM_SEED = 42
PARALLEL_CV = 1  # CV без параллелизма для стабильности (параллелизм в моделях)

MAX_DATASET_SIZE = 100000
SAMPLE_SIZE_FOR_LARGE_DATASETS = 50000

# ============ CSS СТИЛИ (лаконичный премиум стиль) ============
APP_CSS = """
body {
    background: linear-gradient(145deg, #f3f4f8 0%, #f8fbff 100%);
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #0f1419;
}

.floating-hint {
    text-align: center;
    color: #6b7280;
    margin-bottom: 18px;
    letter-spacing: 0.01em;
}

.stButton > button {
    background: linear-gradient(120deg, #4f8df3 0%, #7ec8ff 100%);
    color: #0f1419;
    border-radius: 12px;
    box-shadow: 0 12px 30px rgba(79, 141, 243, 0.26);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    padding: 12px 18px;
    font-weight: 700;
    border: none;
}

.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 18px 40px rgba(79, 141, 243, 0.3); }
.stButton > button:active { transform: translateY(0); }
.stButton > button:disabled { background: #e5e7eb; color: #9ca3af; box-shadow: none; }

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(15,20,25,0.06);
    box-shadow: 12px 0 40px rgba(0,0,0,0.04);
}

.stMarkdown { line-height: 1.55; }

h1, h2, h3 { font-weight: 700; color: #0f1419; }
h2 { border-bottom: none; padding-bottom: 4px; }

.stSelectbox, .stSlider, .stNumberInput, .stMultiSelect, .stTextInput {
    border-radius: 12px;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
}

.stSelectbox:focus-within, .stSlider:focus-within, .stNumberInput:focus-within {
    border-color: #4f8df3;
    box-shadow: 0 0 0 4px rgba(79, 141, 243, 0.12);
}

.stDataFrame, .stForm, .stExpander {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(17, 24, 39, 0.08);
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(15,20,25,0.05);
    backdrop-filter: blur(10px);
}

.sticky-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: linear-gradient(120deg, rgba(255,255,255,0.92), rgba(230,239,255,0.92));
    padding: 14px 16px;
    border-radius: 14px;
    box-shadow: 0 10px 38px rgba(79,141,243,0.18);
    margin-bottom: 16px;
    border: 1px solid rgba(79,141,243,0.12);
}

.stMetric {
    background: rgba(255,255,255,0.88);
    padding: 16px;
    border-radius: 14px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 18px 34px rgba(0,0,0,0.06);
    border: 1px solid rgba(15,20,25,0.05);
}

.stMetric [data-testid="stMetricValue"] { font-size: 1.4em; font-weight: 800; color: #0f1419; }
.stMetric label { color: #6b7280; font-weight: 600; }

.stProgress > div {
    background: rgba(15,20,25,0.06);
    border-radius: 999px;
    padding: 4px;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #5fe1c8 0%, #4f8df3 80%);
    border-radius: 999px;
    box-shadow: 0 8px 20px rgba(79,141,243,0.3);
    height: 12px;
}

div[data-baseweb="notification"], .stAlert {
    border-radius: 12px;
    box-shadow: 0 12px 30px rgba(17,24,39,0.12);
    border: 1px solid rgba(15,20,25,0.06);
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(79,141,243,0.55); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(79,141,243,0.75); }

.dashboard-card {
    background: linear-gradient(160deg, rgba(79,141,243,0.08), rgba(95,225,200,0.08));
    border-radius: 16px;
    padding: 16px;
    border: 1px solid rgba(79,141,243,0.16);
    box-shadow: 0 18px 38px rgba(0,0,0,0.08);
}
"""

# ============ ОПИСАНИЯ МОДЕЛЕЙ С УЛУЧШЕНИЯМИ ============
MODEL_DESCRIPTIONS = {
    "LinearRegression": "⚡ Простая линейная модель. Быстро и интерпретируемо. Хорошо на линейных зависимостях.",
    "Ridge": "🔒 Линейная с L2-регуляризацией. Устойчива к мультиколлинеарности.",
    "Lasso": "✂️ Линейная с L1-регуляризацией. Автоматический отбор признаков.",
    "LogisticRegression": "📊 Базовая классификация. Быстро, интерпретируемо, надежно.",
    "RandomForestClassifier": "🌲 Ансамбль деревьев. Отлично на табличных данных, устойчив к переобучению.",
    "RandomForestRegressor": "🌲 Ансамбль для регрессии. Надежный бэйзлайн, не требует масштабирования.",
    "ExtraTreesClassifier": "🎲 Как RF, но быстрее. Больше случайности = меньше переобучение.",
    "ExtraTreesRegressor": "🎲 Как RF регрессия, но быстрее. Хорошо на больших данных.",
    "HistGradientBoostingClassifier": "📈 Встроенный бустинг. Быстро без внешних зависимостей.",
    "HistGradientBoostingRegressor": "📈 Встроенный бустинг регрессия. Хорошее качество из коробки.",
    "XGBClassifier": "🚀 Классический XGBoost. Высочайшее качество, победитель Kaggle.",
    "XGBRegressor": "🚀 XGBoost регрессия. Мощная, гибкая, оптимизированная.",
    "LGBMClassifier": "⚡ LightGBM. Очень быстрый и качественный, экономия памяти.",
    "LGBMRegressor": "⚡ LightGBM регрессия. Быстро даже на огромных данных.",
    "CatBoostClassifier": "🐱 Отлично с категориями. Не требует OHE, работает на GPU.",
    "CatBoostRegressor": "🐱 CatBoost регрессия. Хорошо на малых выборках, устойчив.",
}

def get_model_tags(xgb_available, lgbm_available, catboost_available):
    """Теги доступности моделей с эмодзи."""
    return {
        "LinearRegression": "✅ встроенная",
        "Ridge": "✅ встроенная",
        "Lasso": "✅ встроенная",
        "LogisticRegression": "✅ встроенная",
        "RandomForestClassifier": "✅ встроенная",
        "RandomForestRegressor": "✅ встроенная",
        "ExtraTreesClassifier": "✅ встроенная",
        "ExtraTreesRegressor": "✅ встроенная",
        "HistGradientBoostingClassifier": "✅ встроенная",
        "HistGradientBoostingRegressor": "✅ встроенная",
        "XGBClassifier": "✅ доступна" if xgb_available else "❌ pip install xgboost",
        "XGBRegressor": "✅ доступна" if xgb_available else "❌ pip install xgboost",
        "LGBMClassifier": "✅ доступна" if lgbm_available else "❌ pip install lightgbm",
        "LGBMRegressor": "✅ доступна" if lgbm_available else "❌ pip install lightgbm",
        "CatBoostClassifier": "✅ доступна" if catboost_available else "❌ pip install catboost",
        "CatBoostRegressor": "✅ доступна" if catboost_available else "❌ pip install catboost",
    }

# ============ СОВЕТЫ И ПОДСКАЗКИ ============
TIPS = {
    "data_loading": [
        "💡 Файлы CSV быстрее загружаются чем Excel",
        "💡 UTF-8 кодировка предпочтительнее",
        "💡 Удалите лишние столбцы перед загрузкой",
        "💡 Проверьте что нет пропусков в целевой переменной"
    ],
    "feature_selection": [
        "💡 Больше признаков ≠ лучше качество",
        "💡 Удалите сильно коррелирующие признаки",
        "💡 Текстовые признаки требуют много памяти",
        "💡 Даты лучше разбивать на компоненты"
    ],
    "training": [
        "💡 Параллелизм ускоряет обучение в разы",
        "💡 Точный режим дает +2-5% к качеству",
        "💡 Начните с быстрого режима",
        "💡 XGBoost обычно лучший выбор"
    ],
    "evaluation": [
        "💡 R² показывает долю объясненной дисперсии",
        "💡 ROC-AUC лучше для несбалансированных классов",
        "💡 Важность признаков помогает понять модель",
        "💡 Проверьте модель на тестовой выборке"
    ]
}
