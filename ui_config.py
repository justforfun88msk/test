# ui_config.py - Sminex AutoML v0.25 ULTIMATE - Конфигурация UI
# ✅ ВСЕ ИСПРАВЛЕНИЯ: Оптимизированные параметры

APP_VERSION = "v0.25 ULTIMATE"
RANDOM_SEED = 42
PARALLEL_CV = 1  # CV без параллелизма для стабильности (параллелизм в моделях)

MAX_DATASET_SIZE = 100000
SAMPLE_SIZE_FOR_LARGE_DATASETS = 50000

# ============ CSS СТИЛИ (Улучшенная светлая тема) ============
APP_CSS = """
body {
    background-color: #f5f5f7;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1d1d1f;
}

.stButton > button {
    background: linear-gradient(135deg, #007aff 0%, #0051d5 100%);
    color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 122, 255, 0.2);
    transition: all 0.3s ease;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    box-shadow: 0 4px 12px rgba(0, 122, 255, 0.4);
    transform: translateY(-1px);
}

.stButton > button:disabled {
    background: #cccccc;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
}

.stButton > button:active {
    transform: translateY(0);
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
}

.stExpander {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
}

.stMarkdown { line-height: 1.6; }

h1 { 
    color: #1d1d1f; 
    font-weight: 700; 
    font-size: 2.5em; 
    margin-bottom: 0.5em;
}

h2 { 
    color: #1d1d1f; 
    border-bottom: 3px solid #007aff; 
    padding-bottom: 10px;
    margin-top: 1em;
}

h3 { 
    color: #2d2d2f; 
    font-weight: 600;
    margin-top: 1em;
}

.stSelectbox, .stSlider, .stNumberInput, .stMultiSelect, .stTextInput {
    border-radius: 8px;
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
}

.stSelectbox:focus-within, .stSlider:focus-within, .stNumberInput:focus-within {
    border-color: #007aff;
    box-shadow: 0 0 0 2px rgba(0, 122, 255, 0.1);
}

.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    border: 1px solid #e0e0e0;
}

.stDataFrame table { 
    font-size: 14px; 
    line-height: 1.6; 
}

.stDataFrame th { 
    background-color: #f0f0f0; 
    font-weight: 700; 
    color: #1d1d1f;
    padding: 12px 16px !important;
}

.stDataFrame td { 
    padding: 10px 12px !important; 
}

.sticky-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: linear-gradient(135deg, #ffffff 0%, #f5f5f7 100%);
    padding: 16px;
    border-bottom: 2px solid #e0e0e0;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 16px;
}

.stMetric {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}

.stMetric label {
    font-weight: 600;
    color: #666;
    font-size: 0.9em;
}

.stMetric [data-testid="stMetricValue"] {
    font-size: 1.5em;
    font-weight: 700;
    color: #007aff;
}

.stProgress > div > div { 
    background: linear-gradient(90deg, #007aff, #0051d5) !important; 
    border-radius: 4px;
}

.stAlert { 
    border-radius: 8px; 
    padding: 12px 16px; 
    border-left: 4px solid;
}

div[data-baseweb="notification"] {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

::-webkit-scrollbar { 
    width: 8px; 
    height: 8px; 
}

::-webkit-scrollbar-track { 
    background: #f1f1f1; 
    border-radius: 4px;
}

::-webkit-scrollbar-thumb { 
    background: #888; 
    border-radius: 4px; 
}

::-webkit-scrollbar-thumb:hover { 
    background: #555; 
}

.info-box {
    background-color: #e8f4ff;
    border-left: 4px solid #007aff;
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
}

.success-box {
    background-color: #e8f5e9;
    border-left: 4px solid #4caf50;
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
}

.warning-box {
    background-color: #fff8e1;
    border-left: 4px solid #fbc02d;
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
}

.error-box {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
}

/* Улучшенные стили для файлового загрузчика */
[data-testid="stFileUploader"] {
    border: 2px dashed #007aff;
    border-radius: 8px;
    padding: 20px;
    background-color: #f8f9fa;
}

[data-testid="stFileUploader"]:hover {
    background-color: #e8f4ff;
    border-color: #0051d5;
}

/* Стили для форм */
.stForm {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}

/* Стили для tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background-color: #007aff;
    color: white;
}

/* Анимации */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.stMarkdown, .stDataFrame, .stMetric {
    animation: fadeIn 0.3s ease-out;
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
