# ui_config.py - Sminex AutoML v0.25 ULTIMATE - Конфигурация UI
# ✅ ВСЕ ИСПРАВЛЕНИЯ: Оптимизированные параметры

APP_VERSION = "v0.25 ULTIMATE"
RANDOM_SEED = 42
PARALLEL_CV = 1  # CV без параллелизма для стабильности (параллелизм в моделях)

MAX_DATASET_SIZE = 100000
SAMPLE_SIZE_FOR_LARGE_DATASETS = 50000

# ============ CSS СТИЛИ (светлая минималистичная тема) ============
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --font-sans: 'Inter', 'Inter var', 'Segoe UI', system-ui, -apple-system, sans-serif;
    --font-display: 'Inter', 'Segoe UI', system-ui;

    --bg: #f7f9fb;
    --bg-2: #eef2f7;
    --surface: #ffffff;
    --surface-2: #f5f7fa;
    --text: #0f172a;
    --muted: #4b5563;
    --border: rgba(15, 23, 42, 0.08);
    --accent: #4f46e5;
    --accent-2: #22c55e;
    --accent-3: #0ea5e9;
    --glass: rgba(79, 70, 229, 0.08);
    --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    --radius: 14px;
    --blur: 14px;

    --space-2xs: 4px;
    --space-xs: 8px;
    --space-sm: 12px;
    --space-md: 16px;
    --space-lg: 24px;
    --space-xl: 32px;
    --space-2xl: 48px;
}

* { box-sizing: border-box; }

body {
    background: var(--bg);
    font-family: var(--font-sans);
    color: var(--text);
    transition: background 0.3s ease, color 0.3s ease;
}

.floating-hint {
    text-align: center;
    color: var(--muted);
    margin-bottom: var(--space-md);
    letter-spacing: 0.02em;
    font-size: 0.95rem;
}

.stMarkdown { line-height: 1.6; color: var(--text); }
h1, h2, h3 { font-family: var(--font-display); color: var(--text); letter-spacing: -0.01em; }
h2 { border-bottom: none; padding-bottom: 6px; }

.stApp header { background: transparent; }

section[data-testid="stSidebar"] {
    background: var(--surface);
    backdrop-filter: blur(var(--blur));
    border-right: 1px solid var(--border);
    box-shadow: var(--shadow);
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-3) 100%);
    color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 10px 24px rgba(79, 70, 229, 0.18);
    transition: transform 0.12s ease, box-shadow 0.18s ease;
    padding: 11px 16px;
    font-weight: 700;
    border: none;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(79, 70, 229, 0.24);
}

.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); }
.stButton > button:disabled, .stDownloadButton > button:disabled { background: var(--border); color: var(--muted); box-shadow: none; }

.cta-ghost button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border);
    box-shadow: none;
}

.stSelectbox, .stSlider, .stNumberInput, .stMultiSelect, .stTextInput {
    border-radius: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), var(--shadow);
    color: var(--text);
}

.stSelectbox:focus-within, .stSlider:focus-within, .stNumberInput:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px rgba(124, 108, 255, 0.18);
}

.stDataFrame, .stForm, .stExpander {
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
    background: var(--surface);
    border: 1px solid var(--border);
}

.sticky-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--surface);
    padding: var(--space-md);
    border-radius: 14px;
    box-shadow: var(--shadow);
    margin-bottom: var(--space-md);
    border: 1px solid var(--border);
}

.stMetric {
    background: var(--surface);
    padding: var(--space-md);
    border-radius: 14px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), var(--shadow);
    border: 1px solid var(--border);
}

.stMetric [data-testid="stMetricValue"] { font-size: 1.25em; font-weight: 800; color: var(--text); }
.stMetric label { color: var(--muted); font-weight: 600; }

.stProgress > div {
    background: var(--surface-2);
    border-radius: 999px;
    padding: 4px;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 70%, var(--accent-3) 100%);
    border-radius: 999px;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.22);
    height: 12px;
}

div[data-baseweb="notification"], .stAlert {
    border-radius: 14px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    background: var(--surface);
}

::-webkit-scrollbar { width: 10px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: linear-gradient(var(--accent), var(--accent-2)); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(var(--accent-2), var(--accent)); }

.dashboard-card {
    background: linear-gradient(145deg, var(--surface-2), rgba(124,108,255,0.08));
    border-radius: 18px;
    padding: var(--space-lg);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

.ui-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: var(--space-md);
    align-items: stretch;
}

.ui-card {
    padding: var(--space-md);
    border-radius: var(--radius);
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

.ui-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--surface-2);
    color: var(--text);
    border-radius: 999px;
    border: 1px solid var(--border);
    font-weight: 600;
    letter-spacing: 0.01em;
}

.ui-stepper {
    display: flex;
    gap: var(--space-sm);
    align-items: center;
    flex-wrap: wrap;
}

.ui-stepper .step {
    padding: 8px 12px;
    border-radius: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    transition: all 0.15s ease;
}

.ui-stepper .step.active {
    color: var(--text);
    border-color: var(--accent);
    box-shadow: 0 6px 24px rgba(124,108,255,0.28);
}

.hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: var(--space-xl);
    box-shadow: var(--shadow);
}

.hero h1 { font-size: 2.1rem; margin-bottom: var(--space-sm); }
.hero p { max-width: 640px; color: var(--muted); margin-bottom: var(--space-md); }

.analytics-stack {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--space-sm);
}

.analytics-stack .item {
    padding: var(--space-md);
    border-radius: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
}

.skeleton-row {
    height: 12px;
    background: linear-gradient(90deg, rgba(15,23,42,0.06), rgba(15,23,42,0.12), rgba(15,23,42,0.06));
    background-size: 200% 100%;
    animation: shimmer 1.4s ease-in-out infinite;
    border-radius: 999px;
}

.skeleton-card {
    padding: var(--space-md);
    border-radius: var(--radius);
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    display: grid;
    gap: var(--space-sm);
}

.avatar-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: var(--surface-2);
    display: grid;
    place-items: center;
    font-size: 1.3rem;
    color: var(--text);
}

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
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
