# ui_config.py - Sminex AutoML v0.25 ULTIMATE - Конфигурация UI
# ✅ ВСЕ ИСПРАВЛЕНИЯ: Оптимизированные параметры

APP_VERSION = "v0.25 ULTIMATE"
RANDOM_SEED = 42
PARALLEL_CV = 1  # CV без параллелизма для стабильности (параллелизм в моделях)

MAX_DATASET_SIZE = 100000
SAMPLE_SIZE_FOR_LARGE_DATASETS = 50000

# ============ CSS СТИЛИ (дизайн-система неон / soft-dark) ============
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --font-sans: 'Inter', 'Inter var', 'Space Grotesk', 'Segoe UI', system-ui, -apple-system, sans-serif;
    --font-display: 'Space Grotesk', 'Inter', 'Segoe UI', system-ui;

    --bg: #0c0f1a;
    --bg-2: #11162b;
    --surface: rgba(255, 255, 255, 0.04);
    --surface-2: rgba(255, 255, 255, 0.07);
    --text: #eef2ff;
    --muted: #94a3b8;
    --border: rgba(255, 255, 255, 0.08);
    --accent: #7c6cff;
    --accent-2: #2ee6c5;
    --accent-3: #ffa7c4;
    --glass: rgba(124, 108, 255, 0.12);
    --shadow: 0 14px 50px rgba(0, 0, 0, 0.45);
    --radius: 16px;
    --blur: 18px;

    --space-2xs: 4px;
    --space-xs: 8px;
    --space-sm: 12px;
    --space-md: 16px;
    --space-lg: 24px;
    --space-xl: 32px;
    --space-2xl: 48px;
}

[data-theme="light"] {
    --bg: #f6f8ff;
    --bg-2: #e7edff;
    --surface: rgba(255, 255, 255, 0.9);
    --surface-2: rgba(255, 255, 255, 0.8);
    --text: #0b1021;
    --muted: #4b5563;
    --border: rgba(12, 16, 33, 0.08);
    --accent: #6651ff;
    --accent-2: #00d6b8;
    --accent-3: #ff6fb8;
    --glass: rgba(102, 81, 255, 0.12);
    --shadow: 0 18px 50px rgba(10, 20, 40, 0.12);
}

* { box-sizing: border-box; }

body {
    background: radial-gradient(120% 140% at 20% 20%, rgba(124,108,255,0.24), transparent 40%),
                radial-gradient(80% 90% at 80% 10%, rgba(46,230,197,0.26), transparent 35%),
                var(--bg);
    font-family: var(--font-sans);
    color: var(--text);
    transition: background 0.4s ease, color 0.3s ease;
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
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    color: #0c0f1a;
    border-radius: 14px;
    box-shadow: 0 14px 40px rgba(124, 108, 255, 0.28);
    transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.2s ease;
    padding: 12px 18px;
    font-weight: 700;
    border: none;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 18px 50px rgba(46, 230, 197, 0.35);
    filter: saturate(1.1);
}

.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); filter: saturate(0.95); }
.stButton > button:disabled, .stDownloadButton > button:disabled { background: var(--border); color: var(--muted); box-shadow: none; }

.cta-ghost button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border);
    box-shadow: none;
}

.stSelectbox, .stSlider, .stNumberInput, .stMultiSelect, .stTextInput {
    border-radius: 14px;
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
    background: var(--surface-2);
    border: 1px solid var(--border);
    backdrop-filter: blur(var(--blur));
}

.sticky-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: linear-gradient(120deg, var(--surface-2), var(--surface));
    padding: var(--space-md);
    border-radius: 16px;
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
    box-shadow: 0 8px 20px rgba(124,108,255,0.3);
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
    position: relative;
    overflow: hidden;
}

.ui-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(124,108,255,0.2), transparent 40%);
    pointer-events: none;
}

.ui-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--glass);
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
    background: linear-gradient(135deg, rgba(124,108,255,0.18), rgba(46,230,197,0.14));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: var(--space-xl);
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(255,167,196,0.18), transparent 60%);
    top: -60px;
    right: -40px;
    filter: blur(10px);
}

.hero h1 { font-size: 2.2rem; margin-bottom: var(--space-sm); }
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
    background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.18), rgba(255,255,255,0.06));
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
    background: var(--glass);
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
