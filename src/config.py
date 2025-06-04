# src/config.py
import os

# Базовая директория проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # credit_risk_app/

# Пути к данным и моделям
DATA_PATH = os.path.join(BASE_DIR, 'data', 'german_credit_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_risk_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'risk_label_encoder.joblib')
TRAINED_COLUMNS_PATH = os.path.join(MODEL_DIR, 'trained_model_columns.joblib')

# Параметры модели и данных
TARGET_COLUMN = 'Risk'
# Исходные числовые колонки (до OneHotEncoding)
ORIGINAL_NUMERICAL_COLS = ['Age', 'Credit amount', 'Duration']
# Категориальные колонки для OneHotEncoding (включая 'Job', если он обрабатывается как категориальный)
CATEGORICAL_COLS_FOR_OHE = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Job']

# Параметры обучения
TEST_SIZE = 0.25
RANDOM_STATE = 42
EPOCHS = 100 # Может быть больше, early stopping поможет
BATCH_SIZE = 4 # Маленький batch_size для маленького датасета
PATIENCE_EARLY_STOPPING = 10

# Убедимся, что директория для моделей существует
os.makedirs(MODEL_DIR, exist_ok=True)