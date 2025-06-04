# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

from .config import (
    TARGET_COLUMN, ORIGINAL_NUMERICAL_COLS, CATEGORICAL_COLS_FOR_OHE,
    SCALER_PATH, LABEL_ENCODER_PATH, TRAINED_COLUMNS_PATH,
    TEST_SIZE, RANDOM_STATE
)

def load_data(path):
    # 'NA' в исходном файле заменены на пустые строки, чтобы pandas их прочитал как NaN
    # если они не были заменены, то na_values=['NA']
    df = pd.read_csv(path, na_values=['NA', ''])
    return df

def preprocess_features(df_input, scaler=None, fit_scaler=False, trained_columns=None):
    """
    Предобрабатывает признаки: обработка пропусков, OHE, масштабирование.
    Если fit_scaler=True, scaler будет обучен и сохранен.
    Если trained_columns предоставлен, приводит df к этому набору колонок.
    """
    df = df_input.copy()

    # Обработка пропущенных значений для категориальных колонок (до OHE)
    for col in ['Saving accounts', 'Checking account']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').replace('', 'Unknown')

    # Кодирование категориальных признаков (One-Hot Encoding)
    # 'Job' также обрабатывается как категориальный для OHE
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS_FOR_OHE, drop_first=True)

    # Приведение набора колонок к тому, что был при обучении (для predict.py)
    if trained_columns:
        df = df.reindex(columns=trained_columns, fill_value=0)
    else:
        # Сохраняем список колонок после OHE (для train_model.py)
        current_columns = list(df.columns)
        joblib.dump(current_columns, TRAINED_COLUMNS_PATH)
        print(f"Список колонок сохранен в {TRAINED_COLUMNS_PATH}")


    # Масштабирование числовых признаков
    # Убедимся, что колонки существуют перед масштабированием
    cols_to_scale = [col for col in ORIGINAL_NUMERICAL_COLS if col in df.columns]

    if fit_scaler:
        scaler = StandardScaler()
        if cols_to_scale:
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler сохранен в {SCALER_PATH}")
    elif scaler: # Используем существующий scaler (для predict.py)
        if cols_to_scale:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    else:
        if cols_to_scale:
            print("Предупреждение: Scaler не предоставлен и не будет обучен. Числовые данные не масштабированы.")

    return df, scaler

def preprocess_target(series_target, fit_encoder=False):
    """ Кодирует целевую переменную. Если fit_encoder=True, обучает и сохраняет LabelEncoder. """
    label_encoder = LabelEncoder()
    if fit_encoder:
        y_encoded = label_encoder.fit_transform(series_target)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        print(f"LabelEncoder для Risk сохранен в {LABEL_ENCODER_PATH}")
        print(f"Классы LabelEncoder: {list(label_encoder.classes_)} -> {list(label_encoder.transform(label_encoder.classes_))}")
    else:
        # Если не обучаем, значит, предполагается, что данные уже закодированы или это для inverse_transform
        # Для простоты, пока что не загружаем энкодер здесь, он будет загружаться в predict.py
        # Но для консистентности, можно было бы и здесь загружать, если бы он был нужен для кодирования y_test например
        y_encoded = series_target # Предполагаем, что уже закодировано, если не fit_encoder

    return y_encoded, label_encoder


def get_train_test_data(data_path):
    """Полный пайплайн подготовки данных для обучения."""
    df = load_data(data_path)
    print("Исходные данные загружены.")
    print(f"Пропущенные значения:\n{df.isnull().sum()}")

    X_raw = df.drop(TARGET_COLUMN, axis=1)
    y_raw = df[TARGET_COLUMN]

    # Предобработка признаков (обучаем scaler)
    X_processed, _ = preprocess_features(X_raw, fit_scaler=True)
    print("Признаки X предобработаны.")

    # Предобработка целевой переменной (обучаем label_encoder)
    y_processed, _ = preprocess_target(y_raw, fit_encoder=True)
    print("Целевая переменная y предобработана.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_processed
    )
    print(f"Данные разделены: X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test