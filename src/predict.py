# src/predict.py
import pandas as pd
import numpy as np
import tensorflow as tf  # Убедитесь, что tensorflow импортирован
from tensorflow import keras
import joblib

from .config import (
    MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, TRAINED_COLUMNS_PATH,
    ORIGINAL_NUMERICAL_COLS, CATEGORICAL_COLS_FOR_OHE
)
from .data_preprocessing import preprocess_features  # Используем ту же функцию

# Глобальная загрузка артефактов, чтобы не делать это при каждом вызове predict
try:
    LOADED_MODEL = keras.models.load_model(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    RISK_LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)
    TRAINED_COLUMNS = joblib.load(TRAINED_COLUMNS_PATH)
    print("Модель и артефакты для предсказания успешно загружены.")
except Exception as e:
    print(f"Ошибка при загрузке модели или артефактов: {e}")
    print("Убедитесь, что модель была обучена и артефакты сохранены (запустите model_training.py).")
    LOADED_MODEL, SCALER, RISK_LABEL_ENCODER, TRAINED_COLUMNS = None, None, None, None


def make_prediction(raw_input_data):
    """
    Делает предсказание для одного или нескольких входных образцов.
    raw_input_data: dict или list of dicts, или pd.DataFrame
                    с сырыми данными (колонки как в исходном CSV).
    """
    if not all([LOADED_MODEL, SCALER, RISK_LABEL_ENCODER, TRAINED_COLUMNS]):
        return {"error": "Модель или артефакты не загружены. Сначала обучите модель."}

    if isinstance(raw_input_data, dict):
        input_df = pd.DataFrame([raw_input_data])
    elif isinstance(raw_input_data, list) and all(isinstance(item, dict) for item in raw_input_data):
        input_df = pd.DataFrame(raw_input_data)
    elif isinstance(raw_input_data, pd.DataFrame):
        input_df = raw_input_data.copy()
    else:
        return {"error": "Неверный формат входных данных. Ожидается dict, list of dicts или DataFrame."}

    # Предобработка входных данных с использованием загруженных scaler и trained_columns
    # fit_scaler=False, так как мы используем уже обученный scaler
    processed_df, _ = preprocess_features(
        input_df,
        scaler=SCALER,
        fit_scaler=False,
        trained_columns=TRAINED_COLUMNS
    )

    # Предсказание
    predictions_proba = LOADED_MODEL.predict(processed_df)
    predictions_class_encoded = (predictions_proba > 0.5).astype(int).flatten()
    predicted_risk_labels = RISK_LABEL_ENCODER.inverse_transform(predictions_class_encoded)

    results = []
    for i in range(len(input_df)):
        result = {
            "input_data": input_df.iloc[i].to_dict(),  # Исходные данные, которые подали на вход этой функции
            "probability_good": float(predictions_proba[i][0]),
            "predicted_class_encoded": int(predictions_class_encoded[i]),
            "predicted_risk_label": predicted_risk_labels[i]
        }
        results.append(result)

    return results if len(results) > 1 else results[0]


# Пример использования, если запускаем этот файл напрямую
if __name__ == '__main__':
    # Пример новых данных (одна или несколько записей)
    # Структура должна быть такой же, как у исходного CSV, ДО предобработки
    sample_new_data_single = {
        'Age': 30, 'Sex': 'male', 'Job': 2, 'Housing': 'own',
        'Saving accounts': 'little', 'Checking account': 'moderate',
        'Credit amount': 2000, 'Duration': 12, 'Purpose': 'car'
    }

    sample_new_data_multiple = [
        {
            'Age': 30, 'Sex': 'male', 'Job': 2, 'Housing': 'own',
            'Saving accounts': 'little', 'Checking account': 'moderate',
            'Credit amount': 2000, 'Duration': 12, 'Purpose': 'car'
        },
        {
            'Age': 45, 'Sex': 'female', 'Job': 1, 'Housing': 'rent',
            'Saving accounts': None, 'Checking account': 'little',  # Пример с NA/None
            'Credit amount': 8000, 'Duration': 36, 'Purpose': 'education'
        }
    ]

    print("\n--- Предсказание для одного клиента ---")
    prediction_single = make_prediction(sample_new_data_single)
    if prediction_single:
        print(prediction_single)

    print("\n--- Предсказание для нескольких клиентов ---")
    predictions_multiple = make_prediction(sample_new_data_multiple)
    if predictions_multiple:
        for pred in predictions_multiple:
            print(pred)