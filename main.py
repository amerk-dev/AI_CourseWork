import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Конфигурация
MODEL_PATH = "models/credit_risk_model-v1.joblib"
DATA_PATH = "data/german_credit_data.csv"


def preprocess_data(df):
    """Предобработка данных"""
    df = df.copy()
    # Заполнение пропущенных значений
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')

    # Кодирование категориальных переменных
    label_encoders = {}
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Создание целевой переменной
    df['Credit amount'] = df['Credit amount'].astype(float)
    y = df['Credit amount'] > df['Credit amount'].median()
    X = df.drop(columns=['Credit amount'])

    return X, y, label_encoders


def train_and_save_model():
    """Обучение и сохранение модели"""
    df = pd.read_csv(DATA_PATH, index_col=0)
    X, y, label_encoders = preprocess_data(df)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Сохранение артефактов
    artifacts = {
        'model': model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_columns': X.columns.tolist()
    }
    joblib.dump(artifacts, MODEL_PATH)
    print("\nМодель успешно обучена и сохранена")

    return model, scaler, label_encoders


def load_model():
    """Загрузка сохраненной модели"""
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None

    artifacts = joblib.load(MODEL_PATH)
    return (
        artifacts['model'],
        artifacts['scaler'],
        artifacts['label_encoders'],
        artifacts['feature_columns']  # Добавляем feature_columns в возвращаемые значения
    )


def predict_credit_risk(model, scaler, label_encoders, feature_columns):
    """Интерактивное предсказание риска"""
    try:
        user_data = {}
        print("\nВведите данные клиента:")
        user_data['Age'] = int(input("Возраст: "))
        user_data['Sex'] = input("Пол (male/female): ").lower()
        user_data['Job'] = int(input("Профессия (0-3): "))
        user_data['Housing'] = input("Жильё (own/rent/free): ").lower()
        user_data['Saving accounts'] = input("Сберегательный счёт (little/moderate/quite rich/rich/unknown): ").lower()
        user_data['Checking account'] = input("Текущий счёт (little/moderate/quite rich/rich/unknown): ").lower()
        user_data['Duration'] = int(input("Срок кредита (месяцев): "))
        user_data['Purpose'] = input("Цель кредита: ").strip().lower()

        # Кодирование категориальных признаков
        encoded_data = {}
        for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
            le = label_encoders[col]
            value = user_data[col]

            if value not in le.classes_:
                raise ValueError(f"Недопустимое значение для {col}. Допустимые: {list(le.classes_)}")

            encoded_data[col] = le.transform([value])[0]

        # Создание DataFrame
        input_data = {
            'Age': user_data['Age'],
            'Sex': encoded_data['Sex'],
            'Job': user_data['Job'],
            'Housing': encoded_data['Housing'],
            'Saving accounts': encoded_data['Saving accounts'],
            'Checking account': encoded_data['Checking account'],
            'Duration': user_data['Duration'],
            'Purpose': encoded_data['Purpose']
        }

        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Масштабирование
        scaled_input = scaler.transform(input_df)

        # Предсказание
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0]

        print("\nРезультат оценки:")
        print(f"Прогноз: {'Высокий риск' if prediction[0] else 'Низкий риск'}")
        print(f"Вероятность: Низкий риск - {probability[0] * 100:.1f}%, Высокий риск - {probability[1] * 100:.1f}%")

    except Exception as e:
        print(f"\nОшибка: {str(e)}")


if __name__ == "__main__":
    # Загрузка или обучение модели
    model, scaler, label_encoders, feature_columns = load_model()

    if model is None:
        print("Модель не найдена, начинаем обучение...")
        model, scaler, label_encoders = train_and_save_model()
    else:
        print("Модель успешно загружена")

    # Запуск интерактивного предсказания
    while True:
        predict_credit_risk(model, scaler, label_encoders, feature_columns)
        repeat = input("\nСделать еще одно предсказание? (y/n): ").lower()
        if repeat != 'y':
            break