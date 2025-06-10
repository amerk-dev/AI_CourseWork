import pandas as pd
import numpy as np
import tensorflow as tf # Убедитесь, что tensorflow импортирован
from tensorflow import keras
import joblib # Для загрузки scaler, label_encoder, columns

# --- Функция для предобработки новых данных ---
def preprocess_new_data(new_data_df, scaler_path, trained_cols_path, original_num_cols, cat_cols_for_ohe):
    """
    Предобрабатывает новые данные для модели.
    new_data_df: DataFrame с новыми данными.
    scaler_path: Путь к сохраненному scaler.
    trained_cols_path: Путь к файлу со списком колонок обучающей выборки.
    original_num_cols: Список исходных числовых колонок для масштабирования.
    cat_cols_for_ohe: Список категориальных колонок для One-Hot Encoding (включая 'Job').
    """
    # Загрузка сохраненных объектов
    scaler = joblib.load(scaler_path)
    trained_columns = joblib.load(trained_cols_path)

    # Обработка 'NA' и пропусков (аналогично обучающим данным)
    new_data_df.replace('NA', np.nan, inplace=True)
    for col in ['Saving accounts', 'Checking account']:
        if col in new_data_df.columns:
            new_data_df[col].fillna('Unknown', inplace=True) # Такая же стратегия, как при обучении

    # Кодирование категориальных признаков (One-Hot Encoding)
    # 'Job' тоже обрабатывается как категориальный для OHE
    new_data_df = pd.get_dummies(new_data_df, columns=cat_cols_for_ohe, drop_first=True)

    # Приведение набора колонок к тому, что был при обучении
    # Добавляем недостающие колонки (если какая-то категория не встретилась в новых данных)
    # и заполняем их нулями. Удаляем лишние колонки (если появилась новая категория).
    # Порядок колонок также будет соответствовать trained_columns.
    new_data_df = new_data_df.reindex(columns=trained_columns, fill_value=0)

    # Масштабирование числовых признаков
    # Убедимся, что колонки существуют перед масштабированием
    cols_to_scale = [col for col in original_num_cols if col in new_data_df.columns]
    if cols_to_scale:
        new_data_df[cols_to_scale] = scaler.transform(new_data_df[cols_to_scale])
    else:
        print("Предупреждение: Ни одна из числовых колонок для масштабирования не найдена в новых данных.")

    return new_data_df

# --- Загрузка модели и предсказание ---

# Пути к сохраненным файлам
model_path = 'models/credit_risk_model.keras'  # Если сохраняли как SavedModel
scaler_path = 'scaler.joblib'
risk_label_encoder_path = 'risk_label_encoder.joblib'
trained_model_columns_path = 'trained_model_columns.joblib'

# Загрузка модели
loaded_model = keras.models.load_model(model_path)
print("Модель успешно загружена.")
loaded_model.summary() # Можно посмотреть архитектуру

# Загрузка LabelEncoder для расшифровки предсказаний
risk_label_encoder = joblib.load(risk_label_encoder_path)

# Пример новых данных (одна или несколько записей)
# Структура должна быть такой же, как у исходного CSV, ДО предобработки
new_raw_data = {
    'Age': [30, 45, 25],
    'Sex': ['male', 'female', 'male'],
    'Job': [2, 1, 3], # 0,1,2,3
    'Housing': ['own', 'rent', 'free'],
    'Saving accounts': ['little', 'NA', 'moderate'],
    'Checking account': ['moderate', 'little', 'NA'],
    'Credit amount': [2000, 8000, 1500],
    'Duration': [12, 36, 24],
    'Purpose': ['car', 'education', 'radio/TV']
}
new_df = pd.DataFrame(new_raw_data)

print("\nНовые данные (до предобработки):")
print(new_df)

# Колонки, которые были изначально категориальными (для get_dummies) + 'Job'
# Важно: этот список должен соответствовать тому, что использовался при обучении
# Мы его определили так: categorical_cols = X.select_dtypes(include=['object']).columns
# И потом добавили 'Job', если он не был object.
# В нашем случае это: 'Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Job'
categorical_cols_for_ohe = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Job']
original_numerical_cols = ['Age', 'Credit amount', 'Duration'] # Исходные числовые

# Предобработка новых данных
new_df_processed = preprocess_new_data(new_df.copy(), scaler_path, trained_model_columns_path, original_numerical_cols, categorical_cols_for_ohe)

print("\nНовые данные (после предобработки):")
print(new_df_processed.head())
print(f"Размерность обработанных данных: {new_df_processed.shape}")


# Получение предсказаний
predictions_proba = loaded_model.predict(new_df_processed)
# predictions_proba будет массивом вероятностей. Если > 0.5, то класс 1 (good), иначе 0 (bad)
predictions_class = (predictions_proba > 0.5).astype(int).flatten()

# Расшифровка предсказанных классов
predicted_risk_labels = risk_label_encoder.inverse_transform(predictions_class)

# Вывод результатов
results = pd.DataFrame({
    'Predicted_Probability_Good': predictions_proba.flatten(),
    'Predicted_Class_Encoded': predictions_class,
    'Predicted_Risk_Label': predicted_risk_labels
})

# Объединим с исходными данными для наглядности (опционально)
# Важно: new_df_original должен быть копией до всех изменений
final_results = pd.concat([new_df.reset_index(drop=True), results], axis=1)

print("\nРезультаты предсказания для новых данных:")
print(final_results)

# Пример предсказания для одного клиента
if not new_df_processed.empty:
    single_client_processed = new_df_processed.iloc[[0]] # Берем первую запись
    single_prediction_proba = loaded_model.predict(single_client_processed)
    single_prediction_class = (single_prediction_proba > 0.5).astype(int).item() # .item() чтобы получить скаляр
    single_predicted_label = risk_label_encoder.inverse_transform([single_prediction_class])[0]

    print(f"\nПредсказание для первого клиента:")
    print(f"  Исходные данные: {new_raw_data['Age'][0]}, {new_raw_data['Sex'][0]}, ...") # Можно вывести все поля
    print(f"  Вероятность 'good' риска: {single_prediction_proba.item():.4f}")
    print(f"  Предсказанный класс (0=bad, 1=good): {single_prediction_class}")
    print(f"  Предсказанная метка риска: {single_predicted_label}")