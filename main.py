import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf # <--- ДОБАВЛЕНА ЭТА СТРОКА
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import io # Для чтения строки как файла

# 1. Загрузка данных
# Используем io.StringIO для чтения данных из строки
csv_data = """
,Age,Sex,Job,Housing,Saving accounts,Checking account,Credit amount,Duration,Purpose,Risk
0,67,male,2,own,NA,little,1169,6,radio/TV,good
1,22,female,2,own,little,moderate,5951,48,radio/TV,bad
2,49,male,1,own,little,NA,2096,12,education,good
3,45,male,2,free,little,little,7882,42,furniture/equipment,good
4,53,male,2,free,little,little,4870,24,car,bad
5,35,male,1,free,NA,NA,9055,36,education,good
6,53,male,2,own,quite rich,NA,2835,24,furniture/equipment,good
7,35,male,3,rent,little,moderate,6948,36,car,good
8,61,male,1,own,rich,NA,3059,12,radio/TV,good
9,28,male,3,own,little,moderate,5234,30,car,bad
10,25,female,2,rent,little,moderate,1295,12,car,bad
11,24,female,2,rent,little,little,4308,48,business,bad
12,22,female,2,own,little,moderate,1567,12,radio/TV,good
13,60,male,1,own,little,little,1199,24,car,bad
14,28,female,2,rent,little,little,1403,15,car,good
15,32,female,1,own,moderate,little,1282,24,radio/TV,bad
16,53,male,2,own,NA,NA,2424,24,radio/TV,good
17,25,male,2,own,NA,little,8072,30,business,good
18,44,female,3,free,little,moderate,12579,24,car,bad
19,31,male,2,own,quite rich,NA,3430,24,radio/TV,good
"""

df = pd.read_csv(io.StringIO(csv_data))

# Удалим первый столбец (индекс из CSV)
df = df.iloc[:, 1:]

print("Исходные данные (первые 5 строк):")
print(df.head())
print("\nИнформация о данных:")
df.info()
print("\nПропущенные значения до обработки:")
print(df.isnull().sum()) # 'NA' читается как строка, а не как NaN pandas'ом

# 2. Предобработка данных

# Замена 'NA' на np.nan, чтобы pandas корректно их обрабатывал как пропуски
df.replace('NA', np.nan, inplace=True)

# Обработка пропущенных значений
# Для категориальных: заполним модой или специальным значением 'Unknown'
for col in ['Saving accounts', 'Checking account']:
    df[col].fillna('Unknown', inplace=True)

print("\nПропущенные значения после обработки 'NA' и заполнения:")
print(df.isnull().sum()) # Должны быть нули

# Кодирование целевой переменной 'Risk'
label_encoder_risk = LabelEncoder()
df['Risk'] = label_encoder_risk.fit_transform(df['Risk'])
# 'good' -> 1, 'bad' -> 0 (или наоборот, зависит от порядка)
# Давайте проверим:
# print(label_encoder_risk.classes_) # ['bad', 'good'] -> bad=0, good=1. Отлично.

# Разделение признаков (X) и целевой переменной (y)
X = df.drop('Risk', axis=1)
y = df['Risk']

# Определение категориальных и числовых признаков
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns # Job тоже числовой, но категориальный по смыслу

print(f"\nКатегориальные колонки: {list(categorical_cols)}")
print(f"Числовые колонки: {list(numerical_cols)}")

# Кодирование категориальных признаков (One-Hot Encoding)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True) # drop_first для избежания мультиколлинеарности

# 'Job' уже числовой (0, 1, 2, 3), но это категории.
# Если мы считаем их порядковыми, можно оставить как есть.
# Если нет, то лучше тоже One-Hot Encode. Давайте сделаем One-Hot для 'Job' тоже.
# Если 'Job' еще не в categorical_cols, добавим его и переделаем get_dummies
if 'Job' in X.columns and 'Job' not in categorical_cols: # Проверка, что 'Job' еще не обработан
    # Если Job не был object, он не попал в categorical_cols
    # Его нужно явно превратить в категорию для get_dummies, если он не является порядковым
    # В нашем случае, он уже числовой, но мы хотим его как one-hot.
    # Для простоты предположим, что 'Job' - это номинальный признак
    X = pd.get_dummies(X, columns=['Job'], drop_first=True)


print("\nПризнаки X после One-Hot Encoding (первые 5 строк):")
print(X.head())
print(f"\nРазмерность X после One-Hot Encoding: {X.shape}")


# Масштабирование числовых признаков
# Числовые признаки, которые остались после one-hot (исходные числовые)
# numerical_cols теперь нужно обновить, так как get_dummies мог создать новые числовые колонки
# Но мы масштабируем только ИСХОДНЫЕ числовые.
original_numerical_cols = ['Age', 'Credit amount', 'Duration'] # Это те, что были изначально числовыми

scaler = StandardScaler()
X[original_numerical_cols] = scaler.fit_transform(X[original_numerical_cols])

print("\nПризнаки X после масштабирования (первые 5 строк):")
print(X.head())

# 3. Разделение данных
# Из-за малого размера данных, тестовая выборка будет очень маленькой.
# Установим random_state для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nРазмер обучающей выборки: {X_train.shape}, {y_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}, {y_test.shape}")
print(f"Распределение классов в y_train:\n{y_train.value_counts(normalize=True)}")
print(f"Распределение классов в y_test:\n{y_test.value_counts(normalize=True)}")


# 4. Создание модели нейронной сети
# Установим seed для TensorFlow для воспроизводимости
tf.random.set_seed(42)
np.random.seed(42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Входной слой
    Dropout(0.3), # Слой Dropout для регуляризации
    Dense(32, activation='relu'), # Скрытый слой
    Dropout(0.3), # Слой Dropout
    Dense(1, activation='sigmoid') # Выходной слой для бинарной классификации
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5. Обучение модели
# Из-за малого размера данных, используем небольшой batch_size и немного эпох
# Добавим EarlyStopping для предотвращения сильного переобучения
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10, # Количество эпох без улучшения, после которых обучение остановится
    restore_best_weights=True # Восстановить веса модели с лучшей val_loss
)

history = model.fit(X_train, y_train,
                    epochs=100, # Может быть больше, early stopping поможет
                    batch_size=4, # Маленький batch_size для маленького датасета
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=1) # verbose=1 для отображения прогресса

# 6. Оценка модели
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nПотери на тестовой выборке: {loss:.4f}")
print(f"Точность на тестовой выборке: {accuracy:.4f}")

# Получение предсказаний
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten() # Преобразование вероятностей в классы 0 или 1

print("\nОтчет по классификации:")
# target_names можно взять из label_encoder_risk.classes_
print(classification_report(y_test, y_pred, target_names=label_encoder_risk.classes_, zero_division=0))

print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred))

print("\nСравнение истинных и предсказанных значений на тестовой выборке:")
results_df = pd.DataFrame({'Actual': y_test, 'Predicted_Proba': y_pred_proba.flatten(), 'Predicted_Class': y_pred})
print(results_df)

# 7. Сохранение модели
model_path = 'models/credit_risk_model.keras'  # ИЛИ .h5, но .keras предпочтительнее для Keras 3
model.save(model_path)
print(f"\nМодель сохранена в: {model_path}")

# Также нам нужно сохранить объекты, необходимые для предобработки новых данных:
# - StandardScaler (scaler)
# - LabelEncoder для целевой переменной (label_encoder_risk)
# - Список колонок, которые были у X_train (для правильного OneHotEncoding новых данных)
import joblib

joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder_risk, 'risk_label_encoder.joblib')
# Сохраним список колонок X_train, чтобы новые данные имели те же признаки
# Это важно, так как OneHotEncoder может создать разное количество колонок
# в зависимости от уникальных значений в данных.
trained_columns = list(X_train.columns)
joblib.dump(trained_columns, 'trained_model_columns.joblib')

print("Объекты для предобработки (scaler, label_encoder, columns) сохранены.")

# Построим график обучения, если нужно (но с таким датасетом он может быть не очень информативен)
import matplotlib.pyplot as plt

if history and hasattr(history, 'history'):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title('Точность модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('Потери модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.tight_layout()
    plt.show()