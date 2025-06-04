# src/model_training.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import joblib

from .config import (
    DATA_PATH, MODEL_PATH, LABEL_ENCODER_PATH,
    RANDOM_STATE, EPOCHS, BATCH_SIZE, PATIENCE_EARLY_STOPPING
)
from .data_preprocessing import get_train_test_data

def build_model(input_shape):
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X_train, X_test, y_train, y_test = get_train_test_data(DATA_PATH)

    model = build_model(X_train.shape[1])
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE_EARLY_STOPPING,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Оценка
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nПотери на тестовой выборке: {loss:.4f}")
    print(f"Точность на тестовой выборке: {accuracy:.4f}")

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    label_encoder_risk = joblib.load(LABEL_ENCODER_PATH)
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred, target_names=label_encoder_risk.classes_, zero_division=0))
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, y_pred))

    # Сохранение модели
    model.save(MODEL_PATH)
    print(f"\nМодель сохранена в: {MODEL_PATH}")

    # Графики обучения (опционально, но полезно)
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

if __name__ == '__main__':
    train_model()