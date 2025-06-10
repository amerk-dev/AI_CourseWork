# src/model_training.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz # <--- Добавлено
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os # <--- Добавлено


from .config import (
    DATA_PATH, MODEL_PATH, LABEL_ENCODER_PATH,
    RANDOM_STATE, EPOCHS, BATCH_SIZE, PATIENCE_EARLY_STOPPING, MODEL_DIR

)
from .data_preprocessing import get_train_test_data

RISK_LABELS_RU_FOR_TREE = {
    'bad': 'Плохой', # Порядок важен, должен совпадать с label_encoder_risk.classes_
    'good': 'Хороший'
}

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


def train_and_visualize_decision_tree(X_train, y_train, feature_names, class_names):
    """Обучает дерево решений и сохраняет его визуализацию."""
    print("\n--- Обучение Дерева Решений ---")
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5) # Ограничим глубину для наглядности
    dt_model.fit(X_train, y_train)

    # Сохранение модели дерева решений (опционально, если нужно потом использовать)
    dt_model_path = os.path.join(MODEL_DIR, 'decision_tree_model.joblib')
    joblib.dump(dt_model, dt_model_path)
    print(f"Модель дерева решений сохранена в: {dt_model_path}")

    # Визуализация дерева
    dot_data = export_graphviz(dt_model, out_file=None,
                               feature_names=feature_names,
                               class_names=[str(c) for c in class_names], # Имена классов должны быть строками
                               filled=True, rounded=True,
                               special_characters=True,
                               max_depth=3) # Ограничим глубину для визуализации

    # Путь для сохранения изображения дерева
    tree_image_path_dot = os.path.join(MODEL_DIR, 'decision_tree.dot')
    tree_image_path_png = os.path.join(MODEL_DIR, 'decision_tree.png')

    # Сохраняем .dot файл (можно открыть в Graphviz)
    with open(tree_image_path_dot, 'w') as f:
        f.write(dot_data)
    print(f"DOT файл дерева сохранен в: {tree_image_path_dot}")

    # Генерируем PNG из DOT
    try:
        graph = graphviz.Source(dot_data, format="png")
        graph.render(filename='decision_tree', directory=MODEL_DIR, cleanup=True) # cleanup=True удалит .dot после PNG
        print(f"Изображение дерева (PNG) сохранено в: {tree_image_path_png}")
    except graphviz.backend.execute.CalledProcessError as e:
        print(f"Ошибка при генерации PNG из DOT: {e}")
        print("Убедитесь, что Graphviz установлен и добавлен в PATH.")
        print("Вы можете вручную конвертировать .dot в .png командой: dot -Tpng {tree_image_path_dot} -o {tree_image_path_png}")
    except Exception as e:
        print(f"Неизвестная ошибка при рендеринге дерева: {e}")

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

    # Оценка нейронной сети
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nПотери на тестовой выборке (NN): {loss:.4f}")
    print(f"Точность на тестовой выборке (NN): {accuracy:.4f}")

    y_pred_proba_nn = model.predict(X_test)
    y_pred_nn = (y_pred_proba_nn > 0.5).astype(int).flatten()

    label_encoder_risk = joblib.load(LABEL_ENCODER_PATH)
    print("\nОтчет по классификации (NN):")
    print(classification_report(y_test, y_pred_nn, target_names=label_encoder_risk.classes_, zero_division=0))
    print("\nМатрица ошибок (NN):")
    print(confusion_matrix(y_test, y_pred_nn))

    model.save(MODEL_PATH)
    print(f"\nМодель нейронной сети сохранена в: {MODEL_PATH}")
    # ... (код для графиков обучения нейросети) ...

    # Обучение и визуализация дерева решений
    # Получаем имена признаков из X_train (DataFrame)
    feature_names = list(X_train.columns)
    # Получаем имена классов из LabelEncoder
    class_names_str = [str(c) for c in label_encoder_risk.classes_]

    train_and_visualize_decision_tree(X_train, y_train, feature_names, class_names_str)

    # Графики обучения (опционально, но полезно)
    if history and hasattr(history, 'history'):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Точность на обучении (NN)')
        plt.plot(history.history['val_accuracy'], label='Точность на валидации (NN)')
        plt.title('Точность модели (NN)')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Потери на обучении (NN)')
        plt.plot(history.history['val_loss'], label='Потери на валидации (NN)')
        plt.title('Потери модели (NN)')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    train_model()