# credit_risk_app/app.py
from flask import Flask, request, render_template, jsonify, send_from_directory  # <--- Добавлено send_from_directory
import pandas as pd
import numpy as np
import os  # <--- Добавлено

from src.predict import make_prediction
from src.config import (
    CATEGORICAL_COLS_FOR_OHE, ORIGINAL_NUMERICAL_COLS,
    MODEL_DIR  # <--- Добавлено для пути к изображению дерева
)

app = Flask(__name__)

# ... (остальной код app.py, включая FORM_FIELDS, OPTIONS, @app.route('/'), @app.route('/api/predict')) ...
# (Код для @app.route('/') и @app.route('/api/predict') остается таким же, как в предыдущем ответе)
# ... (весь код до if __name__ == '__main__':)

# Копируем сюда предыдущий код для index и api_predict
# Определяем поля, которые должны быть в форме.
# Мы возьмем их из конфигурации, чтобы быть консистентными.
# 'Job' уже есть в CATEGORICAL_COLS_FOR_OHE
FORM_FIELDS_NUMERICAL = ORIGINAL_NUMERICAL_COLS
FORM_FIELDS_CATEGORICAL = [col for col in CATEGORICAL_COLS_FOR_OHE if col != 'Job']  # 'Job' отдельно обработаем

# Возможные значения для категориальных полей (можно расширить на основе уникальных значений датасета)
# Для 'NA'/'Unknown' в select можно добавить опцию с value="" или "Unknown"
SEX_OPTIONS = ['male', 'female']
HOUSING_OPTIONS = ['own', 'rent', 'free']
SAVING_ACCOUNTS_OPTIONS = ['Unknown', 'little', 'moderate', 'quite rich', 'rich']  # Добавили 'Unknown'
CHECKING_ACCOUNT_OPTIONS = ['Unknown', 'little', 'moderate', 'rich']  # Добавили 'Unknown'
PURPOSE_OPTIONS = ['radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'repairs', 'domestic appliance',
                   'others', 'vacation']  # Расширил немного
JOB_OPTIONS = [
    {"value": 0, "label": "0: Unskilled - non-resident"},
    {"value": 1, "label": "1: Unskilled - resident"},
    {"value": 2, "label": "2: Skilled"},
    {"value": 3, "label": "3: Highly skilled"}
]


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result_display = None
    form_data = {}  # Для сохранения введенных пользователем значений

    if request.method == 'POST':
        try:
            raw_form_data = request.form.to_dict()
            form_data = raw_form_data.copy()

            input_data_for_prediction = {}
            for field in FORM_FIELDS_NUMERICAL:
                value = raw_form_data.get(field)
                if value:
                    input_data_for_prediction[field] = float(value) if '.' in value else int(value)

            for field in FORM_FIELDS_CATEGORICAL:
                value = raw_form_data.get(field)
                if value == "Unknown" or not value:
                    input_data_for_prediction[field] = None
                else:
                    input_data_for_prediction[field] = value

            job_value = raw_form_data.get('Job')
            if job_value:
                input_data_for_prediction['Job'] = int(job_value)

            all_expected_fields = ORIGINAL_NUMERICAL_COLS + CATEGORICAL_COLS_FOR_OHE
            for field_name in all_expected_fields:
                if field_name not in input_data_for_prediction:
                    if field_name in CATEGORICAL_COLS_FOR_OHE:
                        input_data_for_prediction[field_name] = None
                    else:
                        app.logger.warning(f"Поле {field_name} отсутствует во входных данных для предсказания.")

            app.logger.info(f"Данные для предсказания: {input_data_for_prediction}")
            prediction = make_prediction(input_data_for_prediction)

            if prediction and "error" not in prediction:
                prediction_result_display = {
                    "label": prediction.get('predicted_risk_label', 'N/A'),
                    "probability_good": f"{prediction.get('probability_good', 0) * 100:.2f}%"
                }
            elif prediction and "error" in prediction:
                prediction_result_display = {"error": prediction["error"]}
            else:
                prediction_result_display = {"error": "Не удалось получить предсказание."}

        except Exception as e:
            app.logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            prediction_result_display = {"error": f"Произошла внутренняя ошибка: {str(e)}"}

    return render_template('index.html',
                           numerical_fields=FORM_FIELDS_NUMERICAL,
                           categorical_fields_data={
                               "Sex": SEX_OPTIONS,
                               "Housing": HOUSING_OPTIONS,
                               "Saving accounts": SAVING_ACCOUNTS_OPTIONS,
                               "Checking account": CHECKING_ACCOUNT_OPTIONS,
                               "Purpose": PURPOSE_OPTIONS
                           },
                           job_options=JOB_OPTIONS,
                           prediction=prediction_result_display,
                           form_data=form_data
                           )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Input data should be a JSON object"}), 400
    try:
        prediction_result = make_prediction(data)
        if "error" in prediction_result:
            return jsonify(prediction_result), 400
        return jsonify(prediction_result)
    except Exception as e:
        app.logger.error(f"API Error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during prediction"}), 500


# Новый маршрут для отображения дерева решений
@app.route('/decision_tree')
def decision_tree_page():
    # Путь к изображению дерева. Имя файла должно совпадать с тем, что генерируется в model_training.py
    tree_image_filename = 'decision_tree.png'
    tree_image_exists = os.path.exists(os.path.join(MODEL_DIR, tree_image_filename))
    return render_template('decision_tree.html', tree_image_filename=tree_image_filename,
                           tree_image_exists=tree_image_exists)


# Маршрут для отдачи статических файлов (изображения дерева) из папки models
# Flask по умолчанию ищет статику в папке 'static'. Мы делаем это для 'models'.
@app.route('/models/<filename>')
def serve_model_file(filename):
    return send_from_directory(MODEL_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)