# credit_risk_app/app.py
from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np  # Может понадобиться для обработки некоторых значений из формы

# Импортируем нашу функцию предсказания и конфигурацию (для полей формы)
from src.predict import make_prediction
from src.config import CATEGORICAL_COLS_FOR_OHE, ORIGINAL_NUMERICAL_COLS

app = Flask(__name__)

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
            form_data = raw_form_data.copy()  # Сохраняем для повторного отображения в форме

            # Преобразуем числовые поля из строки в число
            # и обработаем пустые строки для опциональных категориальных полей
            input_data_for_prediction = {}
            for field in FORM_FIELDS_NUMERICAL:
                value = raw_form_data.get(field)
                if value:
                    input_data_for_prediction[field] = float(value) if '.' in value else int(value)
                else:
                    # Можно установить значение по умолчанию или вернуть ошибку, если поле обязательное
                    # Для нашего случая, если числовое поле не заполнено, модель может выдать ошибку при предобработке
                    # или preprocess_features должен уметь это обрабатывать (например, импутацией)
                    # Пока что сделаем их обязательными на стороне клиента (HTML) или здесь вернем ошибку.
                    # Для простоты, предположим, что они будут заполнены (или HTML5 validation сработает)
                    pass  # или input_data_for_prediction[field] = None и обработать в preprocess

            for field in FORM_FIELDS_CATEGORICAL:
                value = raw_form_data.get(field)
                # Если 'Unknown' передается как пустая строка из select, или мы хотим None
                if value == "Unknown" or not value:  # Пустая строка или "Unknown"
                    input_data_for_prediction[field] = None  # или 'Unknown' если так обрабатывается в preprocess
                else:
                    input_data_for_prediction[field] = value

            # 'Job' - числовой, но категориальный по смыслу
            job_value = raw_form_data.get('Job')
            if job_value:
                input_data_for_prediction['Job'] = int(job_value)

            # Убедимся, что все ожидаемые поля есть, даже если они None
            # Это важно, т.к. make_prediction ожидает все ключи, которые были при обучении
            # (хотя preprocess_features с reindex должен это покрывать)
            all_expected_fields = ORIGINAL_NUMERICAL_COLS + CATEGORICAL_COLS_FOR_OHE
            for field_name in all_expected_fields:
                if field_name not in input_data_for_prediction:
                    # Это может произойти, если поле не было в форме или не обработано выше
                    # Для категориальных, если не выбрано, может быть None или "Unknown"
                    if field_name in CATEGORICAL_COLS_FOR_OHE:
                        input_data_for_prediction[field_name] = None  # или 'Unknown'
                    else:  # Числовые лучше сделать обязательными
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
                           categorical_fields_data={  # Данные для генерации select-ов
                               "Sex": SEX_OPTIONS,
                               "Housing": HOUSING_OPTIONS,
                               "Saving accounts": SAVING_ACCOUNTS_OPTIONS,
                               "Checking account": CHECKING_ACCOUNT_OPTIONS,
                               "Purpose": PURPOSE_OPTIONS
                           },
                           job_options=JOB_OPTIONS,
                           prediction=prediction_result_display,
                           form_data=form_data  # Передаем введенные данные обратно в шаблон
                           )


# Можно добавить эндпоинт API, если нужно
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()

    # Простая валидация (можно улучшить)
    # Убедимся, что data - это dict (для одного предсказания)
    if not isinstance(data, dict):
        return jsonify({"error": "Input data should be a JSON object"}), 400

    # Здесь можно добавить более строгую валидацию ключей и типов данных

    try:
        prediction_result = make_prediction(data)
        if "error" in prediction_result:
            return jsonify(prediction_result), 400  # или 500 в зависимости от ошибки
        return jsonify(prediction_result)
    except Exception as e:
        app.logger.error(f"API Error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during prediction"}), 500


if __name__ == '__main__':
    # Перед запуском убедитесь, что модель обучена!
    # python -m src.model_training
    app.run(debug=True, port=5000)