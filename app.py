# credit_risk_app/app.py
from flask import Flask, request, render_template, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os

from src.predict import make_prediction
from src.config import (
    CATEGORICAL_COLS_FOR_OHE, ORIGINAL_NUMERICAL_COLS,
    MODEL_DIR
)

app = Flask(__name__)

# --- Русские отображаемые имена и опции ---
DISPLAY_NAMES_NUMERICAL = {
    'Age': 'Возраст',
    'Credit amount': 'Сумма кредита',
    'Duration': 'Срок кредита (месяцы)'
}

DISPLAY_NAMES_CATEGORICAL = {
    'Job': 'Тип занятости',
    'Sex': 'Пол',
    'Housing': 'Тип жилья',
    'Saving accounts': 'Сберегательный счет',
    'Checking account': 'Текущий счет',
    'Purpose': 'Цель кредита'
}

# Русские опции для select-ов
SEX_OPTIONS_RU = {
    'male': 'Мужской',
    'female': 'Женский'
}
HOUSING_OPTIONS_RU = {
    'own': 'Собственное',
    'rent': 'Аренда',
    'free': 'Бесплатное'
}
# Для счетов, 'Unknown' уже обрабатывается в шаблоне как "Не указано"
SAVING_ACCOUNTS_OPTIONS_RU = {
    'little': 'Мало',
    'moderate': 'Средне',
    'quite rich': 'Достаточно много',
    'rich': 'Много'
    # 'Unknown' будет "Не указано"
}
CHECKING_ACCOUNT_OPTIONS_RU = {
    'little': 'Мало',
    'moderate': 'Средне',
    'rich': 'Много'
    # 'Unknown' будет "Не указано"
}
PURPOSE_OPTIONS_RU = {
    'radio/TV': 'Радио/Телевизор',
    'education': 'Образование',
    'furniture/equipment': 'Мебель/Оборудование',
    'car': 'Автомобиль',
    'business': 'Бизнес',
    'repairs': 'Ремонт',
    'domestic appliance': 'Бытовая техника',
    'others': 'Другое',
    'vacation': 'Отпуск/Путешествие'
}
JOB_OPTIONS_RU = [
    {"value": 0, "label": "0: Неквалифицированный (нерезидент)"},
    {"value": 1, "label": "1: Неквалифицированный (резидент)"},
    {"value": 2, "label": "2: Квалифицированный"},
    {"value": 3, "label": "3: Высококвалифицированный"}
]

# Маппинг для отображения опций в шаблоне
# Ключ - имя поля, значение - словарь {оригинальное_значение: русское_значение}
DISPLAY_OPTIONS_MAP = {
    'Sex': SEX_OPTIONS_RU,
    'Housing': HOUSING_OPTIONS_RU,
    'Saving accounts': SAVING_ACCOUNTS_OPTIONS_RU,
    'Checking account': CHECKING_ACCOUNT_OPTIONS_RU,
    'Purpose': PURPOSE_OPTIONS_RU,
    # Job обрабатывается отдельно через job_options_ru
}

# Русские метки для предсказаний
RISK_LABELS_RU = {
    'good': 'Хороший',
    'bad': 'Плохой'
}


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result_display = None
    form_data = {}

    if request.method == 'POST':
        try:
            raw_form_data = request.form.to_dict()
            form_data = raw_form_data.copy()

            input_data_for_prediction = {}
            for field_key in DISPLAY_NAMES_NUMERICAL.keys():  # Используем ключи из нашего словаря
                value = raw_form_data.get(field_key)
                if value:
                    input_data_for_prediction[field_key] = float(value) if '.' in value else int(value)
                # else: обработка обязательных полей, если нужно

            # 'Job' отдельно
            job_value = raw_form_data.get('Job')
            if job_value:
                input_data_for_prediction['Job'] = int(job_value)

            for field_key in DISPLAY_NAMES_CATEGORICAL.keys():
                if field_key == 'Job': continue  # Уже обработали
                value = raw_form_data.get(field_key)
                # "Unknown" или пустая строка из select обрабатывается в preprocess_features
                # Поэтому передаем как есть или None если пустая строка (если поле не required)
                if value == "Unknown" or not value:
                    input_data_for_prediction[field_key] = "Unknown"  # Или None, если preprocess_features ожидает None
                else:
                    input_data_for_prediction[field_key] = value

            app.logger.info(
                f"Данные для предсказания (сырые из формы, но ключи как в модели): {input_data_for_prediction}")

            # Делаем предсказание
            prediction_raw = make_prediction(
                input_data_for_prediction)  # функция make_prediction вернет {'predicted_risk_label': 'good', ...}

            if prediction_raw and "error" not in prediction_raw:
                # Русифицируем метку
                original_label = prediction_raw.get('predicted_risk_label', 'N/A')
                prediction_result_display = {
                    "label": RISK_LABELS_RU.get(original_label, original_label),  # Получаем русскую метку
                    "probability_good": f"{prediction_raw.get('probability_good', 0) * 100:.2f}%"
                }
            elif prediction_raw and "error" in prediction_raw:
                prediction_result_display = {
                    "error": prediction_raw["error"]}  # Ошибки оставляем как есть или тоже русифицируем
            else:
                prediction_result_display = {"error": "Не удалось получить предсказание."}  # Русская ошибка

        except Exception as e:
            app.logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            prediction_result_display = {
                "error": f"Произошла внутренняя ошибка. Пожалуйста, попробуйте позже."}  # Русская ошибка

    # Собираем данные для передачи в шаблон
    categorical_fields_for_template = {}
    # Имена ключей, которые мы хотим передать в шаблон для генерации select-ов
    categorical_keys_for_select = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    for key_original in categorical_keys_for_select:
        # Формируем имя переменной с опциями, заменяя пробел на подчеркивание
        options_variable_name = key_original.replace(' ', '_').upper() + '_OPTIONS_RU'
        if options_variable_name in globals():
            # globals()[options_variable_name] это, например, SEX_OPTIONS_RU
            # .keys() вернет ['male', 'female']
            categorical_fields_for_template[key_original] = list(globals()[options_variable_name].keys())
        else:
            app.logger.warning(f"Переменная с опциями {options_variable_name} не найдена для ключа '{key_original}'")
            categorical_fields_for_template[key_original] = []  # Пустой список, чтобы шаблон не упал

    return render_template('index.html',
                           display_names_numerical=DISPLAY_NAMES_NUMERICAL,
                           display_names_categorical=DISPLAY_NAMES_CATEGORICAL,
                           job_options=JOB_OPTIONS_RU,
                           categorical_fields_data=categorical_fields_for_template,
                           display_options_map=DISPLAY_OPTIONS_MAP,
                           prediction=prediction_result_display,
                           form_data=form_data
                           )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    # ... (API эндпоинт можно оставить без изменений или добавить русификацию ошибок)
    if not request.is_json:
        return jsonify({"error": "Запрос должен содержать JSON"}), 400  # Русская ошибка
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Входные данные должны быть JSON объектом"}), 400  # Русская ошибка
    try:
        prediction_result_raw = make_prediction(data)
        if "error" in prediction_result_raw:
            return jsonify(prediction_result_raw), 400

        # Русификация ответа API (опционально, зависит от требований к API)
        original_label = prediction_result_raw.get('predicted_risk_label')
        if original_label:
            prediction_result_raw['predicted_risk_label_ru'] = RISK_LABELS_RU.get(original_label, original_label)

        return jsonify(prediction_result_raw)

    except Exception as e:
        app.logger.error(f"API Error: {e}", exc_info=True)
        return jsonify({"error": "Внутренняя ошибка сервера при предсказании"}), 500  # Русская ошибка


@app.route('/decision_tree')
def decision_tree_page():
    tree_image_filename = 'decision_tree.png'
    tree_image_exists = os.path.exists(os.path.join(MODEL_DIR, tree_image_filename))
    return render_template('decision_tree.html', tree_image_filename=tree_image_filename,
                           tree_image_exists=tree_image_exists)


@app.route('/models/<filename>')
def serve_model_file(filename):
    return send_from_directory(MODEL_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)