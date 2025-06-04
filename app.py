# app.py (Заготовка для UI, например, Flask)
# from flask import Flask, request, jsonify, render_template
# from src.predict import make_prediction
# import pandas as pd

# app = Flask(__name__)

# @app.route('/')
# def home():
#    # return render_template('index.html') # Если будет HTML форма
#    return "Credit Risk Prediction API"

# @app.route('/predict', methods=['POST'])
# def predict_api():
#    data = request.get_json(force=True) # Получаем данные в JSON
#    # Ожидаем, что data - это dict или list of dicts
#    # Например: {'Age': 30, 'Sex': 'male', ...}
#    # или [{'Age': 30, ...}, {'Age': 45, ...}]

#    prediction_result = make_prediction(data)
#    return jsonify(prediction_result)

# if __name__ == '__main__':
#    # app.run(debug=True, port=5000)
#    print("Это заготовка для UI. Запустите model_training.py для обучения,")
#    print("затем src/predict.py для примера предсказания.")
#    print("Для запуска Flask API раскомментируйте код выше и установите Flask.")
pass  # Пока оставим пустым или с комментариями