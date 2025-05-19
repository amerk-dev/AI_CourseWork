import pandas as pd


def make_prediction(data: dict, artifacts: dict):
    model = artifacts['model']
    scaler = artifacts['scaler']
    label_encoders = artifacts['label_encoders']
    feature_columns = artifacts['feature_columns']

    # Кодирование категориальных признаков
    encoded_data = {}
    for col in ['sex', 'housing', 'saving_accounts', 'checking_account', 'purpose']:
        le = label_encoders[col.capitalize() if col in ['sex', 'housing', 'purpose'] else col]
        value = data[col]

        if value not in le.classes_:
            raise ValueError(f"Недопустимое значение для {col}. Допустимые: {list(le.classes_)}")

        encoded_data[col] = le.transform([value])[0]

    # Формирование финального входного DataFrame
    input_data = {
        'age': data['age'],
        'sex': encoded_data['sex'],
        'job': data['job'],
        'housing': encoded_data['housing'],
        'saving_accounts': encoded_data['saving_accounts'],
        'checking_account': encoded_data['checking_account'],
        'duration': data['duration'],
        'purpose': encoded_data['purpose']
    }

    input_df = pd.DataFrame([input_data], columns=feature_columns)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0]

    return {
        'prediction': int(prediction[0]),
        'probability': {
            'low_risk': round(probability[0], 3),
            'high_risk': round(probability[1], 3)
        }
    }

