import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

MODEL_PATH = "models/credit_risk_model-v1.joblib"
DATA_PATH = "data/german_credit_data.csv"


def preprocess_data(df):
    """Предобработка данных"""
    df = df.copy()

    # Приведение названий колонок к snake_case
    df = df.rename(columns={
        "Saving accounts": "saving_accounts",
        "Checking account": "checking_account"
    })

    # Заполнение пропущенных значений
    df['saving_accounts'] = df['saving_accounts'].fillna('unknown')
    df['checking_account'] = df['checking_account'].fillna('unknown')

    # Кодирование категориальных переменных
    label_encoders = {}
    categorical_cols = ['Sex', 'Housing', 'saving_accounts', 'checking_account', 'Purpose']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Целевая переменная
    df['Credit amount'] = df['Credit amount'].astype(float)
    y = df['Credit amount'] > df['Credit amount'].median()
    X = df.drop(columns=['Credit amount'])

    return X, y, label_encoders



def train_and_save_model():
    df = pd.read_csv(DATA_PATH, index_col=0)
    X, y, label_encoders = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    artifacts = {
        'model': model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_columns': X.columns.tolist()
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)
    print("Модель сохранена")
