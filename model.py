import time

import joblib
import os
from train import train_and_save_model

MODEL_PATH = "models/credit_risk_model-v1.joblib"


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
        return joblib.load(MODEL_PATH)
    return joblib.load(MODEL_PATH)
