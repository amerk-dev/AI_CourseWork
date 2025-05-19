import time

import joblib
import os
from train import train_and_save_model
from config import MODEL_PATH



def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
        time.sleep(10)
        return joblib.load(MODEL_PATH)
    return joblib.load(MODEL_PATH)
