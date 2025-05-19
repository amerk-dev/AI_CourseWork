from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import load_model
from predict import make_prediction

app = FastAPI()

origins = [
    "*",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic модель
class CreditRiskInput(BaseModel):
    age: int
    sex: str
    job: int
    housing: str
    saving_accounts: str
    checking_account: str
    duration: int
    purpose: str
    credit_amount: int


# Загрузка артефактов
artifacts = load_model()

print(artifacts)


@app.post("/predict")
def predict_credit_risk(input_data: CreditRiskInput):
    result = make_prediction(input_data.dict(), artifacts)
    return result
