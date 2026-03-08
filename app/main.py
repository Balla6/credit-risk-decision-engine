from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# -----------------------------
# Load model artifacts
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_credit_model.pkl"
CONFIG_PATH = BASE_DIR / "models" / "decision_config.pkl"

model = joblib.load(MODEL_PATH)
decision_config = joblib.load(CONFIG_PATH)

APPROVE_THRESHOLD = decision_config["approve_threshold"]
REVIEW_THRESHOLD = decision_config["review_threshold"]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Credit Risk Decision Engine",
    description="Predicts default probability and returns APPROVE / REVIEW / REJECT decision.",
    version="1.0.0"
)

# -----------------------------
# Input schema
# -----------------------------
class CreditApplicant(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

# -----------------------------
# Decision logic
# -----------------------------
def credit_decision(probability: float) -> str:
    if probability < APPROVE_THRESHOLD:
        return "APPROVE"
    elif probability < REVIEW_THRESHOLD:
        return "REVIEW"
    else:
        return "REJECT"

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"message": "Credit Risk Decision Engine API is running."}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(applicant: CreditApplicant):
    input_df = pd.DataFrame([applicant.model_dump()])

    default_probability = float(model.predict_proba(input_df)[:, 1][0])
    decision = credit_decision(default_probability)

    return {
        "default_probability": round(default_probability, 6),
        "decision": decision,
        "thresholds": {
            "approve_below": APPROVE_THRESHOLD,
            "review_below": REVIEW_THRESHOLD,
            "reject_at_or_above": REVIEW_THRESHOLD
        }
    }