"""Predict churn for one customer. Run:  python src/predict.py"""
import joblib
import pandas as pd
from config import MODEL_PATH


def predict(customer: dict):
    bundle = joblib.load(MODEL_PATH)
    row = pd.DataFrame([customer])[bundle["columns"]]
    p = bundle["model"].predict_proba(row)[0, 1]
    return p, ("WILL LIKELY CHURN" if p >= 0.5 else "likely to stay")


if __name__ == "__main__":
    sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    }
    prob, label = predict(sample)
    print(f"Churn probability: {prob:.1%} -> {label}")
