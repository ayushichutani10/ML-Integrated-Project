"""Creates a synthetic dataset with the SAME columns as the Kaggle
'Telco Customer Churn' dataset. Used only when the real CSV is missing,
so the project always runs end to end.

For your submission, download the real file from Kaggle and place it at
data/Telco-Customer-Churn.csv (see README)."""
import numpy as np
import pandas as pd
from config import DATA_PATH


def generate(n=7043, seed=42):
    rng = np.random.default_rng(seed)
    yn = lambda p: np.where(rng.random(n) < p, "Yes", "No")

    tenure = np.clip(rng.exponential(30, n).astype(int) + 1, 1, 72)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    phone = yn(0.90)
    base = np.where(internet == "Fiber optic", 75, np.where(internet == "DSL", 50, 20))
    monthly = np.round(base + rng.normal(0, 8, n) + np.where(phone == "Yes", 5, 0), 2).clip(18.25, 118.75)
    total = np.round(monthly * tenure + rng.normal(0, 30, n), 2).clip(18.8)

    def addon():
        return np.where(internet == "No", "No internet service", yn(0.40))

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-SYN" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yn(0.48),
        "Dependents": yn(0.30),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", yn(0.42)),
        "InternetService": internet,
        "OnlineSecurity": addon(),
        "OnlineBackup": addon(),
        "DeviceProtection": addon(),
        "TechSupport": addon(),
        "StreamingTV": addon(),
        "StreamingMovies": addon(),
        "Contract": contract,
        "PaperlessBilling": yn(0.59),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.34, 0.23, 0.22, 0.21]),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    })

    z = (-1.0
         + 1.3 * (contract == "Month-to-month")
         - 1.2 * (contract == "Two year")
         + 0.7 * (internet == "Fiber optic")
         - 0.03 * tenure
         + 0.012 * monthly
         + 0.4 * (df["PaymentMethod"] == "Electronic check")
         + 0.3 * df["SeniorCitizen"]
         - 0.5 * (df["OnlineSecurity"] == "Yes")
         - 0.4 * (df["TechSupport"] == "Yes"))
    p = 1 / (1 + np.exp(-z))
    df["Churn"] = np.where(rng.random(n) < p, "Yes", "No")

    # mimic the real dataset's quirk: blank TotalCharges for a few customers
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    blanks = rng.choice(n, 11, replace=False)
    df.loc[blanks, "TotalCharges"] = " "
    return df


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset written to {DATA_PATH}")
