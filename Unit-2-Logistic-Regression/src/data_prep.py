import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import DATA_PATH, TARGET


def load_data():
    if not DATA_PATH.exists():
        print("Real dataset not found -> generating synthetic stand-in.")
        print("(Put the Kaggle CSV at data/Telco-Customer-Churn.csv for real results.)\n")
        from generate_synthetic_data import generate
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def clean(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"], errors="ignore")
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df


def build_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat),
    ])


def make_pipeline(X, classifier):
    return Pipeline([("prep", build_preprocessor(X)), ("clf", classifier)])
