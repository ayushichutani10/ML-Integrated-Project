from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.txt"

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
