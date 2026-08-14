# config.py

from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"

# Configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
TARGET = "Churn"
DROP_COLS = ["customerID"]
# MODEL_NAMES = ["Logistic Regression", "KNN", "Decision Tree"]

