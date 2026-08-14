# config.py

from pathlib import Path

# Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "Train_Test_Network_dataset"
DATA_FILE = DATA_DIR / "train_test_network.csv"
LOG_DIR = PROJECT_ROOT / "logs"

BINARY_TARGET_COLUMN = "label"
ATTACK_TARGET_COLUMN = "type"

BINARY_LEAKAGE_COLUMNS = ["type"]
ATTACK_LEAKAGE_COLUMNS = ["label"]

IDENTIFIER_COLUMNS = ["src_ip", "dst_ip","dns_query"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25

CV_FOLDS = 5