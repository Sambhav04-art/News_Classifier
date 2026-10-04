"""Central configuration: paths, label mapping and hyper-parameters."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "news_classifier.joblib"

# Class IDs in the dataset are 1-4
LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

RANDOM_STATE = 42
VALIDATION_SIZE = 0.1

# TF-IDF settings
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=300_000,
    sublinear_tf=True,
    stop_words="english",
)

# Candidate values of C for Logistic Regression (tuned on a validation split)
C_GRID = [1, 5, 10, 20]
