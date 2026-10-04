"""Dataset loading and validation."""
import pandas as pd

from .config import LABEL_MAP, TEST_PATH, TRAIN_PATH
from .preprocess import combine_title_description


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """Create the raw model input from a dataframe with Title/Description."""
    return pd.Series(
        [
            combine_title_description(t, d)
            for t, d in zip(df["Title"].fillna(""), df["Description"].fillna(""))
        ],
        index=df.index,
    )


def load_dataset(path) -> tuple[pd.Series, pd.Series]:
    """Load a CSV and return (texts, labels)."""
    df = pd.read_csv(path)
    expected = {"Class Index", "Title", "Description"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if not set(df["Class Index"].unique()) <= set(LABEL_MAP):
        raise ValueError("Unexpected class ids found in 'Class Index'.")
    return build_text_column(df), df["Class Index"]


def load_train():
    return load_dataset(TRAIN_PATH)


def load_test():
    return load_dataset(TEST_PATH)
