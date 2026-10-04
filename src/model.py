"""Model definition: TF-IDF + Logistic Regression as a single sklearn Pipeline."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TFIDF_PARAMS
from .preprocess import clean_text


def build_pipeline(C: float = 10.0) -> Pipeline:
    """Return an untrained pipeline.

    Preprocessing lives inside the pipeline (via ``preprocessor=clean_text``)
    so training and serving always apply exactly the same transformations.
    """
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, **TFIDF_PARAMS)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
