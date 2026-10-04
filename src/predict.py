"""Inference wrapper used by the API (and usable from the command line)."""
from functools import lru_cache

import joblib
import numpy as np

from .config import LABEL_MAP, MODEL_PATH
from .preprocess import combine_title_description


class NewsClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)
        # classes_ are the class ids (1-4) in sorted order
        self.class_ids = [int(c) for c in self.model.classes_]

    def predict(self, title: str, description: str = "") -> dict:
        return self.predict_batch([(title, description)])[0]

    def predict_batch(self, items: list[tuple[str, str]]) -> list[dict]:
        texts = [combine_title_description(t, d) for t, d in items]
        probs = self.model.predict_proba(texts)
        results = []
        for row in probs:
            idx = int(np.argmax(row))
            class_id = self.class_ids[idx]
            results.append(
                {
                    "class_id": class_id,
                    "category": LABEL_MAP[class_id],
                    "confidence": round(float(row[idx]), 4),
                    "probabilities": {
                        LABEL_MAP[c]: round(float(p), 4)
                        for c, p in zip(self.class_ids, row)
                    },
                }
            )
        return results


@lru_cache(maxsize=1)
def get_classifier() -> NewsClassifier:
    return NewsClassifier()


if __name__ == "__main__":
    import sys

    title = sys.argv[1] if len(sys.argv) > 1 else "Stocks rally as Fed signals rate pause"
    desc = sys.argv[2] if len(sys.argv) > 2 else ""
    print(get_classifier().predict(title, desc))
