"""Training entry point.

Usage:  python -m src.train
Steps : load data -> tune C on a validation split -> refit on full train
        -> evaluate on the test set -> save model + reports.
"""
import json
import time

import joblib
from sklearn.model_selection import train_test_split

from .config import (
    C_GRID,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    VALIDATION_SIZE,
)
from .data_loader import load_test, load_train
from .evaluate import evaluate
from .model import build_pipeline


def main() -> None:
    X, y = load_train()
    X_test, y_test = load_test()
    print(f"Train: {len(X):,} samples | Test: {len(X_test):,} samples")

    # 1) Hyper-parameter tuning on a held-out validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    results = {}
    for C in C_GRID:
        t0 = time.time()
        pipe = build_pipeline(C=C).fit(X_tr, y_tr)
        acc = pipe.score(X_val, y_val)
        results[C] = acc
        print(f"C={C:<5} val_accuracy={acc:.4f}  ({time.time() - t0:.1f}s)")
    best_C = max(results, key=results.get)
    print(f"\nBest C: {best_C}")

    # 2) Refit on the full training set with the best C
    model = build_pipeline(C=best_C).fit(X, y)

    # 3) Final evaluation on the provided test set
    metrics = evaluate(model, X_test, y_test, split_name="test")

    # 4) Persist artefacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tuning.json").write_text(
        json.dumps({"validation_accuracy_by_C": results, "best_C": best_C}, indent=2)
    )
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
