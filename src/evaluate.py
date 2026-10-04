"""Evaluation helpers: metrics, confusion matrix plot, JSON report."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import LABEL_MAP, REPORT_DIR


def evaluate(model, X, y, split_name: str = "test", save: bool = True) -> dict:
    """Evaluate ``model`` on (X, y); optionally save metrics + confusion matrix."""
    preds = model.predict(X)
    labels = sorted(LABEL_MAP)
    names = [LABEL_MAP[i] for i in labels]

    metrics = {
        "split": split_name,
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "per_class": classification_report(
            y, preds, labels=labels, target_names=names, output_dict=True
        ),
    }
    print(f"\n=== {split_name.upper()} RESULTS ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}\n")
    print(classification_report(y, preds, labels=labels, target_names=names, digits=4))

    if save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        _plot_confusion_matrix(y, preds, labels, names, split_name)
    return metrics


def _plot_confusion_matrix(y, preds, labels, names, split_name):
    cm = confusion_matrix(y, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({split_name})")
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"{split_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)
