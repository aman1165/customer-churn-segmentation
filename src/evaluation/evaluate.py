"""
Model evaluation utilities for the customer churn project.

This module keeps evaluation logic reusable outside notebooks.

The notebook remains responsible for visual exploration and
business interpretation. This module focuses on repeatable
metric calculation and validation.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def load_model(model_path: str | Path):
    """Load a previously saved model pipeline."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


def calculate_metrics(
    y_true,
    predicted_classes,
    predicted_probabilities,
) -> dict:
    """
    Calculate the main classification metrics.

    ROC-AUC uses predicted probabilities, while the other
    classification metrics use the selected prediction threshold.
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predicted_classes
        ),
        "precision": precision_score(
            y_true,
            predicted_classes,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predicted_classes,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predicted_classes,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            predicted_probabilities
        ),
    }

    return metrics


def evaluate_model(
    model,
    X,
    y,
    threshold: float = 0.50,
) -> dict:
    """
    Evaluate a trained churn model at a configurable threshold.

    Returns metrics, confusion matrix and classification report.
    """

    predicted_probabilities = (
        model.predict_proba(X)[:, 1]
    )

    predicted_classes = (
        predicted_probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_true=y,
        predicted_classes=predicted_classes,
        predicted_probabilities=predicted_probabilities,
    )

    confusion = confusion_matrix(
        y,
        predicted_classes
    )

    report = classification_report(
        y,
        predicted_classes,
        zero_division=0
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "classification_report": report,
        "threshold": threshold,
    }


def evaluate_multiple_thresholds(
    model,
    X,
    y,
    thresholds=None,
) -> pd.DataFrame:
    """
    Evaluate the same model across multiple probability thresholds.

    This is useful when the business cost of false positives and
    false negatives is different.
    """

    if thresholds is None:
        thresholds = [
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ]

    predicted_probabilities = (
        model.predict_proba(X)[:, 1]
    )

    results = []

    for threshold in thresholds:

        predicted_classes = (
            predicted_probabilities >= threshold
        ).astype(int)

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(
                y,
                predicted_classes
            ),
            "precision": precision_score(
                y,
                predicted_classes,
                zero_division=0
            ),
            "recall": recall_score(
                y,
                predicted_classes,
                zero_division=0
            ),
            "f1": f1_score(
                y,
                predicted_classes,
                zero_division=0
            ),
            "customers_flagged": int(
                predicted_classes.sum()
            ),
        })

    return pd.DataFrame(results)
