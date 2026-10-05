"""
Customer segmentation utilities.

This module contains reusable KMeans segmentation logic.

Important:
The scaler and KMeans model must be fitted during training.
For new customers, we only transform the customer using the
existing scaler and then assign the existing cluster.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DEFAULT_SEGMENTATION_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "ServiceCount",
]


def build_segmentation_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the features used by the segmentation model.

    These features are intentionally kept separate from the
    churn target.
    """

    missing_columns = [
        column
        for column in DEFAULT_SEGMENTATION_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing segmentation features: "
            + ", ".join(missing_columns)
        )

    return df[
        DEFAULT_SEGMENTATION_FEATURES
    ].copy()


def fit_segmentation_model(
    df: pd.DataFrame,
    number_of_clusters: int,
    random_state: int = 42,
):
    """
    Fit the scaler and KMeans model on customer data.

    Returns:
        scaler
        kmeans_model
    """

    segmentation_features = build_segmentation_features(
        df
    )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        segmentation_features
    )

    kmeans_model = KMeans(
        n_clusters=number_of_clusters,
        random_state=random_state,
        n_init=10
    )

    kmeans_model.fit(
        scaled_features
    )

    return scaler, kmeans_model


def assign_segments(
    df: pd.DataFrame,
    scaler: StandardScaler,
    kmeans_model: KMeans,
) -> pd.DataFrame:
    """
    Assign existing customer data to previously fitted segments.

    No new clustering is performed here.
    """

    result = df.copy()

    segmentation_features = build_segmentation_features(
        result
    )

    scaled_features = scaler.transform(
        segmentation_features
    )

    result["Segment"] = (
        kmeans_model.predict(
            scaled_features
        )
    )

    return result


def save_segmentation_artifacts(
    scaler,
    kmeans_model,
    output_directory: str | Path,
):
    """Save the fitted scaler and KMeans model."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    scaler_path = (
        output_directory
        / "segmentation_scaler.joblib"
    )

    model_path = (
        output_directory
        / "customer_segmentation_kmeans.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    joblib.dump(
        kmeans_model,
        model_path
    )

    return scaler_path, model_path


def load_segmentation_artifacts(
    scaler_path: str | Path,
    model_path: str | Path,
):
    """Load previously fitted segmentation artifacts."""

    scaler_path = Path(scaler_path)
    model_path = Path(model_path)

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Segmentation scaler not found: {scaler_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Segmentation model not found: {model_path}"
        )

    scaler = joblib.load(
        scaler_path
    )

    kmeans_model = joblib.load(
        model_path
    )

    return scaler, kmeans_model
