"""
Final training script for the Customer Churn & Segmentation project.

Run this script from the project root:

    python scripts/train_final.py

Before running:
- Make sure data/raw/Customer-Churn.csv exists.
- Set FINAL_MODEL_NAME below to the model selected after
  reviewing 04_modeling.ipynb and 05_model_evaluation.ipynb.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SegmentationScaler


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Customer-Churn.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------------------------------------------------
# Final model configuration
# -------------------------------------------------------------------

# IMPORTANT:
# Replace this with the model you selected after evaluating
# 04_modeling.ipynb and 05_model_evaluation.ipynb.
#
# Logistic Regression is kept as the safe default because it
# provides probability estimates and is easy to interpret.
FINAL_MODEL_NAME = "logistic_regression"

# Keep the threshold configurable.
FINAL_CHURN_THRESHOLD = 0.50

RANDOM_STATE = 42


# -------------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------------

def load_and_prepare_data():
    """Load the raw dataset and create the final model features."""

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    # Convert TotalCharges from text to numeric.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    missing_total_charges = (
        df["TotalCharges"].isna()
    )

    df.loc[
        missing_total_charges,
        "TotalCharges"
    ] = (
        df.loc[
            missing_total_charges,
            "tenure"
        ]
        * df.loc[
            missing_total_charges,
            "MonthlyCharges"
        ]
    )

    # Feature engineering.
    df["AverageMonthlySpend"] = (
        df["TotalCharges"]
        / df["tenure"].clip(lower=1)
    )

    df["IsNewCustomer"] = (
        df["tenure"] <= 12
    ).astype(int)

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["ServiceCount"] = 0

    for column in service_columns:
        df["ServiceCount"] += (
            df[column] == "Yes"
        ).astype(int)

    # Target.
    y = (
        df["Churn"]
        .map({
            "No": 0,
            "Yes": 1
        })
    )

    # customerID is retained separately for traceability,
    # but never enters the predictive model.
    X = df.drop(
        columns=[
            "customerID",
            "Churn"
        ]
    )

    return df, X, y


# -------------------------------------------------------------------
# Churn model
# -------------------------------------------------------------------

def build_churn_pipeline(X):
    """Build the complete preprocessing + model pipeline."""

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # This is the default production model.
    # Change this section only after selecting the final model
    # based on the actual evaluation results.
    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


def train_churn_model(
    X,
    y,
):
    """Train and save the complete churn pipeline."""

    pipeline = build_churn_pipeline(
        X
    )

    pipeline.fit(
        X,
        y
    )

    output_path = (
        MODEL_DIR
        / "churn_pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        output_path
    )

    print(
        f"Churn pipeline saved to: {output_path}"
    )

    return pipeline


# -------------------------------------------------------------------
# Customer segmentation
# -------------------------------------------------------------------

def train_segmentation_model(
    df,
):
    """
    Train the final segmentation scaler and KMeans model.

    Churn is intentionally not used for clustering.
    """

    segmentation_features = [
        "tenure",
        "MonthlyCharges",
        "ServiceCount",
    ]

    X_segmentation = df[
        segmentation_features
    ].copy()

    scaler = SegmentationScaler()

    X_scaled = scaler.fit_transform(
        X_segmentation
    )

    # The selected cluster count should match the final
    # decision from 06_customer_segmentation.ipynb.
    number_of_clusters = 3

    kmeans_model = KMeans(
        n_clusters=number_of_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    kmeans_model.fit(
        X_scaled
    )

    scaler_path = (
        MODEL_DIR
        / "segmentation_scaler.joblib"
    )

    model_path = (
        MODEL_DIR
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

    print(
        f"Segmentation scaler saved to: {scaler_path}"
    )

    print(
        f"Segmentation model saved to: {model_path}"
    )

    return scaler, kmeans_model


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    print("Loading and preparing data...")

    df, X, y = load_and_prepare_data()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Features used by churn model: {X.shape[1]}"
    )

    print(
        f"Configured churn threshold: "
        f"{FINAL_CHURN_THRESHOLD:.2f}"
    )

    print("\nTraining churn model...")

    train_churn_model(
        X,
        y
    )

    print("\nTraining customer segmentation model...")

    train_segmentation_model(
        df
    )

    print("\nFinal training completed.")


if __name__ == "__main__":
    main()
