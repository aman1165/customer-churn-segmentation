"""
Final training script for the Customer Churn & Segmentation project.

Run this script from the project root:

    python scripts/train_final.py

Before running:
- Make sure data/raw/Customer-Churn.csv exists.
- Logistic Regression is used as the final churn model.
- Customer segmentation uses 2 clusters.
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

# Final churn model selected from the modeling/evaluation workflow.
FINAL_MODEL_NAME = "logistic_regression"

# Production churn threshold.
FINAL_CHURN_THRESHOLD = 0.50

# Reproducibility.
RANDOM_STATE = 42


# -------------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------------

def load_and_prepare_data():
    """
    Load the raw dataset and create the final model features.
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    # ---------------------------------------------------------------
    # Convert TotalCharges from text to numeric
    # ---------------------------------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    missing_total_charges = (
        df["TotalCharges"].isna()
    )

    # Reconstruct missing TotalCharges.
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

    # Final validation.
    remaining_missing = (
        df["TotalCharges"].isna().sum()
    )

    if remaining_missing > 0:
        raise ValueError(
            "TotalCharges still contains missing values "
            "after cleaning."
        )

    # ---------------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------

    y = (
        df["Churn"]
        .map({
            "No": 0,
            "Yes": 1
        })
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected values found in Churn target."
        )

    # ---------------------------------------------------------------
    # Model features
    # ---------------------------------------------------------------

    # customerID is retained in df for traceability,
    # but is never used by the predictive model.

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
    """
    Build the complete preprocessing + Logistic Regression pipeline.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # ---------------------------------------------------------------
    # Numerical preprocessing
    # ---------------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # ---------------------------------------------------------------
    # Categorical preprocessing
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Combined preprocessing
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Final model
    # ---------------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    # ---------------------------------------------------------------
    # Complete pipeline
    # ---------------------------------------------------------------

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
    """
    Train and save the complete churn pipeline.
    """

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

    # ---------------------------------------------------------------
    # Segmentation features
    # ---------------------------------------------------------------

    segmentation_features = [
        "tenure",
        "MonthlyCharges",
        "ServiceCount",
    ]

    X_segmentation = df[
        segmentation_features
    ].copy()

    # ---------------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------------

    scaler = SegmentationScaler()

    X_scaled = scaler.fit_transform(
        X_segmentation
    )

    # ---------------------------------------------------------------
    # Final cluster count
    # ---------------------------------------------------------------

    # Final segmentation model uses 2 clusters.
    number_of_clusters = 2

    kmeans_model = KMeans(
        n_clusters=number_of_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    kmeans_model.fit(
        X_scaled
    )

    # ---------------------------------------------------------------
    # Save scaler
    # ---------------------------------------------------------------

    scaler_path = (
        MODEL_DIR
        / "segmentation_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        f"Segmentation scaler saved to: {scaler_path}"
    )

    # ---------------------------------------------------------------
    # Save KMeans model
    # ---------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "customer_segmentation_kmeans.joblib"
    )

    joblib.dump(
        kmeans_model,
        model_path
    )

    print(
        f"Segmentation model saved to: {model_path}"
    )

    # ---------------------------------------------------------------
    # Add segment labels to dataset
    # ---------------------------------------------------------------

    df_with_segments = df.copy()

    df_with_segments["Segment"] = (
        kmeans_model.predict(
            X_scaled
        )
    )

    # ---------------------------------------------------------------
    # Save processed customer segments
    # ---------------------------------------------------------------

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    segments_path = (
        processed_dir
        / "customer_segments.csv"
    )

    df_with_segments.to_csv(
        segments_path,
        index=False
    )

    print(
        f"Customer segments saved to: {segments_path}"
    )

    return (
        scaler,
        kmeans_model,
        df_with_segments
    )


# -------------------------------------------------------------------
# Model validation
# -------------------------------------------------------------------

def validate_saved_models():
    """
    Verify that all final model artifacts were created successfully.
    """

    required_files = [
        MODEL_DIR / "churn_pipeline.joblib",
        MODEL_DIR / "segmentation_scaler.joblib",
        MODEL_DIR / "customer_segmentation_kmeans.joblib",
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Expected model artifact not found: "
                f"{file_path}"
            )

    print("\nAll model artifacts verified.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CUSTOMER CHURN & SEGMENTATION")
    print("FINAL TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Check dataset
    # ---------------------------------------------------------------

    if not RAW_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    print(
        f"\nDataset: {RAW_DATA_PATH}"
    )

    # ---------------------------------------------------------------
    # Load and prepare data
    # ---------------------------------------------------------------

    print(
        "\nLoading and preparing data..."
    )

    df, X, y = load_and_prepare_data()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Original columns: {df.shape[1]}"
    )

    print(
        f"Features used by churn model: {X.shape[1]}"
    )

    print(
        f"Churn target values: "
        f"{y.value_counts().to_dict()}"
    )

    print(
        f"Configured churn threshold: "
        f"{FINAL_CHURN_THRESHOLD:.2f}"
    )

    # ---------------------------------------------------------------
    # Train churn model
    # ---------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "Training final churn model..."
    )

    churn_model = train_churn_model(
        X,
        y
    )

    print(
        f"Final model: {FINAL_MODEL_NAME}"
    )

    # ---------------------------------------------------------------
    # Train segmentation model
    # ---------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "Training final customer segmentation model..."
    )

    scaler, kmeans_model, segmented_df = (
        train_segmentation_model(
            df
        )
    )

    print(
        f"Number of clusters: "
        f"{kmeans_model.n_clusters}"
    )

    # ---------------------------------------------------------------
    # Segment distribution
    # ---------------------------------------------------------------

    print(
        "\nCustomer segment distribution:"
    )

    print(
        segmented_df["Segment"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------------
    # Verify artifacts
    # ---------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "Validating saved artifacts..."
    )

    validate_saved_models()

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved artifacts:"
    )

    print(
        f"  Churn model:"
        f"\n    {MODEL_DIR / 'churn_pipeline.joblib'}"
    )

    print(
        f"\n  Segmentation scaler:"
        f"\n    {MODEL_DIR / 'segmentation_scaler.joblib'}"
    )

    print(
        f"\n  Segmentation model:"
        f"\n    {MODEL_DIR / 'customer_segmentation_kmeans.joblib'}"
    )

    print(
        f"\n  Customer segments:"
        f"\n    {PROJECT_ROOT / 'data' / 'processed' / 'customer_segments.csv'}"
    )

    print(
        f"\n  Churn model:"
        f"\n    {FINAL_MODEL_NAME}"
    )

    print(
        f"\n  Churn threshold:"
        f"\n    {FINAL_CHURN_THRESHOLD:.2f}"
    )

    print(
        f"\n  Number of clusters:"
        f"\n    {kmeans_model.n_clusters}"
    )

    print(
        "\nFinal training completed."
    )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()