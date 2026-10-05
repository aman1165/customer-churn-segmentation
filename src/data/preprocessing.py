"""
Data loading and cleaning utilities for the customer churn project.

This module contains only reusable data preparation logic.
Business analysis and visualization stay inside the notebooks.
"""

from pathlib import Path

import pandas as pd


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Load the raw customer churn CSV file."""

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    return pd.read_csv(data_path)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges to numeric and handle blank values.

    In the original dataset, some TotalCharges values are blank
    for customers with zero tenure. For this project, the missing
    value is reconstructed as:

        tenure * MonthlyCharges
    """

    cleaned_df = df.copy()

    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    missing_total_charges = cleaned_df["TotalCharges"].isna()

    cleaned_df.loc[
        missing_total_charges,
        "TotalCharges"
    ] = (
        cleaned_df.loc[
            missing_total_charges,
            "tenure"
        ]
        * cleaned_df.loc[
            missing_total_charges,
            "MonthlyCharges"
        ]
    )

    remaining_missing = cleaned_df["TotalCharges"].isna().sum()

    if remaining_missing > 0:
        raise ValueError(
            "TotalCharges still contains missing values after cleaning."
        )

    return cleaned_df
