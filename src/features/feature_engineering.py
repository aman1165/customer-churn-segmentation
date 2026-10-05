"""
Feature engineering utilities for the customer churn project.

Keep feature definitions in one place so that training and
future predictions use the same logic.
"""

import pandas as pd


SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create reusable customer-level features.

    Features created:
    - AverageMonthlySpend
    - IsNewCustomer
    - ServiceCount
    """

    featured_df = df.copy()

    featured_df["AverageMonthlySpend"] = (
        featured_df["TotalCharges"]
        / featured_df["tenure"].clip(lower=1)
    )

    featured_df["IsNewCustomer"] = (
        featured_df["tenure"] <= 12
    ).astype(int)

    featured_df["ServiceCount"] = 0

    for column in SERVICE_COLUMNS:
        featured_df["ServiceCount"] += (
            featured_df[column] == "Yes"
        ).astype(int)

    return featured_df
