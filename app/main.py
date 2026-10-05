from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
TEMPLATE_DIR = PROJECT_ROOT / "app" / "templates"

INDEX_HTML_PATH = TEMPLATE_DIR / "index.html"

CHURN_MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
SEGMENTATION_SCALER_PATH = (
    MODEL_DIR / "segmentation_scaler.joblib"
)
SEGMENTATION_MODEL_PATH = (
    MODEL_DIR / "customer_segmentation_kmeans.joblib"
)


# ============================================================
# Load trained artifacts
# ============================================================

if not CHURN_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Churn model not found: {CHURN_MODEL_PATH}"
    )

if not SEGMENTATION_SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Segmentation scaler not found: {SEGMENTATION_SCALER_PATH}"
    )

if not SEGMENTATION_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Segmentation model not found: {SEGMENTATION_MODEL_PATH}"
    )


churn_model = joblib.load(CHURN_MODEL_PATH)
segmentation_scaler = joblib.load(
    SEGMENTATION_SCALER_PATH
)
segmentation_model = joblib.load(
    SEGMENTATION_MODEL_PATH
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Customer Churn & Segmentation API",
    description=(
        "API for customer churn prediction and "
        "customer segmentation."
    ),
    version="1.0.0",
)


# ============================================================
# Request Schema
# ============================================================

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ============================================================
# Feature Engineering
# ============================================================

def prepare_customer_data(
    customer: CustomerInput,
) -> pd.DataFrame:

    data = customer.model_dump()

    df = pd.DataFrame([data])

    # Feature used by the churn model
    df["AverageMonthlySpend"] = (
        df["TotalCharges"]
        / df["tenure"].clip(lower=1)
    )

    # New customer indicator
    df["IsNewCustomer"] = (
        df["tenure"] <= 12
    ).astype(int)

    # Service count
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

    return df


# ============================================================
# Web Interface
# ============================================================

@app.get("/")
def home():
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Frontend not found: {INDEX_HTML_PATH}",
        )

    return FileResponse(INDEX_HTML_PATH)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "churn_model_loaded": True,
        "segmentation_model_loaded": True,
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(customer: CustomerInput):

    try:
        customer_df = prepare_customer_data(customer)

        # ----------------------------------------------------
        # Churn prediction
        # ----------------------------------------------------

        churn_probability = float(
            churn_model.predict_proba(
                customer_df
            )[0][1]
        )

        churn_prediction = int(
            churn_probability >= 0.50
        )

        # ----------------------------------------------------
        # Customer segmentation
        # ----------------------------------------------------

        segmentation_features = customer_df[
            [
                "tenure",
                "MonthlyCharges",
                "ServiceCount",
            ]
        ]

        scaled_features = (
            segmentation_scaler.transform(
                segmentation_features
            )
        )

        segment = int(
            segmentation_model.predict(
                scaled_features
            )[0]
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "churn_probability": round(
                churn_probability,
                4,
            ),
            "churn_prediction": churn_prediction,
            "segment": segment,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )