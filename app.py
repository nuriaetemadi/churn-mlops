"""
Telco Customer Churn – Prediction API
Group 7: Pablo Infante · Nuria Etemadi · Tenaw Belete · Jan Wilhelm · Selim El Khoury

FastAPI app exposing:
  POST /predict          – single customer prediction
  POST /predict/batch    – CSV batch prediction
  GET  /health           – liveness probe
  GET  /model/info       – model metadata
"""

import io
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

def load_artifacts():
    model_path = MODELS_DIR / "churn_model_latest.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    metadata_path = MODELS_DIR / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info("Loaded model: %s | Threshold: %s",
                metadata["model_name"], metadata["optimal_threshold"])
    return model, preprocessor, metadata


try:
    MODEL, PREPROCESSOR, METADATA = load_artifacts()
    THRESHOLD = METADATA["optimal_threshold"]
    FEATURE_COLUMNS = METADATA["feature_columns"]
except Exception as e:
    logger.warning("Could not load model on startup: %s", e)
    MODEL, PREPROCESSOR, METADATA, THRESHOLD, FEATURE_COLUMNS = None, None, {}, 0.5, []

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn with calibrated probability scores.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Schemas ───────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """All features required to make a prediction for one customer."""
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=29.85)
    TotalCharges: float = Field(..., ge=0, example=29.85)

    @validator("gender")
    def validate_gender(cls, v):
        if v not in ("Male", "Female"):
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @validator("Contract")
    def validate_contract(cls, v):
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v

    @validator("InternetService")
    def validate_internet(cls, v):
        valid = {"DSL", "Fiber optic", "No"}
        if v not in valid:
            raise ValueError(f"InternetService must be one of {valid}")
        return v


class PredictionResult(BaseModel):
    churn: bool
    churn_probability: float = Field(..., description="Calibrated probability of churning [0–1]")
    no_churn_probability: float
    threshold_used: float
    risk_label: str = Field(..., description="Low / Medium / High risk")


class BatchPredictionResult(BaseModel):
    total_customers: int
    churn_count: int
    no_churn_count: int
    churn_rate: float
    predictions: list


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _check_model():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")


def _risk_label(prob: float, threshold: float = None) -> str:
    """Risk label scales with the current threshold."""
    t = threshold if threshold is not None else THRESHOLD
    if prob < t * 0.6:
        return "Low"
    elif prob < t:
        return "Medium"
    return "High"


def _predict_df(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """Run preprocessor + model on a DataFrame of customer features."""
    t = threshold if threshold is not None else THRESHOLD
    X = PREPROCESSOR.transform(df[FEATURE_COLUMNS])
    probas = MODEL.predict_proba(X)
    churn_prob = probas[:, 1]
    churn_pred = (churn_prob >= t).astype(int)
    return pd.DataFrame({
        "churn": churn_pred.astype(bool),
        "churn_probability": np.round(churn_prob, 4),
        "no_churn_probability": np.round(1 - churn_prob, 4),
        "risk_label": [_risk_label(p, t) for p in churn_prob],
    })


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_name": METADATA.get("model_name", "unknown"),
    }


@app.get("/model/info", tags=["System"])
def model_info():
    _check_model()
    return {
        "model_name": METADATA.get("model_name"),
        "calibration_method": METADATA.get("calibration_method"),
        "optimal_threshold": THRESHOLD,
        "training_samples": METADATA.get("training_samples"),
        "metrics": METADATA.get("metrics"),
        "timestamp": METADATA.get("timestamp"),
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict_single(customer: CustomerFeatures, threshold: float = None):
    """
    Predict churn for a single customer.
    Pass ?threshold=0.35 to override the default optimised threshold.
    Lower = higher recall. Higher = higher precision.
    """
    _check_model()
    t = threshold if threshold is not None else THRESHOLD
    if not 0.0 < t < 1.0:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")
    try:
        df = pd.DataFrame([customer.dict()])
        result = _predict_df(df, threshold=t).iloc[0]
        return PredictionResult(
            churn=bool(result["churn"]),
            churn_probability=float(result["churn_probability"]),
            no_churn_probability=float(result["no_churn_probability"]),
            threshold_used=t,
            risk_label=result["risk_label"],
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...), threshold: float = None):
    """
    Upload a CSV file with customer data and get churn predictions for all rows.
    The CSV may optionally include a customerID column.
    Pass ?threshold=0.35 to override the default optimised threshold.
    """
    _check_model()
    t = threshold if threshold is not None else THRESHOLD
    if not 0.0 < t < 1.0:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Validate required columns
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

    # Fix TotalCharges if needed
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    try:
        preds = _predict_df(df, threshold=t)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Merge customer IDs if present
    if "customerID" in df.columns:
        preds.insert(0, "customerID", df["customerID"].values)

    records = preds.to_dict(orient="records")
    churn_count = int(preds["churn"].sum())

    return BatchPredictionResult(
        total_customers=len(preds),
        churn_count=churn_count,
        no_churn_count=len(preds) - churn_count,
        churn_rate=round(churn_count / len(preds), 4),
        predictions=records,
    )
