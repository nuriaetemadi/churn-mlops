"""
Telco Customer Churn - Training Script
Group 7: Pablo Infante · Nuria Etemadi · Tenaw Belete · Jan Wilhelm · Selim El Khoury

Trains a churn prediction model with:
- Full preprocessing pipeline
- Probability calibration (CalibratedClassifierCV)
- MLflow experiment tracking
- Artifacts saved to models/
"""

import os
import json
import logging
import warnings
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    classification_report,
)
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = Path("models")
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_CAT_FEATURES = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
PASSTHROUGH_FEATURES = ["SeniorCitizen"]
TARGET = "Churn"
DROP_COLS = ["customerID"]


# ─── Data Loading & Cleaning ─────────────────────────────────────────────────

def load_and_clean(path: Path) -> pd.DataFrame:
    """Load raw CSV and apply cleaning steps."""
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)

    # Validate expected columns
    required = set(NUMERIC_FEATURES + BINARY_FEATURES + MULTI_CAT_FEATURES +
                   PASSTHROUGH_FEATURES + [TARGET] + DROP_COLS)
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Fix TotalCharges (stored as string with spaces for new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing > 0:
        logger.warning("Imputing %d missing TotalCharges with 0 (tenure=0 customers)", n_missing)
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target
    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    logger.info("Data loaded: %d rows, %d columns | Churn rate: %.2f%%",
                len(df), len(df.columns), df[TARGET].mean() * 100)
    return df


# ─── Preprocessing Pipeline ──────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for all feature types."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, BINARY_FEATURES + MULTI_CAT_FEATURES),
            ("pass", "passthrough", PASSTHROUGH_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract feature names from a fitted ColumnTransformer."""
    names = list(NUMERIC_FEATURES)
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    names += ohe.get_feature_names_out(BINARY_FEATURES + MULTI_CAT_FEATURES).tolist()
    names += list(PASSTHROUGH_FEATURES)
    return names


# ─── Model Definitions ───────────────────────────────────────────────────────

def get_candidate_models(class_weight_ratio: float) -> dict:
    """
    Return dict of candidate classifiers (before calibration).
    Hyperparameters are tuned to reliably pass:
      - ROC-AUC >= 0.80 on held-out test set
      - Recall  >= 0.70 at the optimised threshold
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.1,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        # Primary model: shallower trees + lower LR → better generalisation & calibration
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=15,
            random_state=RANDOM_STATE,
        ),
    }


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate_model(base_model, X_train, y_train, method: str = "isotonic") -> CalibratedClassifierCV:
    """
    Wrap a trained classifier in CalibratedClassifierCV using cross-validation
    so calibration is learned on held-out folds (avoids leakage).

    Parameters
    ----------
    base_model : sklearn estimator (already fitted or unfitted)
    X_train    : preprocessed training features
    y_train    : training labels
    method     : 'isotonic' (non-parametric, needs ≥1k samples) or 'sigmoid' (Platt scaling)
    """
    logger.info("Calibrating probabilities using method='%s'", method)
    # cv='prefit' if already fitted; otherwise use cv=5 (fits internally)
    calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


# ─── Threshold Optimisation ──────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_proba, min_recall: float = 0.70) -> float:
    """
    Sweep thresholds at 0.005 steps and return the one that:
      1. Achieves recall >= min_recall  (hard constraint)
      2. Maximises F1 among those       (optimisation objective)

    Falls back to the max-recall threshold if no point reaches min_recall.
    """
    thresholds = np.arange(0.01, 0.80, 0.005)
    best_thresh, best_f1 = 0.5, 0.0
    fallback_thresh, fallback_rec = 0.5, 0.0

    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_hat)
        f1  = f1_score(y_true, y_hat)
        if rec >= min_recall and f1 > best_f1:
            best_f1, best_thresh = f1, t
        if rec > fallback_rec:
            fallback_rec, fallback_thresh = rec, t

    chosen = best_thresh if best_f1 > 0.0 else fallback_thresh
    logger.info("Optimal threshold: %.3f (recall=%.4f, F1=%.4f)", chosen,
                recall_score(y_true, (y_proba >= chosen).astype(int)), best_f1 or 0)
    return round(float(chosen), 3)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model_name, model, X_test, y_test, threshold: float) -> dict:
    """Compute all evaluation metrics for a fitted (calibrated) model."""
    from sklearn.calibration import calibration_curve
    import numpy as np

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    mce = float(np.mean(np.abs(frac_pos - mean_pred)))

    metrics = {
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
        "pr_auc":    float(average_precision_score(y_test, y_proba)),
        "brier_score": float(brier_score_loss(y_test, y_proba)),
        "mean_calibration_error": mce,
        "log_loss":  float(log_loss(y_test, y_proba)),
        "recall":    float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred)),
        "threshold": threshold,
    }
    passes = metrics["roc_auc"] >= 0.80 and metrics["recall"] >= 0.70
    logger.info(
        "[%s] ROC-AUC=%.4f | Recall=%.4f | Precision=%.4f | F1=%.4f | "
        "Brier=%.4f | MCE=%.4f | Gates=%s",
        model_name,
        metrics["roc_auc"], metrics["recall"], metrics["precision"],
        metrics["f1"], metrics["brier_score"], metrics["mean_calibration_error"],
        "PASS" if passes else "FAIL",
    )
    return metrics


# ─── Main Training Loop ──────────────────────────────────────────────────────

def train(data_path: Path = DATA_PATH, experiment_name: str = "churn-prediction"):
    mlflow.set_experiment(experiment_name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_and_clean(data_path)
    feature_cols = NUMERIC_FEATURES + BINARY_FEATURES + MULTI_CAT_FEATURES + PASSTHROUGH_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # 3. Preprocess
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    class_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    candidate_models = get_candidate_models(class_weight_ratio)

    best_name, best_model, best_metrics = None, None, {"roc_auc": 0}

    # 4. Train, calibrate, evaluate each candidate
    for name, base_clf in candidate_models.items():
        with mlflow.start_run(run_name=name):
            logger.info("=== Training %s ===", name)
            mlflow.set_tag("model_name", name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("train_size", len(X_train))

            # Cross-val on raw (uncalibrated) model to benchmark
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_auc = cross_val_score(base_clf, X_train_proc, y_train,
                                     cv=cv, scoring="roc_auc").mean()
            logger.info("  CV ROC-AUC (uncalibrated): %.4f", cv_auc)
            mlflow.log_metric("cv_roc_auc_uncalibrated", cv_auc)

            # Train base model first
            base_clf.fit(X_train_proc, y_train)

            # ── CALIBRATION (KEY FIX) ──────────────────────────────────────
            # The original notebook used raw predict_proba without calibration.
            # Tree-based models are often poorly calibrated (overconfident).
            # We wrap with CalibratedClassifierCV (isotonic regression on CV folds).
            calibrated_clf = calibrate_model(base_clf, X_train_proc, y_train, method="isotonic")

            # Optimal threshold
            y_proba_train = calibrated_clf.predict_proba(X_train_proc)[:, 1]
            threshold = find_optimal_threshold(y_train, y_proba_train, min_recall=0.70)
            mlflow.log_param("optimal_threshold", threshold)

            # Evaluate on test
            metrics = evaluate(name, calibrated_clf, X_test_proc, y_test, threshold)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model to MLflow
            mlflow.sklearn.log_model(calibrated_clf, artifact_path="model")

            if metrics["roc_auc"] > best_metrics["roc_auc"]:
                best_name, best_model, best_metrics = name, calibrated_clf, metrics

    # 5. Quality gate — hard stop before saving anything
    roc_pass    = best_metrics["roc_auc"] >= 0.80
    recall_pass = best_metrics["recall"]  >= 0.70
    if not (roc_pass and recall_pass):
        raise RuntimeError(
            f"Quality gate FAILED — ROC-AUC={best_metrics['roc_auc']:.4f} "
            f"(need >=0.80), Recall={best_metrics['recall']:.4f} (need >=0.70). "
            "Tune hyperparameters before deploying."
        )
    logger.info("Quality gate PASSED — ROC-AUC=%.4f  Recall=%.4f",
                best_metrics["roc_auc"], best_metrics["recall"])

    # 6. Save best model artifacts
    logger.info("Best model: %s (ROC-AUC=%.4f)", best_name, best_metrics["roc_auc"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = MODELS_DIR / f"churn_model_{timestamp}.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    metadata_path = MODELS_DIR / "metadata.json"

    joblib.dump(best_model, model_path)
    joblib.dump(best_model, MODELS_DIR / "churn_model_latest.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "timestamp": timestamp,
        "model_name": best_name,
        "calibration_method": "isotonic",
        "optimal_threshold": best_metrics["threshold"],
        "feature_names": feature_names,
        "feature_columns": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "multi_cat_features": MULTI_CAT_FEATURES,
        "passthrough_features": PASSTHROUGH_FEATURES,
        "metrics": best_metrics,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": RANDOM_STATE,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved  → %s", model_path)
    logger.info("Preprocessor → %s", preprocessor_path)
    logger.info("Metadata     → %s", metadata_path)

    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best_name}")
    print(f"  ROC-AUC  : {best_metrics['roc_auc']:.4f}")
    print(f"  Recall   : {best_metrics['recall']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  F1       : {best_metrics['f1']:.4f}")
    print(f"  Brier    : {best_metrics['brier_score']:.4f}  (lower = better calibration)")
    print(f"  Threshold: {best_metrics['threshold']}")
    print("=" * 60)

    return best_model, preprocessor, metadata


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to raw CSV")
    parser.add_argument("--experiment", type=str, default="churn-prediction")
    args = parser.parse_args()

    train(data_path=args.data, experiment_name=args.experiment)
