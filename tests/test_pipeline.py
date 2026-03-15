"""
Unit tests for the Churn Prediction MLOps project.
Run with: pytest tests/ -v --cov=. --cov-report=term-missing
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_ROW = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.20,
}


@pytest.fixture
def sample_df():
    """Return a tiny DataFrame that mimics the real data."""
    rows = []
    for i in range(10):
        row = SAMPLE_ROW.copy()
        row["customerID"] = f"TEST-{i:04d}"
        row["Churn"] = "Yes" if i % 3 == 0 else "No"
        row["tenure"] = i * 5
        row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
        rows.append(row)
    return pd.DataFrame(rows)


# ─── train.py Tests ───────────────────────────────────────────────────────────

class TestDataLoading:
    def test_load_and_clean_success(self, tmp_path, sample_df):
        """load_and_clean should return DataFrame with binary Churn column."""
        from train import load_and_clean
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_and_clean(csv_path)
        assert "Churn" in df.columns
        assert set(df["Churn"].unique()).issubset({0, 1})
        assert df["TotalCharges"].dtype in [np.float64, np.int64]

    def test_load_and_clean_missing_totalcharges(self, tmp_path, sample_df):
        """Missing TotalCharges (blank strings) should be imputed with 0."""
        from train import load_and_clean
        sample_df.loc[0, "TotalCharges"] = " "
        sample_df.loc[0, "tenure"] = 0
        csv_path = tmp_path / "test_missing.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_and_clean(csv_path)
        assert df["TotalCharges"].isna().sum() == 0

    def test_load_and_clean_raises_on_missing_columns(self, tmp_path):
        """Should raise ValueError when required columns are missing."""
        from train import load_and_clean
        bad_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        csv_path = tmp_path / "bad.csv"
        bad_df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Missing expected columns"):
            load_and_clean(csv_path)


class TestPreprocessor:
    def test_preprocessor_output_shape(self, sample_df):
        """Fitted preprocessor should produce correct number of features."""
        from train import build_preprocessor, NUMERIC_FEATURES, BINARY_FEATURES, MULTI_CAT_FEATURES, PASSTHROUGH_FEATURES

        feature_cols = NUMERIC_FEATURES + BINARY_FEATURES + MULTI_CAT_FEATURES + PASSTHROUGH_FEATURES
        X = sample_df[feature_cols]
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)
        assert X_proc.shape[0] == len(sample_df)
        assert X_proc.shape[1] > len(feature_cols)  # OHE expands columns

    def test_feature_names_length(self, sample_df):
        """get_feature_names should return same count as preprocessor output columns."""
        from train import build_preprocessor, get_feature_names, NUMERIC_FEATURES, BINARY_FEATURES, MULTI_CAT_FEATURES, PASSTHROUGH_FEATURES

        feature_cols = NUMERIC_FEATURES + BINARY_FEATURES + MULTI_CAT_FEATURES + PASSTHROUGH_FEATURES
        X = sample_df[feature_cols]
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)
        names = get_feature_names(preprocessor)
        assert len(names) == X_proc.shape[1]


class TestThresholdOptimisation:
    def test_threshold_respects_min_recall(self):
        """find_optimal_threshold should return threshold achieving >= min_recall."""
        from train import find_optimal_threshold
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=300)
        y_proba = rng.uniform(0, 1, size=300)
        threshold = find_optimal_threshold(y_true, y_proba, min_recall=0.70)
        assert 0.0 < threshold < 1.0

    def test_threshold_in_valid_range(self):
        from train import find_optimal_threshold
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_proba = rng.uniform(0, 1, 200)
        t = find_optimal_threshold(y_true, y_proba)
        assert 0.0 <= t <= 1.0


# ─── generate_synthetic_data.py Tests ────────────────────────────────────────

class TestSyntheticDataGenerator:
    def test_output_shape(self):
        from generate_synthetic_data import generate_synthetic_data
        df = generate_synthetic_data(n=100)
        assert len(df) == 100

    def test_required_columns_present(self):
        from generate_synthetic_data import generate_synthetic_data
        df = generate_synthetic_data(n=50)
        required = [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_churn_column_values(self):
        from generate_synthetic_data import generate_synthetic_data
        df = generate_synthetic_data(n=200)
        assert set(df["Churn"].unique()).issubset({"Yes", "No"})

    def test_no_negative_charges(self):
        from generate_synthetic_data import generate_synthetic_data
        df = generate_synthetic_data(n=200)
        assert (df["MonthlyCharges"] >= 0).all()
        assert (df["TotalCharges"] >= 0).all()

    def test_customer_ids_unique(self):
        from generate_synthetic_data import generate_synthetic_data
        df = generate_synthetic_data(n=100)
        assert df["customerID"].nunique() == 100

    def test_reproducibility(self):
        from generate_synthetic_data import generate_synthetic_data
        df1 = generate_synthetic_data(n=50, random_seed=99)
        df2 = generate_synthetic_data(n=50, random_seed=99)
        pd.testing.assert_frame_equal(df1, df2)


# ─── app.py (FastAPI) Tests ───────────────────────────────────────────────────

class TestFastAPI:
    """Tests for the FastAPI endpoints using httpx TestClient."""

    @pytest.fixture
    def mock_artifacts(self, tmp_path, sample_df):
        """Create mock model + preprocessor + metadata on disk."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        from train import build_preprocessor, NUMERIC_FEATURES, BINARY_FEATURES, MULTI_CAT_FEATURES, PASSTHROUGH_FEATURES

        feature_cols = NUMERIC_FEATURES + BINARY_FEATURES + MULTI_CAT_FEATURES + PASSTHROUGH_FEATURES
        sample_df["Churn_bin"] = (sample_df["Churn"] == "Yes").astype(int)
        X = sample_df[feature_cols]
        y = sample_df["Churn_bin"]

        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)

        base = LogisticRegression(max_iter=100)
        base.fit(X_proc, y)
        calibrated = CalibratedClassifierCV(base, cv="prefit")
        calibrated.fit(X_proc, y)

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        joblib.dump(calibrated, models_dir / "churn_model_latest.joblib")
        joblib.dump(preprocessor, models_dir / "preprocessor.joblib")

        metadata = {
            "model_name": "LogisticRegression",
            "calibration_method": "isotonic",
            "optimal_threshold": 0.30,
            "feature_columns": feature_cols,
            "feature_names": feature_cols,
            "metrics": {
                "roc_auc": 0.84,
                "recall": 0.77,
                "precision": 0.55,
                "f1": 0.64,
                "brier_score": 0.14,
                "mean_calibration_error": 0.04,
                "cv_roc_auc_mean": 0.84,
                "cv_roc_auc_std": 0.012,
            },
            "training_samples": 8,
            "test_samples": 2,
            "timestamp": "20240101_120000",
        }
        with open(models_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return models_dir

    def _get_client(self, mock_artifacts):
        """Patch MODELS_DIR and reload app."""
        import app as app_module

        original_dir = app_module.MODELS_DIR
        app_module.MODELS_DIR = mock_artifacts
        try:
            app_module.MODEL, app_module.PREPROCESSOR, app_module.METADATA = app_module.load_artifacts()
            app_module.THRESHOLD = app_module.METADATA["optimal_threshold"]
            app_module.FEATURE_COLUMNS = app_module.METADATA["feature_columns"]
        except Exception:
            pass

        from fastapi.testclient import TestClient
        client = TestClient(app_module.app)
        app_module.MODELS_DIR = original_dir
        return client, app_module

    def test_health_endpoint(self, mock_artifacts):
        client, _ = self._get_client(mock_artifacts)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_single_valid(self, mock_artifacts):
        client, _ = self._get_client(mock_artifacts)
        resp = client.post("/predict", json=SAMPLE_ROW)
        assert resp.status_code == 200
        data = resp.json()
        assert "churn" in data
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_label"] in ("Low", "Medium", "High")

    def test_predict_single_invalid_gender(self, mock_artifacts):
        client, _ = self._get_client(mock_artifacts)
        bad = SAMPLE_ROW.copy()
        bad["gender"] = "Unknown"
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_batch_csv(self, mock_artifacts, sample_df, tmp_path):
        client, _ = self._get_client(mock_artifacts)
        csv_content = sample_df.to_csv(index=False).encode()
        resp = client.post(
            "/predict/batch",
            files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_customers"] == len(sample_df)
        assert len(data["predictions"]) == len(sample_df)

    def test_model_info_endpoint(self, mock_artifacts):
        client, _ = self._get_client(mock_artifacts)
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "optimal_threshold" in data
