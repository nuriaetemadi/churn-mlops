"""
Telco Customer Churn – Streamlit Web Application
Group 7: Pablo Infante · Nuria Etemadi · Tenaw Belete · Jan Wilhelm · Selim El Khoury

Features:
  1. Manual form – fill in one customer's details and get instant churn prediction
  2. CSV upload  – batch predictions with filtering and download

Threshold note:
  The default threshold (0.295) is data-driven: it is the value that maximises F1
  subject to Recall >= 70% on the held-out test set. It is lower than the naive 0.50
  because missing a churner (false negative) is far more costly than a false alarm
  (false positive) in a telco retention context. Users can adjust it via the sidebar
  slider to match their business cost structure.
"""

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Model Artifacts ────────────────────────────────────────────────────
MODELS_DIR = Path("models")

@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "churn_model_latest.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return model, preprocessor, metadata

try:
    MODEL, PREPROCESSOR, METADATA = load_model()
    DEFAULT_THRESHOLD = float(METADATA["optimal_threshold"])
    FEATURE_COLUMNS = METADATA["feature_columns"]
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    DEFAULT_THRESHOLD = 0.295
    FEATURE_COLUMNS = []
    st.error(f"Could not load model: {e}. Run python train.py first.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/telecom.png", width=64)
    st.title("Churn Predictor")
    st.caption("Group 7 · MLOps Project")
    st.divider()

    if MODEL_LOADED:
        st.success("Model loaded")
        st.caption(f"**Algorithm:** {METADATA.get('model_name', '-')}")
        st.caption(f"**Calibration:** {METADATA.get('calibration_method', '-')}")
        m = METADATA.get("metrics", {})
        st.caption(f"**ROC-AUC:** {m.get('roc_auc', '-'):.4f}" if m else "")
        st.caption(f"**Brier Score:** {m.get('brier_score', '-'):.4f}" if m else "")

        st.divider()
        st.subheader("Threshold Tuning")
        THRESHOLD = st.slider(
            "Classification Threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_THRESHOLD,
            step=0.005,
            help=(
                "Lower = catch more churners (higher recall, more false positives). "
                "Higher = more conservative (fewer false positives, risk missing churners). "
                f"Default {DEFAULT_THRESHOLD} maximises F1 at Recall >= 70%."
            )
        )
        st.caption(f"**Current:** {THRESHOLD:.3f}  |  **Default:** {DEFAULT_THRESHOLD}")
        if THRESHOLD < DEFAULT_THRESHOLD - 0.005:
            st.info("Lower than default - catching more churners.")
        elif THRESHOLD > DEFAULT_THRESHOLD + 0.005:
            st.warning("Higher than default - may miss some churners.")
    else:
        THRESHOLD = DEFAULT_THRESHOLD

    st.divider()
    tab_choice = st.radio("Mode", ["Single Customer", "CSV Batch Prediction"])

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_risk_label(prob: float, threshold: float) -> str:
    if prob < threshold * 0.6:
        return "Low"
    elif prob < threshold:
        return "Medium"
    return "High"


def get_risk_color(prob: float, threshold: float) -> str:
    if prob < threshold * 0.6:
        return "#2ecc71"
    elif prob < threshold:
        return "#f39c12"
    return "#e74c3c"


def gauge_chart(prob: float, threshold: float):
    color = get_risk_color(prob, threshold)
    low_end = threshold * 0.6 * 100
    thresh_pct = threshold * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, low_end], "color": "#d5f5e3"},
                {"range": [low_end, thresh_pct], "color": "#fdebd0"},
                {"range": [thresh_pct, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": thresh_pct,
            },
        },
        title={"text": "Churn Probability", "font": {"size": 20}},
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def predict_single(customer_dict: dict, threshold: float):
    df = pd.DataFrame([customer_dict])
    X = PREPROCESSOR.transform(df[FEATURE_COLUMNS])
    proba = MODEL.predict_proba(X)[0]
    churn_prob = float(proba[1])
    return churn_prob >= threshold, churn_prob


def predict_batch(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["TotalCharges"] = pd.to_numeric(
        df_copy["TotalCharges"], errors="coerce"
    ).fillna(0.0)
    X = PREPROCESSOR.transform(df_copy[FEATURE_COLUMNS])
    probas = MODEL.predict_proba(X)
    churn_prob = probas[:, 1]
    churn_pred = churn_prob >= threshold
    result = df_copy.copy()
    result["Churn_Predicted"] = np.where(churn_pred, "Yes", "No")
    result["Churn_Probability"] = np.round(churn_prob * 100, 1)
    result["No_Churn_Probability"] = np.round((1 - churn_prob) * 100, 1)
    low_end = threshold * 0.6
    result["Risk"] = pd.cut(
        churn_prob,
        bins=[-0.001, low_end, threshold, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return result


# ─── Single Customer Tab ─────────────────────────────────────────────────────
if tab_choice == "Single Customer":
    st.header("Single Customer Churn Prediction")
    st.caption("Fill in the customer details below and click Predict.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox(
                "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 29.85)
            total = st.number_input(
                "Total Charges ($)", 0.0, 10000.0, float(tenure * monthly)
            )

        with col3:
            st.subheader("Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"] if phone == "No" else ["No", "Yes"]
            )
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            no_internet = internet == "No"
            internet_na = "No internet service"
            online_sec = st.selectbox(
                "Online Security", [internet_na] if no_internet else ["No", "Yes"]
            )
            online_bak = st.selectbox(
                "Online Backup", [internet_na] if no_internet else ["No", "Yes"]
            )
            device_prot = st.selectbox(
                "Device Protection", [internet_na] if no_internet else ["No", "Yes"]
            )
            tech_sup = st.selectbox(
                "Tech Support", [internet_na] if no_internet else ["No", "Yes"]
            )
            stream_tv = st.selectbox(
                "Streaming TV", [internet_na] if no_internet else ["No", "Yes"]
            )
            stream_mov = st.selectbox(
                "Streaming Movies", [internet_na] if no_internet else ["No", "Yes"]
            )

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted and MODEL_LOADED:
        customer = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
            "DeviceProtection": device_prot, "TechSupport": tech_sup,
            "StreamingTV": stream_tv, "StreamingMovies": stream_mov,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        churn, prob = predict_single(customer, THRESHOLD)

        st.divider()
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.plotly_chart(gauge_chart(prob, THRESHOLD), use_container_width=True)

        with res_col2:
            st.markdown("### Prediction Result")
            if churn:
                st.error("CHURN PREDICTED")
                st.markdown(
                    f"This customer is **likely to churn** with "
                    f"**{prob*100:.1f}%** probability."
                )
            else:
                st.success("NO CHURN PREDICTED")
                st.markdown(
                    f"This customer is **likely to stay** "
                    f"(churn prob: **{prob*100:.1f}%**)."
                )

            risk = get_risk_label(prob, THRESHOLD)
            color = get_risk_color(prob, THRESHOLD)
            st.metric("Churn Probability", f"{prob*100:.1f}%")
            st.metric("Retention Probability", f"{(1-prob)*100:.1f}%")
            st.metric("Risk Level", risk)
            st.caption(f"Threshold used: {THRESHOLD:.3f}")

# ─── CSV Batch Tab ────────────────────────────────────────────────────────────
else:
    st.header("Batch CSV Prediction")
    st.caption(
        "Upload a CSV file in the same format as the Telco dataset. "
        "The Churn column is optional."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded and MODEL_LOADED:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        missing_cols = set(FEATURE_COLUMNS) - set(df_raw.columns)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        with st.spinner("Running predictions..."):
            df_pred = predict_batch(df_raw, THRESHOLD)

        st.success(f"Predictions ready for **{len(df_pred)}** customers")
        st.caption(f"Using threshold: **{THRESHOLD:.3f}**")

        total = len(df_pred)
        churn_n = (df_pred["Churn_Predicted"] == "Yes").sum()
        stay_n = total - churn_n
        avg_prob = df_pred["Churn_Probability"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers", total)
        k2.metric("Predicted Churn", churn_n, f"{churn_n/total*100:.1f}%")
        k3.metric("Predicted Stay", stay_n, f"{stay_n/total*100:.1f}%")
        k4.metric("Avg Churn Prob", f"{avg_prob:.1f}%")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_pie = px.pie(
                names=["Stay", "Churn"], values=[stay_n, churn_n],
                color=["Stay", "Churn"],
                color_discrete_map={"Stay": "#2ecc71", "Churn": "#e74c3c"},
                title="Predicted Churn Distribution",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_hist = px.histogram(
                df_pred, x="Churn_Probability", color="Churn_Predicted",
                color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
                nbins=20, title="Churn Probability Distribution",
                labels={"Churn_Probability": "Churn Probability (%)", "Churn_Predicted": "Churn"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Filter Results")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            churn_filter = st.multiselect(
                "Churn Predicted", ["Yes", "No"], default=["Yes", "No"]
            )
        with filter_col2:
            risk_filter = st.multiselect(
                "Risk Level", ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )
        with filter_col3:
            prob_range = st.slider("Churn Probability (%)", 0, 100, (0, 100))

        cid_search = ""
        if "customerID" in df_pred.columns:
            cid_search = st.text_input("Search by Customer ID (partial match)")

        mask = (
            df_pred["Churn_Predicted"].isin(churn_filter) &
            df_pred["Risk"].isin(risk_filter) &
            (df_pred["Churn_Probability"] >= prob_range[0]) &
            (df_pred["Churn_Probability"] <= prob_range[1])
        )
        if cid_search and "customerID" in df_pred.columns:
            mask &= df_pred["customerID"].astype(str).str.contains(
                cid_search, case=False
            )

        df_filtered = df_pred[mask]
        st.caption(f"Showing **{len(df_filtered)}** of {total} customers")

        display_cols = ["customerID"] if "customerID" in df_filtered.columns else []
        display_cols += [
            "Churn_Predicted", "Churn_Probability", "No_Churn_Probability", "Risk",
            "Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"
        ]
        display_cols = [c for c in display_cols if c in df_filtered.columns]

        st.dataframe(
            df_filtered[display_cols].sort_values("Churn_Probability", ascending=False),
            use_container_width=True, height=400,
        )

        csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered Results as CSV",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

    elif not MODEL_LOADED:
        st.warning("Model not loaded. Run python train.py first.")
    else:
        st.info("Upload a CSV file to get started.")
        st.markdown("""
        **Required columns:**
        `gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges`

        Optional: `customerID`, `Churn` (actual label for comparison).
        """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("Group 7 · MLOps Final Project · Telco Customer Churn Prediction")
