"""
Telco Customer Churn – Streamlit Web Application
Group 7: Pablo Infante · Nuria Etemadi · Tenaw Belete · Jan Wilhelm · Selim El Khoury

Features:
  1. Manual form – fill in one customer's details and get instant churn prediction
  2. CSV upload  – batch predictions with filtering and download
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
    THRESHOLD = METADATA["optimal_threshold"]
    FEATURE_COLUMNS = METADATA["feature_columns"]
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Could not load model: {e}\n\nRun `python train.py` first.")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def predict_single(customer_dict: dict):
    df = pd.DataFrame([customer_dict])
    X = PREPROCESSOR.transform(df[FEATURE_COLUMNS])
    proba = MODEL.predict_proba(X)[0]
    churn_prob = float(proba[1])
    churn = churn_prob >= THRESHOLD
    return churn, churn_prob


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["TotalCharges"] = pd.to_numeric(df_copy["TotalCharges"], errors="coerce").fillna(0.0)
    X = PREPROCESSOR.transform(df_copy[FEATURE_COLUMNS])
    probas = MODEL.predict_proba(X)
    churn_prob = probas[:, 1]
    churn_pred = churn_prob >= THRESHOLD
    result = df_copy.copy()
    result["Churn_Predicted"] = np.where(churn_pred, "Yes", "No")
    result["Churn_Probability"] = np.round(churn_prob * 100, 1)
    result["No_Churn_Probability"] = np.round((1 - churn_prob) * 100, 1)
    result["Risk"] = pd.cut(
        churn_prob,
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["🟢 Low", "🟡 Medium", "🔴 High"]
    )
    return result


def risk_color(prob: float) -> str:
    if prob < 0.30:
        return "#2ecc71"
    elif prob < 0.60:
        return "#f39c12"
    return "#e74c3c"


def gauge_chart(prob: float):
    color = risk_color(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#d5f5e3"},
                {"range": [30, 60], "color": "#fdebd0"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": THRESHOLD * 100 if MODEL_LOADED else 50,
            },
        },
        title={"text": "Churn Probability", "font": {"size": 20}},
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/telecom.png", width=64)
    st.title("Churn Predictor")
    st.caption("Group 7 · MLOps Project")
    st.divider()

    if MODEL_LOADED:
        st.success(f"✅ Model loaded")
        st.caption(f"**Algorithm:** {METADATA.get('model_name', '–')}")
        st.caption(f"**Calibration:** {METADATA.get('calibration_method', '–')}")
        st.caption(f"**Threshold:** {THRESHOLD}")
        m = METADATA.get("metrics", {})
        st.caption(f"**ROC-AUC:** {m.get('roc_auc', '–'):.4f}" if m else "")
        st.caption(f"**Brier Score:** {m.get('brier_score', '–'):.4f}" if m else "")

    st.divider()
    tab_choice = st.radio("Mode", ["🧑 Single Customer", "📂 CSV Batch Prediction"])

# ─── Single Customer Tab ─────────────────────────────────────────────────────
if tab_choice == "🧑 Single Customer":
    st.header("🧑 Single Customer Churn Prediction")
    st.caption("Fill in the customer's details below and click **Predict**.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 29.85)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly))

        with col3:
            st.subheader("Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi = st.selectbox("Multiple Lines",
                                  ["No", "Yes", "No phone service"] if phone == "No"
                                  else ["No", "Yes"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

            no_internet = internet == "No"
            internet_na = "No internet service"

            online_sec = st.selectbox("Online Security",
                                       [internet_na] if no_internet else ["No", "Yes"])
            online_bak = st.selectbox("Online Backup",
                                       [internet_na] if no_internet else ["No", "Yes"])
            device_prot = st.selectbox("Device Protection",
                                        [internet_na] if no_internet else ["No", "Yes"])
            tech_sup = st.selectbox("Tech Support",
                                     [internet_na] if no_internet else ["No", "Yes"])
            stream_tv = st.selectbox("Streaming TV",
                                      [internet_na] if no_internet else ["No", "Yes"])
            stream_mov = st.selectbox("Streaming Movies",
                                       [internet_na] if no_internet else ["No", "Yes"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

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

        churn, prob = predict_single(customer)

        st.divider()
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)

        with res_col2:
            st.markdown("### Prediction Result")
            if churn:
                st.error(f"## ⚠️ CHURN PREDICTED")
                st.markdown(f"This customer is **likely to churn** with **{prob*100:.1f}%** probability.")
            else:
                st.success(f"## ✅ NO CHURN PREDICTED")
                st.markdown(f"This customer is **likely to stay** (churn prob: **{prob*100:.1f}%**).")

            st.metric("Churn Probability", f"{prob*100:.1f}%")
            st.metric("Retention Probability", f"{(1-prob)*100:.1f}%")
            st.metric("Risk Level",
                       "🔴 High" if prob >= 0.60 else ("🟡 Medium" if prob >= 0.30 else "🟢 Low"))
            st.caption(f"Threshold used: {THRESHOLD}")

# ─── CSV Batch Tab ────────────────────────────────────────────────────────────
else:
    st.header("📂 Batch CSV Prediction")
    st.caption("Upload a CSV file in the same format as the Telco dataset. "
               "The `Churn` column is optional (if present, it will be kept for comparison).")

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

        with st.spinner("Running predictions…"):
            df_pred = predict_batch(df_raw)

        st.success(f"✅ Predictions ready for **{len(df_pred)}** customers")

        # ── Summary KPIs ─────────────────────────────────────────────────
        total = len(df_pred)
        churn_n = (df_pred["Churn_Predicted"] == "Yes").sum()
        stay_n = total - churn_n
        avg_prob = df_pred["Churn_Probability"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers", total)
        k2.metric("Predicted Churn", churn_n, f"{churn_n/total*100:.1f}%")
        k3.metric("Predicted Stay", stay_n, f"{stay_n/total*100:.1f}%")
        k4.metric("Avg Churn Prob", f"{avg_prob:.1f}%")

        # ── Charts ───────────────────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_pie = px.pie(
                names=["Stay", "Churn"],
                values=[stay_n, churn_n],
                color=["Stay", "Churn"],
                color_discrete_map={"Stay": "#2ecc71", "Churn": "#e74c3c"},
                title="Predicted Churn Distribution",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_hist = px.histogram(
                df_pred, x="Churn_Probability",
                color="Churn_Predicted",
                color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
                nbins=20, title="Churn Probability Distribution",
                labels={"Churn_Probability": "Churn Probability (%)", "Churn_Predicted": "Churn"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Filters ──────────────────────────────────────────────────────
        st.subheader("🔎 Filter Results")
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            churn_filter = st.multiselect(
                "Churn Predicted", ["Yes", "No"], default=["Yes", "No"]
            )
        with filter_col2:
            risk_filter = st.multiselect(
                "Risk Level", ["🔴 High", "🟡 Medium", "🟢 Low"],
                default=["🔴 High", "🟡 Medium", "🟢 Low"]
            )
        with filter_col3:
            prob_range = st.slider("Churn Probability (%)", 0, 100, (0, 100))

        if "customerID" in df_pred.columns:
            cid_search = st.text_input("Search by Customer ID (partial match)")
        else:
            cid_search = ""

        # Apply filters
        mask = (
            df_pred["Churn_Predicted"].isin(churn_filter) &
            df_pred["Risk"].isin(risk_filter) &
            (df_pred["Churn_Probability"] >= prob_range[0]) &
            (df_pred["Churn_Probability"] <= prob_range[1])
        )
        if cid_search and "customerID" in df_pred.columns:
            mask &= df_pred["customerID"].astype(str).str.contains(cid_search, case=False)

        df_filtered = df_pred[mask]
        st.caption(f"Showing **{len(df_filtered)}** of {total} customers")

        # Display columns
        display_cols = ["customerID"] if "customerID" in df_filtered.columns else []
        display_cols += ["Churn_Predicted", "Churn_Probability", "No_Churn_Probability", "Risk",
                          "Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"]
        display_cols = [c for c in display_cols if c in df_filtered.columns]

        st.dataframe(
            df_filtered[display_cols].sort_values("Churn_Probability", ascending=False),
            use_container_width=True,
            height=400,
        )

        # ── Download ─────────────────────────────────────────────────────
        csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered Results as CSV",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

    elif not MODEL_LOADED:
        st.warning("Model not loaded. Run `python train.py` first.")
    else:
        st.info("👆 Upload a CSV file to get started.")
        st.markdown("""
        **Required columns** (same format as the Telco dataset):
        `gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges`

        Optional: `customerID`, `Churn` (actual label for comparison).
        """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("Group 7 · MLOps Final Project · Telco Customer Churn Prediction")
