# ===== app.py (patched with hardcoded key) =====
import os
import sys
import time
import shutil
import platform
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib
import numpy as np
import os
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()  # loads values from .env into environment
import os
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
SIMULATOR_FILE = os.path.join(BASE_DIR, "simulator.py")
LIVE_PATH = os.path.join(DATA_DIR, "live.csv")
HISTORICAL_PATH = os.path.join(DATA_DIR, "historical.csv")
MODEL_PATH = os.path.join(DATA_DIR, "fault_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# --- Page setup ---
st.set_page_config(page_title="FSCUT2000E Monitoring Dashboard", layout="wide")
st.title("🖥️ FSCUT2000E Virtual Monitoring & Predictive Maintenance")

# --- Utilities ---
@st.cache_resource
def ensure_simulator_running():
    """Start the simulator once per Streamlit session (safe on reruns)."""
    if not os.path.exists(LIVE_PATH):
        pd.DataFrame(columns=[
            "timestamp",
            "machine_status",
            "laser_power_w",
            "gas_pressure_bar",
            "head_height_mm",
            "machine_age",
        ]).to_csv(LIVE_PATH, index=False)
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform.system() == "Windows" else 0
    except Exception:
        flags = 0
    return subprocess.Popen([sys.executable, SIMULATOR_FILE], creationflags=flags)


def safe_read_csv(path: str, retries: int = 4, delay: float = 0.25) -> pd.DataFrame:
    last_err = None
    for _ in range(retries):
        try:
            return pd.read_csv(path, on_bad_lines="skip")
        except Exception as e1:
            last_err = e1
            try:
                snap = os.path.join(DATA_DIR, "_live_snapshot.csv")
                shutil.copyfile(path, snap)
                return pd.read_csv(snap, on_bad_lines="skip")
            except Exception as e2:
                last_err = e2
                time.sleep(delay)
    raise RuntimeError(f"Could not read {path} safely after retries. Last error: {last_err}")


def explain_prediction_logreg(model, features: pd.DataFrame) -> pd.DataFrame:
    coefs = model.coef_[0]
    names = list(features.columns)
    vals = features.iloc[0].values.astype(float)
    contrib = coefs * vals
    dfc = pd.DataFrame({
        "feature": names,
        "value": vals,
        "coef": coefs,
        "contribution": contrib,
    }).sort_values("contribution", key=lambda s: s.abs(), ascending=False)
    return dfc


# --- Auto-start simulator ---
import subprocess
try:
    sim_proc = ensure_simulator_running()
except Exception as e:
    st.warning(f"Simulator couldn't be auto-started: {e}. Run manually: `python simulator.py`.")

# --- Load or train model ---
@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        from model import train_model
        train_model()
    return joblib.load(MODEL_PATH)

model = load_or_train_model()

# --- Sidebar ---
st.sidebar.header("Controls")
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 3)

# --- Load live data ---
if not os.path.exists(LIVE_PATH) or os.stat(LIVE_PATH).st_size == 0:
    st.warning("No live data yet. Waiting for simulator...")
    time.sleep(refresh_rate)
    st.rerun()

df = safe_read_csv(LIVE_PATH)

# sanitize
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    df["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))

required_cols = ["machine_status", "laser_power_w", "gas_pressure_bar", "head_height_mm", "machine_age"]
for col in required_cols:
    if col not in df.columns:
        df[col] = 0

df.fillna({
    "laser_power_w": 1000,
    "gas_pressure_bar": 5,
    "head_height_mm": 1,
    "machine_age": 1,
    "machine_status": "Idle",
}, inplace=True)

latest = df.iloc[-1]

# --- Features for prediction ---
features = pd.DataFrame([{
    "laser_power_w": float(latest.get("laser_power_w", 1000)),
    "gas_pressure_bar": float(latest.get("gas_pressure_bar", 5)),
    "head_height_mm": float(latest.get("head_height_mm", 1)),
    "machine_age": float(latest.get("machine_age", 1)),
    "is_running": 1 if str(latest.get("machine_status", "")).strip() == "Running" else 0,
}])

fault_prob = float(model.predict_proba(features)[0][1]) * 100.0

# --- Save risk history ---
risk_record = {
    "timestamp": latest.get("timestamp", pd.Timestamp.now()),
    "laser_power_w": features.loc[0, "laser_power_w"],
    "gas_pressure_bar": features.loc[0, "gas_pressure_bar"],
    "head_height_mm": features.loc[0, "head_height_mm"],
    "machine_age": features.loc[0, "machine_age"],
    "is_running": features.loc[0, "is_running"],
    "predicted_fault_risk": round(fault_prob, 4),
}
hist_df = pd.DataFrame([risk_record])
if not os.path.exists(HISTORICAL_PATH):
    hist_df.to_csv(HISTORICAL_PATH, index=False)
else:
    hist_df.to_csv(HISTORICAL_PATH, mode="a", header=False, index=False)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "📥 Export Data"])

# ========== TAB 1 ==========
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Machine Status", str(latest.get("machine_status", "Unknown")))
    col2.metric("Laser Power (W)", f"{features.loc[0, 'laser_power_w']:.1f}")
    col3.metric("Gas Pressure (bar)", f"{features.loc[0, 'gas_pressure_bar']:.2f}")
    col4.metric("Head Height (mm)", f"{features.loc[0, 'head_height_mm']:.2f}")

    st.subheader("🔮 Predicted Fault Risk")
    st.progress(int(min(max(fault_prob, 0), 100)))
    st.write(f"**Risk:** {fault_prob:.2f}%")
    if fault_prob > 70:
        st.error("⚠ High Fault Risk! Immediate inspection recommended.")

    contrib_df = explain_prediction_logreg(model, features)
    st.subheader("📌 Why this risk?")
    st.dataframe(contrib_df, use_container_width=True)

    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(x=contrib_df["feature"], y=contrib_df["contribution"]))
    fig_c.update_layout(title="Feature Contributions", xaxis_title="Feature", yaxis_title="Contribution")
    st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("⏱ Machine State Summary")
    state_counts = df["machine_status"].value_counts(normalize=True) * 100
    st.write(state_counts.round(2).to_frame("Percentage (%)"))

# ========== TAB 2 ==========
with tab2:
    st.subheader("📊 Live Gauges")
    gc1, gc2, gc3 = st.columns(3)

    gauge1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=features.loc[0, 'laser_power_w'],
        title={'text': "Laser Power (W)"},
        gauge={'axis': {'range': [500, 1600]}}
    ))
    gc1.plotly_chart(gauge1, use_container_width=True)

    gauge2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=features.loc[0, 'gas_pressure_bar'],
        title={'text': "Gas Pressure (bar)"},
        gauge={'axis': {'range': [2, 8]}}
    ))
    gc2.plotly_chart(gauge2, use_container_width=True)

    gauge3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=features.loc[0, 'head_height_mm'],
        title={'text': "Head Height (mm)"},
        gauge={'axis': {'range': [0, 2]}}
    ))
    gc3.plotly_chart(gauge3, use_container_width=True)

    st.subheader("📈 Parameter Trends (Last 50 readings)")
    trend_df = df.tail(50)

    fig_laser = go.Figure()
    fig_laser.add_trace(go.Scatter(x=trend_df["timestamp"], y=trend_df["laser_power_w"], mode="lines+markers", name="Laser Power (W)"))
    fig_laser.update_layout(title="Laser Power Trend", xaxis_title="Time", yaxis_title="Power (W)", height=300)
    st.plotly_chart(fig_laser, use_container_width=True)

    fig_pressure = go.Figure()
    fig_pressure.add_trace(go.Scatter(x=trend_df["timestamp"], y=trend_df["gas_pressure_bar"], mode="lines+markers", name="Gas Pressure (bar)"))
    fig_pressure.update_layout(title="Gas Pressure Trend", xaxis_title="Time", yaxis_title="Pressure (bar)", height=300)
    st.plotly_chart(fig_pressure, use_container_width=True)

    fig_height = go.Figure()
    fig_height.add_trace(go.Scatter(x=trend_df["timestamp"], y=trend_df["head_height_mm"], mode="lines+markers", name="Head Height (mm)"))
    fig_height.update_layout(title="Head Height Trend", xaxis_title="Time", yaxis_title="Height (mm)", height=300)
    st.plotly_chart(fig_height, use_container_width=True)

# ========== TAB 3 ==========
with tab3:
    st.subheader("📥 Download Data")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="fscut2000e_live_data.csv",
        mime="text/csv"
    )

    # --- AI Report ---
    if st.button("Generate AI Report"):
        with st.spinner("Generating AI report..."):
            data_preview = df.head(20).to_string(index=False)
            prompt = f"""
            You are an AI data analyst. Analyze the following machine data and create
            a professional summary highlighting trends, risks, and anomalies:

            {data_preview}
            """
            try:
                # Instantiate the Gemini model
                model = genai.GenerativeModel('gemini-1.5-flash')

                # Call the Gemini API to generate the report
                response = model.generate_content(prompt)
                
                # Access the generated text from the response
                report_text = response.text
                
                st.subheader("📊 AI Generated Report")
                st.write(report_text)
                st.download_button(
                    label="Download AI Report",
                    data=report_text.encode("utf-8"),
                    file_name="AI_Report.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating AI report: {e}")

    if os.path.exists(HISTORICAL_PATH):
        st.subheader("📜 Historical Fault Risks (last 200)")
        hist = pd.read_csv(HISTORICAL_PATH, on_bad_lines="skip").tail(200)
        st.dataframe(hist, use_container_width=True)
    else:
        st.info("No historical risk data yet.")

# --- Auto-refresh ---
time.sleep(int(refresh_rate))
st.rerun()
