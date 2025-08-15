import os
import subprocess
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib
import time

# --- Auto-start simulator in background ---
SIMULATOR_FILE = os.path.join(os.path.dirname(__file__), "simulator.py")
if not any("simulator.py" in p for p in os.popen('tasklist').read().splitlines()):
    subprocess.Popen(["python", SIMULATOR_FILE], creationflags=subprocess.CREATE_NO_WINDOW)

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LIVE_PATH = os.path.join(DATA_DIR, "live.csv")
MODEL_PATH = os.path.join(DATA_DIR, "fault_model.pkl")

# --- Page setup ---
st.set_page_config(page_title="FSCUT2000E Monitoring Dashboard", layout="wide")
st.title("🖥️ FSCUT2000E Virtual Monitoring & Predictive Maintenance")

# --- Helper: refresh page ---
def st_autorefresh():
    st.rerun()

# --- Load model ---
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("No trained model found. Please run `python model.py` first.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Controls")
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 3)

# --- Load live data ---
if not os.path.exists(LIVE_PATH) or os.stat(LIVE_PATH).st_size == 0:
    st.warning("No live data yet. Waiting for simulator to start...")
    time.sleep(refresh_rate)
    st_autorefresh()

df = pd.read_csv(LIVE_PATH)

# --- Ensure timestamp column exists ---
if "timestamp" not in df.columns:
    df["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))

# --- Ensure required columns exist ---
required_cols = ["machine_status", "laser_power_w", "gas_pressure_bar", "head_height_mm", "machine_age"]
for col in required_cols:
    if col not in df.columns:
        df[col] = 0
df.fillna({
    "laser_power_w": 1000,
    "gas_pressure_bar": 5,
    "head_height_mm": 1,
    "machine_age": 1
}, inplace=True)

latest = df.iloc[-1]

# --- Prepare features for prediction ---
features = pd.DataFrame([{
    "laser_power_w": latest["laser_power_w"],
    "gas_pressure_bar": latest["gas_pressure_bar"],
    "head_height_mm": latest["head_height_mm"],
    "machine_age": latest.get("machine_age", 1),
    "is_running": 1 if latest["machine_status"] == "Running" else 0
}])

fault_prob = model.predict_proba(features)[0][1] * 100

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "📥 Export Data"])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Machine Status", latest["machine_status"])
    col2.metric("Laser Power (W)", f"{latest['laser_power_w']:.1f}")
    col3.metric("Gas Pressure (bar)", f"{latest['gas_pressure_bar']:.2f}")
    col4.metric("Head Height (mm)", f"{latest['head_height_mm']:.2f}")

    # Fault risk bar + alert
    st.subheader("🔮 Predicted Fault Risk (next 10 min)")
    st.progress(int(fault_prob))
    st.write(f"**Risk:** {fault_prob:.2f}%")
    if fault_prob > 70:
        st.error("⚠ High Fault Risk! Immediate inspection recommended.")

    # Uptime/Downtime summary
    st.subheader("⏱ Machine State Summary")
    state_counts = df["machine_status"].value_counts(normalize=True) * 100
    st.write(state_counts.round(2).to_frame("Percentage (%)"))

# ========== TAB 2: TRENDS ==========
with tab2:
    st.subheader("📊 Live Gauges")
    gc1, gc2, gc3 = st.columns(3)

    gauge1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['laser_power_w'],
        title={'text': "Laser Power (W)"},
        gauge={'axis': {'range': [500, 1600]}}
    ))
    gc1.plotly_chart(gauge1, use_container_width=True)

    gauge2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['gas_pressure_bar'],
        title={'text': "Gas Pressure (bar)"},
        gauge={'axis': {'range': [2, 8]}}
    ))
    gc2.plotly_chart(gauge2, use_container_width=True)

    gauge3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['head_height_mm'],
        title={'text': "Head Height (mm)"},
        gauge={'axis': {'range': [0, 2]}}
    ))
    gc3.plotly_chart(gauge3, use_container_width=True)

        # Separate trend charts (Full Width)
    st.subheader("📈 Parameter Trends (Last 50 readings)")
    trend_df = df.tail(50)

    # Laser Power Trend
    fig_laser = go.Figure()
    fig_laser.add_trace(go.Scatter(
        x=trend_df["timestamp"],
        y=trend_df["laser_power_w"],
        mode="lines+markers",
        name="Laser Power (W)",
        line=dict(color="red")
    ))
    fig_laser.update_layout(
        title="Laser Power Trend",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        height=300
    )
    st.plotly_chart(fig_laser, use_container_width=True)

    # Gas Pressure Trend
    fig_pressure = go.Figure()
    fig_pressure.add_trace(go.Scatter(
        x=trend_df["timestamp"],
        y=trend_df["gas_pressure_bar"],
        mode="lines+markers",
        name="Gas Pressure (bar)",
        line=dict(color="blue")
    ))
    fig_pressure.update_layout(
        title="Gas Pressure Trend",
        xaxis_title="Time",
        yaxis_title="Pressure (bar)",
        height=300
    )
    st.plotly_chart(fig_pressure, use_container_width=True)

    # Head Height Trend
    fig_height = go.Figure()
    fig_height.add_trace(go.Scatter(
        x=trend_df["timestamp"],
        y=trend_df["head_height_mm"],
        mode="lines+markers",
        name="Head Height (mm)",
        line=dict(color="green")
    ))
    fig_height.update_layout(
        title="Head Height Trend",
        xaxis_title="Time",
        yaxis_title="Height (mm)",
        height=300
    )
    st.plotly_chart(fig_height, use_container_width=True)

# ========== TAB 3: EXPORT ==========
with tab3:
    st.subheader("📥 Download Data")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="fscut2000e_live_data.csv",
        mime="text/csv"
    )

# --- Auto-refresh ---
time.sleep(refresh_rate)
st_autorefresh()
