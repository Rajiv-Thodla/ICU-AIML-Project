"""
app.py  —  ICU Patient Deterioration Prediction — Streamlit Frontend
=====================================================================
Usage:
    streamlit run app.py

Requires the models/ directory to contain the artifacts produced by the training scripts.
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from preprocess import (
    VITAL_COLS, VITAL_CLIP_RANGES, WINDOW,
    load_and_clean_data, create_sequences, create_xgb_features,
    build_manual_sequence, build_xgb_manual_features,
    fit_scaler, apply_scaler,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Deterioration Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main { background: #0d1117; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label { color: #8b949e; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .metric-card .value { color: #e6edf3; font-size: 2rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }

    .risk-high   { background: #2d1b1b; border-color: #f85149; }
    .risk-medium { background: #2d2200; border-color: #e3b341; }
    .risk-low    { background: #0d2114; border-color: #3fb950; }

    .risk-badge {
        display: inline-block;
        padding: 0.35em 1em;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.04em;
        font-family: 'IBM Plex Mono', monospace;
    }
    .badge-high   { background: #f8514933; color: #f85149; border: 1px solid #f85149; }
    .badge-medium { background: #e3b34133; color: #e3b341; border: 1px solid #e3b341; }
    .badge-low    { background: #3fb95033; color: #3fb950; border: 1px solid #3fb950; }

    .section-header {
        color: #8b949e;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
    }

    div[data-testid="stNumberInput"] input {
        font-family: 'IBM Plex Mono', monospace;
        background: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3;
    }

    .stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Model loading  (cached so we don't re-load on every interaction)
# ═══════════════════════════════════════════════════════════════════════════
MODELS_DIR = "models"

@st.cache_resource(show_spinner="Loading model…")
def load_xgb():
    model_path   = os.path.join(MODELS_DIR, "xgb_model.joblib")
    scaler_path  = os.path.join(MODELS_DIR, "xgb_scaler.joblib")
    feature_path = os.path.join(MODELS_DIR, "xgb_features.json")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_path]):
        return None, None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_path) as f:
        features = json.load(f)
    return model, scaler, features


@st.cache_resource(show_spinner="Loading model…")
def load_lstm():
    from tensorflow.keras.models import load_model
    model_path  = os.path.join(MODELS_DIR, "lstm_model.h5")
    scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.joblib")
    meta_path   = os.path.join(MODELS_DIR, "lstm_meta.json")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
        return None, None, None
    model  = load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, scaler, meta


@st.cache_resource(show_spinner="Loading model…")
def load_gru():
    import torch
    import torch.nn as nn

    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
            self.fc      = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            out, _ = self.gru(x)
            return self.sigmoid(self.fc(out[:, -1, :]))

    model_path  = os.path.join(MODELS_DIR, "gru_model.pt")
    scaler_path = os.path.join(MODELS_DIR, "gru_scaler.joblib")
    meta_path   = os.path.join(MODELS_DIR, "gru_meta.json")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)

    model = GRUModel(input_size=meta["n_feat"],
                     hidden_size=meta.get("hidden_size", 64),
                     num_layers=meta.get("num_layers", 2))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler, meta

@st.cache_resource(show_spinner="Loading model…")
def load_tcn():
    import torch
    import torch.nn as nn

    class TemporalBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        def forward(self, x):
            out = self.conv1(x)
            out = out[:, :, :x.size(2)]
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)
            out = out[:, :, :x.size(2)]
            out = self.relu(out)
            out = self.dropout(out)
            residual = x if self.downsample is None else self.downsample(x)
            return self.relu(out + residual)

    class TCNModel(nn.Module):
        def __init__(self, input_size, channels=[64, 64, 64], kernel_size=3, dropout=0.3):
            super().__init__()
            layers = []
            for i, out_channels in enumerate(channels):
                in_channels = input_size if i == 0 else channels[i - 1]
                layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation=2 ** i, dropout=dropout))
            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(channels[-1], 1)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            out = self.network(x)
            out = out[:, :, -1]
            return self.fc(out)

    model_path  = os.path.join(MODELS_DIR, "tcn_model.pt")
    scaler_path = os.path.join(MODELS_DIR, "tcn_scaler.joblib")
    meta_path   = os.path.join(MODELS_DIR, "tcn_meta.json")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)

    model = TCNModel(input_size=meta["n_feat"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler, meta

# --- ARIMA MATH HELPERS ---
def ar1_coeff(series):
    if len(series) < 3: return 0.0
    mean = np.mean(series)
    centered = series - mean
    c0 = np.sum(centered ** 2)
    if c0 == 0: return 0.0
    return np.sum(centered[:-1] * centered[1:]) / c0

def extract_arima_features(window_data, vital_cols):
    feat = {}
    t = np.arange(len(window_data), dtype=np.float64)
    for idx, col in enumerate(vital_cols):
        s = window_data[:, idx].astype(np.float64)
        feat[f"{col}_mean"]  = np.mean(s)
        feat[f"{col}_std"]   = np.std(s)
        feat[f"{col}_min"]   = np.min(s)
        feat[f"{col}_max"]   = np.max(s)
        feat[f"{col}_last"]  = s[-1]
        feat[f"{col}_range"] = np.ptp(s)
        phi = ar1_coeff(s)
        feat[f"{col}_ar1"] = phi
        residuals = s[1:] - phi * s[:-1]
        feat[f"{col}_resid_var"] = np.var(residuals) if len(residuals) > 0 else 0.0
        feat[f"{col}_ma1"] = ar1_coeff(residuals) if len(residuals) >= 3 else 0.0
        feat[f"{col}_trend"] = np.polyfit(t, s, 1)[0] if np.std(s) > 0 else 0.0
        mu = np.mean(s)
        forecast = mu + phi * (s[-1] - mu)
        feat[f"{col}_forecast1"] = forecast
        feat[f"{col}_forecast_delta"] = forecast - s[-1]
    return feat

@st.cache_resource(show_spinner="Loading model…")
def load_arima():
    model_path   = os.path.join(MODELS_DIR, "arima_model.joblib")
    scaler_path  = os.path.join(MODELS_DIR, "arima_scaler.joblib")
    meta_path    = os.path.join(MODELS_DIR, "arima_meta.json")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
        return None, None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, scaler, meta


def model_available(name):
    """Check whether a model's files exist on disk."""
    files = {
        "XGBoost": ["xgb_model.joblib", "xgb_scaler.joblib", "xgb_features.json"],
        "LSTM":    ["lstm_model.h5",     "lstm_scaler.joblib", "lstm_meta.json"],
        "GRU":     ["gru_model.pt",      "gru_scaler.joblib",  "gru_meta.json"],
        "TCN":     ["tcn_model.pt",      "tcn_scaler.joblib",  "tcn_meta.json"],
        "ARIMA":   ["arima_model.joblib", "arima_scaler.joblib", "arima_meta.json"],
    }
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files[name])


# ═══════════════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════════════

def predict_xgb(model, scaler, feature_names, hourly_rows):
    feat_dict = build_xgb_manual_features(hourly_rows)
    row = pd.DataFrame([feat_dict])[feature_names]
    row_s = scaler.transform(row)
    prob  = model.predict_proba(row_s)[0, 1]
    return float(prob)

def predict_arima(model, scaler, feature_names, hourly_rows):
    df = pd.DataFrame(hourly_rows)
    vital_data = df[VITAL_COLS].values 
    feat_dict = extract_arima_features(vital_data, VITAL_COLS)
    row = pd.DataFrame([feat_dict])[feature_names]
    row_s = scaler.transform(row)
    prob  = model.predict_proba(row_s)[0, 1]
    return float(prob)

def predict_sequence_model(model, scaler, feature_cols, hourly_rows, model_type):
    """Shared inference for sequence models."""
    X = build_manual_sequence(hourly_rows, feature_cols, scaler)  # (1, T, F)
    if model_type == "LSTM":
        prob = float(model.predict(X, verbose=0)[0, 0])
    else:  # GRU or TCN (PyTorch)
        import torch
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            prob = float(model(t)[0, 0].item())
            
            # Ensure TCN logits are passed through sigmoid if it lacks one in the final layer
            if model_type == "TCN":
                prob = float(torch.sigmoid(torch.tensor(prob)).item())
                
    return prob

def risk_label(prob, threshold):
    """Dynamically labels risk based on the user-selected threshold."""
    if prob >= threshold: # High Risk
        return "HIGH", "badge-high", "risk-high"
    elif prob >= (threshold - 0.20): # Buffer zone for moderate risk
        return "MODERATE", "badge-medium", "risk-medium"
    else:
        return "LOW", "badge-low", "risk-low"


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏥 ICU Monitor")
    st.markdown("---")

    st.markdown('<div class="section-header">Model Selection</div>', unsafe_allow_html=True)

    available = [m for m in ["XGBoost", "LSTM", "GRU", "TCN", "ARIMA"] if model_available(m)]
    if not available:
        st.error("No trained models found in `models/`.\nRun the training scripts first.")
        st.code("python train_xgb.py  --data data.csv\npython train_lstm.py --data data.csv\npython train_gru.py  --data data.csv\npython train_tcn.py  --data data.csv")
        st.stop()

    model_choice = st.selectbox("Active model", available)

    for m in ["XGBoost", "LSTM", "GRU", "TCN", "ARIMA"]:
        icon = "✅" if model_available(m) else "⬜"
        st.markdown(f"{icon} {m}")

    st.markdown("---")
    # NEW FEATURE: Dynamic Threshold Slider
    st.markdown('<div class="section-header">⚙️ Clinical Settings</div>', unsafe_allow_html=True)
    
    # Define the mathematically optimal thresholds for each model
    optimal_thresholds = {
        "GRU": 0.80,
        "TCN": 0.30,
        "LSTM": 0.30,
        "XGBoost": 0.30,
        "ARIMA": 0.30
    }
    
    user_threshold = st.slider(
        "Alarm Sensitivity (Threshold)", 
        min_value=0.10, 
        max_value=0.90, 
        value=optimal_thresholds[model_choice], # Automatically updates based on selection
        step=0.05,
        help="Lower values catch more cases but increase False Alarms."
    )
    
    st.markdown("---")
    st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
    st.caption(
        "Predicts sepsis onset risk using ICU time-series vitals. "
        "Window: 6 hours history → 3 hours ahead prediction."
    )
    st.markdown("---")
    st.caption("For clinical research only. Not for diagnostic use.")


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<h1 style="color:#e6edf3; font-family:'IBM Plex Sans'; font-weight:300; margin-bottom:0;">
    ICU <span style="font-weight:600;">Deterioration</span> Monitor
</h1>
<p style="color:#8b949e; margin-top:0.2rem;">
    Active model: <span style="font-family:'IBM Plex Mono'; color:#58a6ff;">{model_choice}</span>
    &nbsp;·&nbsp; Prediction window: 6 h history → 3 h ahead
</p>
<hr style="border-color:#21262d; margin:1rem 0 1.5rem;">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Load chosen model
# ═══════════════════════════════════════════════════════════════════════════

if model_choice == "XGBoost":
    mdl, scaler, meta = load_xgb()
    feature_cols = meta
elif model_choice == "LSTM":
    mdl, scaler, meta = load_lstm()
    feature_cols = meta["feature_cols"]
elif model_choice == "GRU":
    mdl, scaler, meta = load_gru()
    feature_cols = meta["feature_cols"]
elif model_choice == "ARIMA":
    mdl, scaler, meta = load_arima()
    feature_cols = meta["arima_feature_cols"]
else:
    mdl, scaler, meta = load_tcn()
    feature_cols = meta["feature_cols"]


# ═══════════════════════════════════════════════════════════════════════════
# Two input modes
# ═══════════════════════════════════════════════════════════════════════════

tab_csv, tab_manual = st.tabs(["📂  Upload Time-Series CSV", "⌨️  Enter Data Manually"])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — CSV Upload
# ───────────────────────────────────────────────────────────────────────────
with tab_csv:
    st.markdown("Upload a patient CSV. The file should match the training dataset schema with "
                "columns: `Patient_ID`, `Hour`, `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `Resp`, "
                "`Age`, `Gender`, `SepsisLabel` (SepsisLabel is optional for inference).")

    uploaded = st.file_uploader("Choose CSV", type=["csv"], key="csv_upload")

    if uploaded:
        with st.spinner("Cleaning data…"):
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                shutil.copyfileobj(uploaded, tmp)
                tmp_path = tmp.name

            df_raw, feat_cols, id_col = load_and_clean_data(tmp_path)
            os.unlink(tmp_path)

        st.success(f"Loaded {df_raw.shape[0]:,} rows · {df_raw[id_col].nunique()} patients")

        # Patient selector
        patients = sorted(df_raw[id_col].unique())
        sel_patient = st.selectbox("Select patient to predict", patients)

        patient_df = df_raw[df_raw[id_col] == sel_patient].sort_values("Hour")

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(f"**Hours on record:** {len(patient_df)}")
        with col_info2:
            if "SepsisLabel" in patient_df.columns:
                actual = int(patient_df["SepsisLabel"].max())
                st.markdown(f"**Actual label:** {'🔴 Sepsis' if actual else '🟢 No sepsis'}")

        if len(patient_df) < WINDOW:
            st.warning(f"Patient has only {len(patient_df)} hours of data. "
                       f"Need at least {WINDOW} hours for prediction.")
        else:
            # Show raw vitals
            with st.expander("📊 Patient vitals over time", expanded=True):
                chart_cols = [c for c in VITAL_COLS if c in patient_df.columns]
                st.line_chart(patient_df.set_index("Hour")[chart_cols])

            # Run prediction
            if st.button("🔍 Run Prediction", key="csv_predict", type="primary"):
                with st.spinner("Running inference…"):
                    hourly_rows = patient_df[
                        [c for c in VITAL_COLS + ["Age", "Gender"] if c in patient_df.columns]
                    ].tail(WINDOW).to_dict("records")

                    if model_choice == "XGBoost":
                        prob = predict_xgb(mdl, scaler, feature_cols, hourly_rows)
                    elif model_choice == "ARIMA":
                        prob = predict_arima(mdl, scaler, feature_cols, hourly_rows)
                    else:
                        prob = predict_sequence_model(
                            mdl, scaler, feature_cols, hourly_rows, model_choice
                        )

                # DYNAMIC LOGIC PASSED HERE
                label, badge_cls, card_cls = risk_label(prob, user_threshold)

                st.markdown("---")
                st.markdown("### Prediction Result")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Sepsis Risk Score</div>
                        <div class="value">{prob:.1%}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card {card_cls}">
                        <div class="label">Risk Level</div>
                        <div class="value" style="font-size:1.4rem;">
                            <span class="risk-badge {badge_cls}">{label}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Model</div>
                        <div class="value" style="font-size:1.2rem;">{model_choice}</div>
                    </div>""", unsafe_allow_html=True)

                if label == "HIGH":
                    st.error(f"⚠️  Risk score exceeds your {user_threshold:.2f} threshold. Consider immediate clinical review.")
                elif label == "MODERATE":
                    st.warning("⚡  Moderate risk. Recommend increased monitoring frequency.")
                else:
                    st.success("✓  Low risk. Continue standard monitoring protocol.")

                # Sliding window — predict every hour
                st.markdown("---")
                st.markdown("#### Risk Score Over Time (sliding window)")

                n_hours = len(patient_df)
                if n_hours >= WINDOW:
                    hour_probs = []
                    hours_axis = []
                    rows_all   = patient_df[
                        [c for c in VITAL_COLS + ["Age", "Gender"] if c in patient_df.columns]
                    ].to_dict("records")

                    for i in range(WINDOW, n_hours + 1):
                        window_rows = rows_all[i - WINDOW : i]
                        if model_choice == "XGBoost":
                            p = predict_xgb(mdl, scaler, feature_cols, window_rows)
                        elif model_choice == "ARIMA":
                            p = predict_arima(mdl, scaler, feature_cols, window_rows)
                        else:
                            p = predict_sequence_model(mdl, scaler, feature_cols,
                                                       window_rows, model_choice)
                        hour_probs.append(p)
                        hours_axis.append(patient_df["Hour"].iloc[i - 1])

                    chart_df = pd.DataFrame({
                        "Hour":       hours_axis,
                        "Risk Score": hour_probs,
                    }).set_index("Hour")
                    st.line_chart(chart_df)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — Manual Entry
# ───────────────────────────────────────────────────────────────────────────
with tab_manual:
    st.markdown("Enter vital signs for each hour manually. "
                "The model uses the last **6 hours** to predict risk over the **next 3 hours**.")

    st.markdown('<div class="section-header">Patient Info</div>', unsafe_allow_html=True)

    col_age, col_gender = st.columns(2)
    with col_age:
        age = st.number_input("Age", min_value=0, max_value=120, value=65)
    with col_gender:
        gender = st.selectbox("Gender", ["Male (1)", "Female (0)"])
        gender_val = 1 if gender.startswith("Male") else 0

    st.markdown("---")
    st.markdown('<div class="section-header">Number of Hours to Enter</div>', unsafe_allow_html=True)

    n_hours = st.slider(
        "Hours of data", min_value=WINDOW, max_value=24, value=WINDOW,
        help=f"Minimum {WINDOW} hours required. Model always uses the last {WINDOW} hours."
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Hourly Vital Signs</div>', unsafe_allow_html=True)

    # Normal reference ranges for placeholder defaults
    DEFAULTS = {
        "HR":    75.0,
        "O2Sat": 97.0,
        "Temp":  37.0,
        "SBP":   120.0,
        "MAP":   80.0,
        "Resp":  16.0,
    }
    UNITS = {
        "HR": "bpm", "O2Sat": "%", "Temp": "°C",
        "SBP": "mmHg", "MAP": "mmHg", "Resp": "breaths/min",
    }
    RANGES = {
        "HR":    (20,  300),
        "O2Sat": (50,  100),
        "Temp":  (25,   45),
        "SBP":   (40,  280),
        "MAP":   (20,  180),
        "Resp":  (4,    60),
    }

    hourly_rows = []

    for h in range(n_hours):
        with st.expander(f"Hour {h + 1}", expanded=(h < 3)):
            cols = st.columns(6)
            row  = {"Age": age, "Gender": gender_val}
            for i, vital in enumerate(VITAL_COLS):
                lo, hi = RANGES[vital]
                with cols[i]:
                    val = st.number_input(
                        f"{vital} ({UNITS[vital]})",
                        min_value=float(lo),
                        max_value=float(hi),
                        value=DEFAULTS[vital],
                        step=0.1,
                        key=f"manual_{vital}_{h}",
                    )
                    row[vital] = val
            hourly_rows.append(row)

    st.markdown("---")

    # Live mini-chart of entered data
    if hourly_rows:
        live_df = pd.DataFrame(hourly_rows)[VITAL_COLS]
        live_df.index = [f"H{i+1}" for i in range(len(live_df))]
        with st.expander("📊 Preview entered vitals", expanded=False):
            st.line_chart(live_df)

    if st.button("🔍 Run Prediction", key="manual_predict", type="primary"):
        if len(hourly_rows) < WINDOW:
            st.error(f"Please enter at least {WINDOW} hours of data.")
        else:
            with st.spinner("Running inference…"):
                window_rows = hourly_rows[-WINDOW:]   # always use last WINDOW hours

                if model_choice == "XGBoost":
                    prob = predict_xgb(mdl, scaler, feature_cols, window_rows)
                elif model_choice == "ARIMA":
                    prob = predict_arima(mdl, scaler, feature_cols, window_rows)
                else:
                    prob = predict_sequence_model(
                        mdl, scaler, feature_cols, window_rows, model_choice
                    )

            # DYNAMIC LOGIC PASSED HERE
            label, badge_cls, card_cls = risk_label(prob, user_threshold)

            st.markdown("---")
            st.markdown("### Prediction Result")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Sepsis Risk Score</div>
                    <div class="value">{prob:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card {card_cls}">
                    <div class="label">Risk Level</div>
                    <div class="value" style="font-size:1.4rem;">
                        <span class="risk-badge {badge_cls}">{label}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Model</div>
                    <div class="value" style="font-size:1.2rem;">{model_choice}</div>
                </div>""", unsafe_allow_html=True)

            if label == "HIGH":
                st.error(f"⚠️  Risk score exceeds your {user_threshold:.2f} threshold. Consider immediate clinical review.")
            elif label == "MODERATE":
                st.warning("⚡  Moderate risk. Recommend increased monitoring frequency.")
            else:
                st.success("✓  Low risk. Continue standard monitoring protocol.")

            # Breakdown table
            st.markdown("---")
            st.markdown("#### Input Summary (last 6 hours)")
            summary_df = pd.DataFrame(window_rows)[VITAL_COLS]
            summary_df.index = [f"Hour {i+1}" for i in range(len(summary_df))]
            st.dataframe(
                summary_df.style.format("{:.1f}").background_gradient(cmap="RdYlGn_r"),
                use_container_width=True,
            )

            # Delta from first to last hour
            st.markdown("#### Trend (first → last hour)")
            first = summary_df.iloc[0]
            last  = summary_df.iloc[-1]
            delta = last - first
            trend_df = pd.DataFrame({
                "First Hour": first,
                "Last Hour":  last,
                "Δ Change":   delta,
            })
            st.dataframe(trend_df.style.format("{:.2f}"), use_container_width=True)