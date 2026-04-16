"""
train_arima.py  --  Train and save the ARIMA-based sepsis prediction model
Run once:  python train_arima.py --data Dataset.txt
Saves:     models/arima_model.joblib  +  models/arima_scaler.joblib  +  models/arima_meta.json

Approach (Hybrid ARIMA + Logistic Regression):
  For each 6-hour patient window, compute ARIMA-equivalent time-series
  features for every vital sign — AR(1) autocorrelation coefficient,
  moving-average residuals, forecast via linear extrapolation, residual
  variance, and trend statistics.  A calibrated Logistic Regression
  classifier then maps these rich temporal features to a sepsis-onset
  probability, producing a survival/risk score compatible with the
  existing ICU Deterioration Monitor app.
"""

import argparse, json, os, sys, warnings, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, recall_score,
    precision_score, f1_score, average_precision_score,
)

warnings.filterwarnings("ignore")

def log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Config (matches preprocess.py constants)
# ---------------------------------------------------------------------------
VITAL_COLS  = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
LABEL_COL   = "SepsisLabel"
WINDOW      = 6
PRED_WINDOW = 3

VITAL_CLIP = {
    "HR": (20, 300), "O2Sat": (50, 100), "Temp": (25, 45),
    "SBP": (40, 280), "MAP": (20, 200), "Resp": (4, 60),
}

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to dataset CSV/TXT")
parser.add_argument("--out",  default="models", help="Output directory")
parser.add_argument("--nrows", type=int, default=200000,
                    help="Max rows to load (default 200k)")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
log("[ ] Loading data...")
t0 = time.time()
df = pd.read_csv(args.data, nrows=args.nrows)
df.columns = df.columns.str.strip()
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "Patient_ID" in df.columns:
    id_col = "Patient_ID"
elif "PatientID" in df.columns:
    id_col = "PatientID"
else:
    df["Patient_ID"] = (df["Hour"] == 0).cumsum()
    id_col = "Patient_ID"

df = df.sort_values([id_col, "Hour"]).reset_index(drop=True)

keep_cols = [id_col, "Hour", LABEL_COL] + [c for c in VITAL_COLS if c in df.columns]
df = df[keep_cols].copy()
agg_cols = [c for c in VITAL_COLS if c in df.columns]

# Fill missing vitals
for col in agg_cols:
    df[col] = df.groupby(id_col)[col].transform(lambda g: g.ffill().bfill())
    df[col] = df[col].fillna(df[col].median())

for col, (lo, hi) in VITAL_CLIP.items():
    if col in df.columns:
        df[col] = df[col].clip(lo, hi)

log(f"[+] Loaded {len(df):,} rows, {df[id_col].nunique():,} patients in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# 2. Fast ARIMA-Equivalent Feature Extraction
#    Computes the same features as ARIMA(1,0,1) but analytically:
#    - AR(1) coeff via lag-1 autocorrelation
#    - MA(1) residual smoothing
#    - Forecast via AR(1) extrapolation
#    - Residual variance, trend, momentum
# ---------------------------------------------------------------------------
def ar1_coeff(series):
    """Compute AR(1) coefficient: lag-1 autocorrelation (Yule-Walker)."""
    if len(series) < 3:
        return 0.0
    mean = np.mean(series)
    centered = series - mean
    c0 = np.sum(centered ** 2)
    if c0 == 0:
        return 0.0
    c1 = np.sum(centered[:-1] * centered[1:])
    return c1 / c0


def extract_arima_features(window_data, vital_cols):
    """
    Extract ARIMA-equivalent features from a (WINDOW x n_vitals) array.
    13 features per vital x 6 vitals = 78 features total.
    """
    feat = {}
    t = np.arange(len(window_data), dtype=np.float64)

    for idx, col in enumerate(vital_cols):
        s = window_data[:, idx].astype(np.float64)
        n = len(s)

        # --- Descriptive statistics ---
        feat[f"{col}_mean"]  = np.mean(s)
        feat[f"{col}_std"]   = np.std(s)
        feat[f"{col}_min"]   = np.min(s)
        feat[f"{col}_max"]   = np.max(s)
        feat[f"{col}_last"]  = s[-1]
        feat[f"{col}_range"] = np.ptp(s)

        # --- AR(1) coefficient (momentum/persistence) ---
        phi = ar1_coeff(s)
        feat[f"{col}_ar1"] = phi

        # --- AR(1) residuals and MA(1) ---
        # Residuals: e(t) = x(t) - phi * x(t-1)
        residuals = s[1:] - phi * s[:-1]
        feat[f"{col}_resid_var"] = np.var(residuals) if len(residuals) > 0 else 0.0

        # MA(1) coefficient approximation (lag-1 autocorrelation of residuals)
        theta = ar1_coeff(residuals) if len(residuals) >= 3 else 0.0
        feat[f"{col}_ma1"] = theta

        # --- Linear trend (slope via least squares) ---
        if np.std(s) > 0:
            slope = np.polyfit(t, s, 1)[0]
        else:
            slope = 0.0
        feat[f"{col}_trend"] = slope

        # --- 1-step forecast via AR(1): x_hat(T+1) = mean + phi*(x(T) - mean) ---
        mu = np.mean(s)
        forecast = mu + phi * (s[-1] - mu)
        feat[f"{col}_forecast1"] = forecast
        feat[f"{col}_forecast_delta"] = forecast - s[-1]

    return feat


def build_dataset(df, id_col, vital_cols):
    """Slide windows over all patients and extract features."""
    rows = []
    n_patients = df[id_col].nunique()
    done = 0

    for pid, group in df.groupby(id_col):
        group = group.sort_values("Hour")
        if len(group) < WINDOW + PRED_WINDOW:
            continue

        vital_data = group[vital_cols].values
        labels = group[LABEL_COL].values

        for i in range(WINDOW, len(group) - PRED_WINDOW):
            if labels[i - WINDOW : i].max() == 1:
                break
            window = vital_data[i - WINDOW : i]
            feat = extract_arima_features(window, vital_cols)
            feat["label"] = int(labels[i : i + PRED_WINDOW].max())
            rows.append(feat)

        done += 1
        if done % 1000 == 0:
            log(f"    {done:,}/{n_patients:,} patients...")

    log(f"    Done: {len(rows):,} windows from {done:,} patients")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Split by Patient
# ---------------------------------------------------------------------------
unique_pids = df[id_col].unique()
train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=42)

df_train = df[df[id_col].isin(train_pids)]
df_test  = df[df[id_col].isin(test_pids)]

log(f"\n[+] Train: {len(train_pids):,} patients  |  Test: {len(test_pids):,} patients")

# ---------------------------------------------------------------------------
# 4. Build Feature Matrices
# ---------------------------------------------------------------------------
log("\n[ ] Extracting ARIMA features (TRAIN)...")
t1 = time.time()
features_train = build_dataset(df_train, id_col, agg_cols)
log(f"    Completed in {time.time()-t1:.1f}s")

log("\n[ ] Extracting ARIMA features (TEST)...")
t2 = time.time()
features_test = build_dataset(df_test, id_col, agg_cols)
log(f"    Completed in {time.time()-t2:.1f}s")

X_train = features_train.drop("label", axis=1)
y_train = features_train["label"]
X_test  = features_test.drop("label", axis=1)
y_test  = features_test["label"]

arima_feature_cols = list(X_train.columns)

log(f"\n    Train: {len(X_train):,} samples (sepsis={int(y_train.sum()):,})")
log(f"    Test:  {len(X_test):,} samples (sepsis={int(y_test.sum()):,})")
log(f"    Features: {len(arima_feature_cols)}")

# ---------------------------------------------------------------------------
# 5. Scale
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

# ---------------------------------------------------------------------------
# 6. Train Classifier
# ---------------------------------------------------------------------------
log("\n[ ] Training Logistic Regression on ARIMA features...")
pos_ratio = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

model = LogisticRegression(
    C=1.0,
    class_weight={0: 1.0, 1: pos_ratio},
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
)
model.fit(X_train_s, y_train)
log("[+] Classifier trained.")

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
y_prob = model.predict_proba(X_test_s)[:, 1]
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)

log(f"\n{'='*50}")
log(f"  ARIMA Model Results (Threshold: {threshold})")
log(f"{'='*50}")
log(f"  AUPRC     : {average_precision_score(y_test, y_prob):.4f}")
log(f"  AUROC     : {roc_auc_score(y_test, y_prob):.4f}")
log(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
log(f"  Precision : {precision_score(y_test, y_pred):.4f}")
log(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
log(f"{'='*50}")

# ---------------------------------------------------------------------------
# 8. Save Artifacts
# ---------------------------------------------------------------------------
joblib.dump(model,  os.path.join(args.out, "arima_model.joblib"))
joblib.dump(scaler, os.path.join(args.out, "arima_scaler.joblib"))

meta = {
    "vital_cols": agg_cols,
    "arima_feature_cols": arima_feature_cols,
    "window": WINDOW,
    "pred_window": PRED_WINDOW,
    "arima_order": [1, 0, 1],
    "threshold": threshold,
    "classifier": "LogisticRegression",
}
with open(os.path.join(args.out, "arima_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

log(f"\n[+] Saved ARIMA artifacts to {args.out}/")
log(f"    -> arima_model.joblib")
log(f"    -> arima_scaler.joblib")
log(f"    -> arima_meta.json")
log(f"\nTotal time: {time.time()-t0:.1f}s")
