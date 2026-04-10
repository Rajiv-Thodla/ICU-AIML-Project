import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Constants
VITAL_COLS   = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
STATIC_COLS  = ["Age", "Gender"]
LABEL_COL    = "SepsisLabel"
NON_FEATURE_COLS = {"Patient_ID", "PatientID", "Hour", "SepsisLabel"}

VITAL_CLIP_RANGES = {
    "HR": (20, 300), "O2Sat": (50, 100), "Temp": (25, 45),
    "SBP": (40, 280), "MAP": (20, 200), "DBP": (20, 200),
    "Resp": (4, 60),
}

WINDOW = 6
PRED_WINDOW = 3

def load_and_clean_data(path, nrows=None, missing_threshold=0.90):
    df = pd.read_csv(path, nrows=nrows)
    df.columns = df.columns.str.strip()
    if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
    
    if "Patient_ID" in df.columns: id_col = "Patient_ID"
    elif "PatientID" in df.columns: id_col = "PatientID"
    else:
        df["Patient_ID"] = (df["Hour"] == 0).cumsum()
        id_col = "Patient_ID"

    df = df.sort_values([id_col, "Hour"]).reset_index(drop=True)

    # Clean & Fill
    missing_frac = df.isnull().mean()
    drop_cols = [c for c in missing_frac[missing_frac > missing_threshold].index if c not in NON_FEATURE_COLS]
    df = df.drop(columns=drop_cols)
    
    for col in [c for c in df.columns if c not in NON_FEATURE_COLS and df[c].isnull().any()]:
        df[f"{col}_missing"] = df[col].isnull().astype(np.int8)

    value_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and not c.endswith("_missing")]
    df[value_cols] = df.groupby(id_col)[value_cols].transform(lambda g: g.ffill().bfill())
    df[value_cols] = df[value_cols].fillna(df[value_cols].median())

    for col, (lo, hi) in VITAL_CLIP_RANGES.items():
        if col in df.columns: df[col] = df[col].clip(lo, hi)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols, id_col

def fit_scaler(X_2d):
    scaler = StandardScaler()
    return scaler.fit_transform(X_2d), scaler

def apply_scaler(X_2d, scaler):
    return scaler.transform(X_2d)

def create_sequences(df, feature_cols, id_col):
    X, y = [], []
    for pid, group in df.groupby(id_col):
        group = group.sort_values("Hour")
        if len(group) < WINDOW + PRED_WINDOW: continue
        feat_vals = group[feature_cols].values
        labels = group[LABEL_COL].values
        for i in range(WINDOW, len(group) - PRED_WINDOW):
            if labels[i - WINDOW : i].max() == 1: break
            X.append(feat_vals[i - WINDOW : i])
            y.append(int(labels[i : i + PRED_WINDOW].max()))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def create_xgb_features(df, id_col):
    agg_cols = [c for c in VITAL_COLS if c in df.columns]
    rows = []
    grouped = df.groupby(id_col)
    for pid, group in grouped:
        group = group.sort_values("Hour")
        if len(group) < WINDOW + PRED_WINDOW: continue
        feat_vals, labels = group[agg_cols].values, group[LABEL_COL].values
        for i in range(WINDOW, len(group) - PRED_WINDOW):
            if labels[i - WINDOW : i].max() == 1: break
            past_window = feat_vals[i - WINDOW : i]
            feat = {}
            for idx, col in enumerate(agg_cols):
                col_data = past_window[:, idx]
                # --- All 6 Stats to perfectly match the App Helper ---
                feat[f"{col}_mean"]  = np.mean(col_data)
                feat[f"{col}_max"]   = np.max(col_data)
                feat[f"{col}_min"]   = np.min(col_data)  # <--- ADDED THIS LINE
                feat[f"{col}_std"]   = np.std(col_data)
                feat[f"{col}_trend"] = col_data[-1] - col_data[0]
                feat[f"{col}_last"]  = col_data[-1]
            feat["label"] = int(labels[i : i + PRED_WINDOW].max())
            rows.append(feat)
    return pd.DataFrame(rows)
def build_xgb_manual_features(hourly_rows):
    df = pd.DataFrame(hourly_rows)
    feat = {}
    for col in VITAL_COLS:
        col_data = df[col] if col in df.columns else pd.Series([0]*WINDOW)
        feat[f"{col}_mean"] = col_data.mean()
        feat[f"{col}_max"]  = col_data.max()
        feat[f"{col}_min"]  = col_data.min()
        feat[f"{col}_std"]  = col_data.std() if len(col_data) > 1 else 0.0
        feat[f"{col}_trend"] = col_data.iloc[-1] - col_data.iloc[0]
        feat[f"{col}_last"]  = col_data.iloc[-1]
    return feat

def build_manual_sequence(hourly_rows, feature_cols, scaler):
    df = pd.DataFrame(hourly_rows)
    for col in feature_cols:
        if col.endswith("_missing"):
            base = col[:-8]
            df[col] = df[base].isnull().astype(int) if base in df.columns else 0
    df = df.ffill().bfill().fillna(0)
    for col, (lo, hi) in VITAL_CLIP_RANGES.items():
        if col in df.columns: df[col] = df[col].clip(lo, hi)
    for col in feature_cols:
        if col not in df.columns: df[col] = 0
    X_2d = apply_scaler(df[feature_cols].values.astype(np.float32), scaler)
    return X_2d[np.newaxis, :, :]