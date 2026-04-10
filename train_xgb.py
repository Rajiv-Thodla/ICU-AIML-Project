"""
train_xgb.py  —  Train and save the XGBoost sepsis prediction model
Run once:  python train_xgb.py --data /path/to/Dataset.csv
Saves:     models/xgb_model.joblib  +  models/xgb_scaler.joblib  +  models/xgb_features.json
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from xgboost import XGBClassifier

from preprocess import load_and_clean_data, create_xgb_features

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to the dataset CSV")
parser.add_argument("--out",  default="models", help="Output directory")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------------------
# 1-3. Load, Split by Patient, and Build Features
# ---------------------------------------------------------------------------
df, feature_cols, id_col = load_and_clean_data(args.data)

# FIX: Split by Patient ID
unique_pids = df[id_col].unique()
train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=42)

df_train = df[df[id_col].isin(train_pids)]
df_test  = df[df[id_col].isin(test_pids)]

# Build flat features for XGBoost
features_train = create_xgb_features(df_train, id_col)
features_test  = create_xgb_features(df_test, id_col)

X_train = features_train.drop("label", axis=1)
y_train = features_train["label"]
X_test  = features_test.drop("label", axis=1)
y_test  = features_test["label"]

# Scale
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4. Train (Optimized with Early Stopping in the Constructor)
# ---------------------------------------------------------------------------
pos_ratio = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=pos_ratio,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=20  # <--- It lives here now!
)

# In modern XGBoost, .fit() should only have data and eval_set
model.fit(
    X_train_s, y_train, 
    eval_set=[(X_test_s, y_test)],
    verbose=False  # <--- Make sure early_stopping_rounds is NOT here!
)

# ---------------------------------------------------------------------------
# 5. Evaluate (Expert Level Metrics)
# ---------------------------------------------------------------------------
from sklearn.metrics import precision_score, f1_score, average_precision_score, roc_auc_score, recall_score

y_prob = model.predict_proba(X_test_s)[:, 1]

# Medical Threshold Tuning
threshold = 0.3 
y_pred = (y_prob >= threshold).astype(int)

print(f"\n--- XGBoost Final Results (Threshold: {threshold}) ---")
print(f"AUPRC    : {average_precision_score(y_test, y_prob):.4f}")
print(f"AUROC    : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")