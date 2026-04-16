"""
train_lstm.py  —  Train and save the Keras LSTM sepsis model
Run once:  python train_lstm.py --data /path/to/Dataset.csv
Saves:     models/lstm_model.h5  +  models/lstm_scaler.joblib  +  models/lstm_meta.json
"""

import argparse
import json
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import load_and_clean_data, create_sequences, fit_scaler, apply_scaler, WINDOW

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data",   required=True)
parser.add_argument("--out",    default="models")
parser.add_argument("--epochs", type=int, default=15)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------------------
# 1-4. Load, Split by Patient, Clean, Sequence, and Scale
# ---------------------------------------------------------------------------
df, feature_cols, id_col = load_and_clean_data(args.data)

# Split by Patient ID
unique_pids = df[id_col].unique()
train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=42)

df_train = df[df[id_col].isin(train_pids)]
df_test  = df[df[id_col].isin(test_pids)]

X_train_raw, y_train = create_sequences(df_train, feature_cols, id_col)
X_test_raw,  y_test  = create_sequences(df_test,  feature_cols, id_col)

n_samples, seq_len, n_feat = X_train_raw.shape

# Scale
X_train_2d, scaler = fit_scaler(X_train_raw.reshape(-1, n_feat))
X_test_2d          = apply_scaler(X_test_raw.reshape(-1, n_feat), scaler)

X_train_s = X_train_2d.reshape(-1, seq_len, n_feat)
X_test_s  = X_test_2d.reshape(-1, seq_len, n_feat)

# ---------------------------------------------------------------------------
# 5. Build model (This is what went missing!)
# ---------------------------------------------------------------------------
model = Sequential([
    Input(shape=(seq_len, n_feat)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------------------------------------------------------
# 6. Train (Optimized Batch Size + Class Weights)
# ---------------------------------------------------------------------------
# Calculate class weights for highly imbalanced data
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
total = len(y_train)

weight_for_0 = (1 / neg_count) * (total / 2.0)
weight_for_1 = (1 / pos_count) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

model.fit(
    X_train_s, y_train,
    epochs=args.epochs,
    batch_size=512, 
    validation_split=0.1, 
    callbacks=[early_stop],
    class_weight=class_weight
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 7. Evaluate (Expert Level Metrics)
# ---------------------------------------------------------------------------
from sklearn.metrics import precision_score, f1_score, average_precision_score, roc_auc_score, recall_score

y_prob = model.predict(X_test_s).ravel()

# Medical Threshold Tuning
threshold = 0.3 
y_pred = (y_prob >= threshold).astype(int)

print(f"\n--- LSTM Final Results (Threshold: {threshold}) ---")
print(f"AUPRC    : {average_precision_score(y_test, y_prob):.4f}")
print(f"AUROC    : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
model.save(os.path.join(args.out, "lstm_model.keras"))
joblib.dump(scaler, os.path.join(args.out, "lstm_scaler.joblib"))

meta = {"feature_cols": feature_cols, "seq_len": seq_len, "n_feat": n_feat}
with open(os.path.join(args.out, "lstm_meta.json"), "w") as f:
    json.dump(meta, f)

print(f"\n✓ Saved to {args.out}/")