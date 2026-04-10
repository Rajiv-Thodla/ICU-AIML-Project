import argparse
import json
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

from preprocess import load_and_clean_data, create_sequences, fit_scaler, apply_scaler

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data",   required=True)
parser.add_argument("--out",    default="models")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch",  type=int, default=512)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# 1-4. Load, Split by Patient, Clean, Sequence, and Scale
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1-4. Load, Split by Patient, Clean, Sequence, and Scale
# ---------------------------------------------------------------------------
df, feature_cols, id_col = load_and_clean_data(args.data)

# Split by Patient ID
unique_pids = df[id_col].unique()
train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=42)

df_train = df[df[id_col].isin(train_pids)]
df_test  = df[df[id_col].isin(test_pids)]

# Create Sequences
X_train_raw, y_train = create_sequences(df_train, feature_cols, id_col)
X_test_raw,  y_test  = create_sequences(df_test,  feature_cols, id_col)

print(f"Train distribution: {np.bincount(y_train.astype(int))}")
print(f"Test distribution:  {np.bincount(y_test.astype(int))}")

n_samples, seq_len, n_feat = X_train_raw.shape

# Scale the data (Fit on train only to prevent leakage!)
X_train_2d, scaler = fit_scaler(X_train_raw.reshape(-1, n_feat))
X_test_2d          = apply_scaler(X_test_raw.reshape(-1, n_feat), scaler)

# Reshape back to (Samples, Time, Features) and name them correctly for the loader
X_train_s = X_train_2d.reshape(-1, seq_len, n_feat).astype(np.float32)
X_test_s  = X_test_2d.reshape(-1, seq_len, n_feat).astype(np.float32)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. DataLoaders (Optimized for speed)
# ---------------------------------------------------------------------------
def make_loader(X_data, y_data, batch_size=64, shuffle=False):
    # from_numpy avoids unnecessary memory copying
    ds = TensorDataset(
        torch.from_numpy(X_data),
        torch.from_numpy(y_data).float()
    )
    # pin_memory speeds up CPU to GPU data transfer
    kwargs = {'pin_memory': True, 'num_workers': 0} if torch.cuda.is_available() else {}
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)

train_loader = make_loader(X_train_s, y_train, batch_size=args.batch, shuffle=True)
test_loader  = make_loader(X_test_s,  y_test,  batch_size=args.batch)

# ---------------------------------------------------------------------------
# 6. Model (Optimized Loss Function)
# ---------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        # REMOVED Sigmoid() here. It is handled safely by BCEWithLogitsLoss.

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]) # Output raw logits

model = GRUModel(input_size=n_feat).to(device)

# Class-weighted loss correctly applied
pos_weight = torch.tensor(
    [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
    dtype=torch.float32,
).to(device)

# BCEWithLogitsLoss is numerically stabler and accepts pos_weight
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

# ---------------------------------------------------------------------------
# 7. Train (Added gradient clipping & averaged loss)
# ---------------------------------------------------------------------------
best_loss = float("inf")
best_state = None

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True).unsqueeze(1)
        
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients in GRU
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader) # Average loss

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True).unsqueeze(1)
            val_loss += criterion(model(Xb), yb).item()
            
    val_loss /= len(test_loader) # Average loss
    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss  = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

model.load_state_dict(best_state)

# ---------------------------------------------------------------------------
## ---------------------------------------------------------------------------
# 8. Evaluate (Expert Level Metrics)
# ---------------------------------------------------------------------------
from sklearn.metrics import precision_score, f1_score, average_precision_score, roc_auc_score, recall_score

model.eval()
y_true_all, y_prob_all = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        logits = model(Xb.to(device))
        probs = torch.sigmoid(logits) 
        y_true_all.extend(yb.numpy())
        y_prob_all.extend(probs.cpu().numpy())

y_true_all = np.array(y_true_all)
y_prob_all = np.array(y_prob_all).ravel()

# Medical Threshold Tuning
threshold = 0.3 
y_pred_all = (y_prob_all >= threshold).astype(int)

print(f"\n--- GRU Final Results (Threshold: {threshold}) ---")
print(f"AUPRC    : {average_precision_score(y_true_all, y_prob_all):.4f}")
print(f"AUROC    : {roc_auc_score(y_true_all, y_prob_all):.4f}")
print(f"Recall   : {recall_score(y_true_all, y_pred_all):.4f}")
print(f"Precision: {precision_score(y_true_all, y_pred_all):.4f}")
print(f"F1 Score : {f1_score(y_true_all, y_pred_all):.4f}")
# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(args.out, "gru_model.pt"))
joblib.dump(scaler, os.path.join(args.out, "gru_scaler.joblib"))

meta = {
    "feature_cols": feature_cols,
    "seq_len": seq_len,
    "n_feat": n_feat,
    "hidden_size": 64,
    "num_layers": 2,
}
with open(os.path.join(args.out, "gru_meta.json"), "w") as f:
    json.dump(meta, f)

print(f"\n✓ Saved to {args.out}/")