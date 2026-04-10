# ICU Patient Deterioration Monitor

Predicts sepsis onset risk from ICU time-series vitals using three interchangeable models:
**XGBoost**, **LSTM** (Keras/TensorFlow), and **GRU** (PyTorch).

---

## Project Structure

```
project/
├── preprocess.py        ← Single unified data cleaning pipeline (shared by all)
├── train_xgb.py         ← Train and save XGBoost model
├── train_lstm.py        ← Train and save LSTM model
├── train_gru.py         ← Train and save GRU model
├── app.py               ← Streamlit frontend
├── requirements.txt
└── models/              ← Created automatically by training scripts
    ├── xgb_model.joblib
    ├── xgb_scaler.joblib
    ├── xgb_features.json
    ├── lstm_model.h5
    ├── lstm_scaler.joblib
    ├── lstm_meta.json
    ├── gru_model.pt
    ├── gru_scaler.joblib
    └── gru_meta.json
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Train the models

Run each script once against your dataset. You can train any subset — the app
gracefully hides models whose files don't exist.

```bash
python train_xgb.py  --data /path/to/Dataset.csv
python train_lstm.py --data /path/to/Dataset.csv
python train_gru.py  --data /path/to/Dataset.csv
```

Optional flags:
```bash
--out models       # output directory (default: models/)
--epochs 20        # number of training epochs (LSTM and GRU only)
```

---

## Step 2 — Launch the app

```bash
streamlit run app.py
```

---

## App Features

### Model selection
Switch between XGBoost, LSTM, and GRU from the sidebar at any time.
Only models with saved files in `models/` are shown.

### CSV Upload mode
- Upload a CSV matching the training schema
- Select any patient by ID
- View vitals chart over time
- Get a prediction + sliding-window risk score chart across all hours

### Manual Entry mode
- Choose 6–24 hours of data to enter
- Fill in each vital per hour using number inputs
- Get risk score, risk level badge, trend table, and delta summary

---

## Data Schema

The CSV should have these columns (extras are handled automatically):

| Column       | Description                        |
|--------------|------------------------------------|
| `Patient_ID` | Patient identifier                 |
| `Hour`       | Hour of ICU stay (integer)         |
| `HR`         | Heart rate (bpm)                   |
| `O2Sat`      | Oxygen saturation (%)              |
| `Temp`       | Temperature (°C)                   |
| `SBP`        | Systolic blood pressure (mmHg)     |
| `MAP`        | Mean arterial pressure (mmHg)      |
| `Resp`       | Respiratory rate (breaths/min)     |
| `Age`        | Patient age                        |
| `Gender`     | 1 = Male, 0 = Female               |
| `SepsisLabel`| 1 = sepsis onset, 0 = no sepsis    |

---

## Cleaning Pipeline (preprocess.py)

All three models share a single pipeline applied in this order:

1. **Drop columns >90% missing** — removes noise-only features
2. **Add `_missing` indicator columns** — preserves "was this test ordered?" as a clinical signal
3. **ffill → bfill → global median** — fills remaining gaps per patient, in priority order
4. **Clip physiological outliers** — removes sensor errors using validated vital ranges
5. **StandardScaler** — fitted on training split only (scaler saved alongside model)

---

## Disclaimer

For clinical research and educational purposes only.  
Not validated for diagnostic or treatment decisions.
