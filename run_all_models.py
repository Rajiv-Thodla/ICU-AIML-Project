import subprocess
import time
import os
import sys
import re

DATA_PATH = "data/Dataset.csv"
OUT_DIR = "models"

commands = [
    [sys.executable, "train_gru.py", "--data", DATA_PATH, "--out", OUT_DIR],
    [sys.executable, "train_lstm.py", "--data", DATA_PATH, "--out", OUT_DIR],
    [sys.executable, "train_tcn.py", "--data", DATA_PATH, "--out", OUT_DIR],
    [sys.executable, "train_xgb.py", "--data", DATA_PATH, "--out", OUT_DIR],
    [sys.executable, "train_arima.py", "--data", DATA_PATH, "--out", OUT_DIR]
]

def extract_metric(text, metric_name):
    """Scans the console output and extracts the metric score"""
    pattern = rf"{metric_name}\s*:\s*([0-9\.]+)"
    matches = re.findall(pattern, text)
    if matches:
        return float(matches[-1]) # Takes the last printed instance
    return None

def run_training():
    print("🚀 Starting Master Training Pipeline...")
    start_all = time.time()

    results = {}
    
    # THE FIX: Force all child processes to output in UTF-8
    custom_env = os.environ.copy()
    custom_env["PYTHONIOENCODING"] = "utf-8"

    for cmd in commands:
        script_name = cmd[1]
        model_name = script_name.replace("train_", "").replace(".py", "").upper()
        if model_name == "XGB": model_name = "XGBoost"

        print(f"\n" + "="*50)
        print(f"⌛ NOW TRAINING: {script_name}")
        print("="*50)

        start_script = time.time()

        try:
            # We pass the custom_env so the scripts don't choke on emojis
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                encoding="utf-8",
                env=custom_env 
            )
            
            output_text = ""
            for line in process.stdout:
                print(line, end="")
                output_text += line
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            duration = (time.time() - start_script) / 60
            print(f"\n✅ FINISHED: {script_name} in {duration:.2f} minutes")

            # Extract metrics from the captured terminal output
            auprc = extract_metric(output_text, "AUPRC")
            auroc = extract_metric(output_text, "AUROC")
            recall = extract_metric(output_text, "Recall")
            precision = extract_metric(output_text, "Precision")
            f1 = extract_metric(output_text, "F1 Score")

            results[model_name] = {
                "AUPRC": auprc,
                "AUROC": auroc,
                "Recall": recall,
                "Precision": precision,
                "F1 Score": f1
            }

        except subprocess.CalledProcessError as e:
            print(f"\n❌ ERROR in {script_name}: {e}")
            print("Stopping pipeline to prevent cascading errors.")
            return

    total_duration = (time.time() - start_all) / 60

    print(f"\n" + "!"*50)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"Total time taken: {total_duration:.2f} minutes")
    print("!"*50)

    # ---------------------------------------------------------
    # PRINT FINAL COMPARISON TABLES
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(" 📊 FINAL MODEL COMPARISON METRICS")
    print("="*65)
    
    header = f"{'Model':<12} | {'AUPRC':<8} | {'AUROC':<8} | {'Recall':<8} | {'Precision':<9} | {'F1 Score':<8}"
    print(header)
    print("-" * len(header))
    
    for model, mets in results.items():
        # Handles missing values gracefully
        def fmt(v): return f"{v:.4f}" if v is not None else "N/A  "
        row = f"{model:<12} | {fmt(mets['AUPRC']):<8} | {fmt(mets['AUROC']):<8} | {fmt(mets['Recall']):<8} | {fmt(mets['Precision']):<9} | {fmt(mets['F1 Score']):<8}"
        print(row)

    print("\n" + "="*65)
    print(" 🏆 BEST MODEL PER METRIC")
    print("="*65)
    
    metrics_list = ["AUPRC", "AUROC", "Recall", "Precision", "F1 Score"]
    best_models = {}
    
    for m in metrics_list:
        best_val = -1
        best_mod = "None"
        for mod, mets in results.items():
            val = mets.get(m)
            if val is not None and val > best_val:
                best_val = val
                best_mod = mod
        best_models[m] = (best_mod, best_val)

    for m in metrics_list:
        mod, val = best_models[m]
        val_str = f"{val:.4f}" if val != -1 else "N/A"
        print(f" {m:<12} : {mod:<12} ({val_str})")
        
    print("="*65 + "\n")

if __name__ == "__main__":
    run_training()