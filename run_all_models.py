import subprocess
import time
import os

# Define the dataset path and output folder
DATA_PATH = "data/Dataset.csv"
OUT_DIR = "models"

# List of training commands to run
commands = [
    ["python", "train_gru.py", "--data", DATA_PATH, "--out", OUT_DIR],
    ["python", "train_lstm.py", "--data", DATA_PATH, "--out", OUT_DIR],
    ["python", "train_xgb.py", "--data", DATA_PATH, "--out", OUT_DIR]
]

def run_training():
    print("🚀 Starting Master Training Pipeline...")
    start_all = time.time()
    
    for cmd in commands:
        script_name = cmd[1]
        print(f"\n" + "="*50)
        print(f"⌛ NOW TRAINING: {script_name}")
        print("="*50)
        
        start_script = time.time()
        
        # This runs the command and waits for it to finish
        try:
            subprocess.run(cmd, check=True)
            end_script = time.time()
            duration = (end_script - start_script) / 60
            print(f"\n✅ FINISHED: {script_name} in {duration:.2f} minutes")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ ERROR in {script_name}: {e}")
            print("Stopping pipeline to prevent cascading errors.")
            return

    end_all = time.time()
    total_duration = (end_all - start_all) / 60
    print(f"\n" + "!"*50)
    print(f"🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"Total time taken: {total_duration:.2f} minutes")
    print("Check your 'models/' folder for the results.")
    print("!"*50)

if __name__ == "__main__":
    run_training()