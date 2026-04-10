import pandas as pd
import numpy as np
import random

# Ensure reproducible random patients
np.random.seed(42)
random.seed(42)

def generate_patient(patient_id, is_sepsis):
    # Length of ICU stay (between 24 and 48 hours)
    stay_length = random.randint(24, 48)
    
    # Static info
    age = random.randint(30, 85)
    gender = random.choice([0, 1])
    
    # Base normal vitals
    hr_base = random.uniform(65, 85)
    o2_base = random.uniform(95, 99)
    temp_base = random.uniform(36.5, 37.3)
    sbp_base = random.uniform(110, 130)
    map_base = random.uniform(75, 90)
    resp_base = random.uniform(14, 18)
    
    rows = []
    
    # Sepsis crash timing (if applicable)
    crash_hour = stay_length - random.randint(6, 12) if is_sepsis else 999
    
    for hour in range(stay_length):
        # Add slight natural fluctuation (noise)
        hr = hr_base + np.random.normal(0, 2)
        o2 = o2_base + np.random.normal(0, 0.5)
        temp = temp_base + np.random.normal(0, 0.2)
        sbp = sbp_base + np.random.normal(0, 3)
        map_val = map_base + np.random.normal(0, 2)
        resp = resp_base + np.random.normal(0, 1)
        
        sepsis_label = 0
        
        # Simulate physiological deterioration if Sepsis is coming
        if is_sepsis and hour >= crash_hour - 6:
            # HR climbs, BP drops, Temp spikes, Resp climbs
            hr += (hour - (crash_hour - 6)) * 4   
            sbp -= (hour - (crash_hour - 6)) * 3  
            map_val -= (hour - (crash_hour - 6)) * 2 
            temp += (hour - (crash_hour - 6)) * 0.3
            resp += (hour - (crash_hour - 6)) * 1.5
            
        if is_sepsis and hour >= crash_hour:
            sepsis_label = 1
            
        # Clip to ensure biological realism
        rows.append({
            "Patient_ID": patient_id,
            "Hour": hour,
            "HR": max(30, min(200, hr)),
            "O2Sat": max(50, min(100, o2)),
            "Temp": max(32, min(42, temp)),
            "SBP": max(50, min(250, sbp)),
            "MAP": max(30, min(150, map_val)),
            "Resp": max(8, min(50, resp)),
            "Age": age,
            "Gender": gender,
            "SepsisLabel": sepsis_label
        })
        
    return rows

# Generate 10 patients (5 healthy, 5 crashing)
all_data = []
for pid in range(1, 6):
    all_data.extend(generate_patient(pid, is_sepsis=False))
for pid in range(6, 11):
    all_data.extend(generate_patient(pid, is_sepsis=True))

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv("test_patients.csv", index=False)
print("✓ Generated test_patients.csv with 10 simulated patients!")
print("Patients 1-5 are healthy. Patients 6-10 will crash from Sepsis.")