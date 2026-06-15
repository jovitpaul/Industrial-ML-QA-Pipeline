import numpy as np
import pandas as pd

# Set random seed for absolute reproducibility
np.random.seed(42)

n_samples = 5500

print("--- Starting Phase 1: Generating Precision Factory Data ---")

# 1. Generate Independent Base Features
batch_ids = [f"BATCH_{str(i).zfill(4)}" for i in range(1, n_samples + 1)]
suppliers = np.random.choice(["Supplier_A", "Supplier_B", "Supplier_C"], size=n_samples, p=[0.50, 0.20, 0.30])
supervisors = np.random.choice(["Sir_Manny", "Maam_Joy", "Sir_Jun"], size=n_samples, p=[0.34, 0.33, 0.33])

ambient_humidity = np.random.uniform(55.0, 92.0, size=n_samples)
flour_moisture_base = np.random.normal(13.5, 0.6, size=n_samples) # Tightened standard deviation
mix_time_mins = np.random.randint(10, 21, size=n_samples)
conveyor_speed_rpm = np.random.randint(40, 56, size=n_samples)
oven_zone3_temp = np.random.normal(180.0, 1.0, size=n_samples) # Tightened standard deviation

# 2. Inject Industrial Anomalies 
defect_labels = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    # Anomaly A: Supplier B raw material issues
    if suppliers[i] == "Supplier_B":
        flour_moisture_base[i] += 1.0
        
    # Anomaly B: The Bulacan Power Grid Sag (Extremely rare: 0.4% frequency)
    if np.random.rand() < 0.004:
        oven_zone3_temp[i] -= np.random.uniform(8.0, 12.0)
        
    # Anomaly C: Monsoon/Habagat operational delays (Only hits at extreme peak humidity)
    if ambient_humidity[i] > 89.0:
        mix_time_mins[i] += np.random.randint(2, 4)

    # 3. DETERMINISTIC DEFECT RULES (Strict constraints)
    
    # Baseline Noise: The acceptable 0.5% normal factory defect rate
    if np.random.rand() < 0.005:
        defect_labels[i] = 1
        continue # Skip other rules to prevent double-counting
        
    # The Crisis Addition: Severe Power Sag destroys the batch
    if oven_zone3_temp[i] < 173.0:
        defect_labels[i] = 1
        
    # The Crisis Addition: Extreme moisture + Extreme mix time destroys the batch
    elif flour_moisture_base[i] > 15.2 and mix_time_mins[i] > 19:
        defect_labels[i] = 1

# 4. Inject Missing Values (Clipboard gaps)
missing_humidity_mask = np.random.rand(n_samples) < 0.04
ambient_humidity[missing_humidity_mask] = np.nan

missing_temp_mask = np.random.rand(n_samples) < 0.02
oven_zone3_temp[missing_temp_mask] = np.nan

# 5. Export
df = pd.DataFrame({
    "Batch_ID": batch_ids,
    "Supplier_Name": suppliers,
    "Shift_Supervisor": supervisors,
    "Ambient_Humidity": ambient_humidity,
    "Flour_Moisture_Base": flour_moisture_base,
    "Mix_Time_Mins": mix_time_mins,
    "Conveyor_Speed_RPM": conveyor_speed_rpm,
    "Oven_Zone3_Temp": oven_zone3_temp,
    "Batch_Status": defect_labels
})

df.to_csv("malolos_biscuit_production.csv", index=False)
print(f"SUCCESS: Dataset generated. Shape: {df.shape}")
print(f"Corrected Defect Rate (Class 1 Ratio): {round((df['Batch_Status'].sum() / n_samples) * 100, 2)}%")