import pandas as pd
import numpy as np

# Load the newly generated production file
df_raw = pd.read_csv("malolos_biscuit_production.csv")

# 1. Structural Imputation Function (Contextual Group Filling)
def clean_factory_data(data):
    df_clean = data.copy()
    
    # Impute missing Ambient Humidity based on individual Shift Supervisor medians
    df_clean['Ambient_Humidity'] = df_clean.groupby('Shift_Supervisor')['Ambient_Humidity'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Impute missing Oven Temperatures globally via data median to preserve distribution structure
    global_temp_median = df_clean['Oven_Zone3_Temp'].median()
    df_clean['Oven_Zone3_Temp'] = df_clean['Oven_Zone3_Temp'].fillna(global_temp_median)
    
    return df_clean

df_processed = clean_factory_data(df_raw)

# 2. Inspecting the Preprocessing Matrix
print("--- Missing Values Corrected ---")
print(df_processed.isnull().sum())
print("\n--- Basic Class Balance Baseline ---")
print(df_processed['Batch_Status'].value_counts(normalize=True))