import pandas as pd
import joblib
from xgboost import XGBClassifier

print("--- Starting Task 6: Model Serialization for Live Deployment --- \n")

# 1. Load and Prep Data (Using all data for final deployment)
df_raw = pd.read_csv("malolos_biscuit_production.csv")

def clean_factory_data(data):
    df_clean = data.copy()
    df_clean['Ambient_Humidity'] = df_clean.groupby('Shift_Supervisor')['Ambient_Humidity'].transform(lambda x: x.fillna(x.median()))
    df_clean['Oven_Zone3_Temp'] = df_clean['Oven_Zone3_Temp'].fillna(df_clean['Oven_Zone3_Temp'].median())
    return df_clean

df_processed = clean_factory_data(df_raw)

# 2. Extract our final feature list (Must match the live floor data exactly)
# We use get_dummies just like before
df_encoded = pd.get_dummies(df_processed, columns=['Supplier_Name', 'Shift_Supervisor'], drop_first=True)

X = df_encoded.drop(['Batch_ID', 'Batch_Status'], axis=1)
y = df_encoded['Batch_Status']

# 3. Initialize XGBoost with our Winning Parameters from Task 4
# These are the exact settings that found the 1.18% needles in the haystack
final_model = XGBClassifier(
    learning_rate=0.05,
    max_depth=5,
    n_estimators=200,
    scale_pos_weight=50, # The mathematical penalty we calculated
    random_state=42,
    eval_metric='logloss'
)

print("Training the final model on 100% of historical data...")
final_model.fit(X, y)

# 4. Save (Serialize) the AI Brain
model_filename = 'malolos_biscuit_predictor.pkl'
joblib.dump(final_model, model_filename)

# 5. Save the required feature columns for the IT Team
# The IT team needs to know the exact format the model expects data in
features_filename = 'required_model_features.pkl'
joblib.dump(list(X.columns), features_filename)

print(f"\nSUCCESS! Model successfully serialized.")
print(f"File 1: '{model_filename}' (The AI Brain)")
print(f"File 2: '{features_filename}' (The IT Schema)")
print("\nHand these two files to your Automation/IT team to deploy to the live SCADA dashboard.")