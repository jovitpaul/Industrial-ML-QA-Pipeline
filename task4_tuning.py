import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("--- Starting Task 4: Precision XGBoost Tuning (Extreme Imbalance) --- \n")
print("Initializing Grid Search... Tuning the engine for the 1.18% crisis. Please wait.\n")

# 1. Load and Prep Data
df_raw = pd.read_csv("malolos_biscuit_production.csv")

def clean_factory_data(data):
    df_clean = data.copy()
    df_clean['Ambient_Humidity'] = df_clean.groupby('Shift_Supervisor')['Ambient_Humidity'].transform(lambda x: x.fillna(x.median()))
    df_clean['Oven_Zone3_Temp'] = df_clean['Oven_Zone3_Temp'].fillna(df_clean['Oven_Zone3_Temp'].median())
    return df_clean

df_processed = clean_factory_data(df_raw)
df_encoded = pd.get_dummies(df_processed, columns=['Supplier_Name', 'Shift_Supervisor'], drop_first=True)

X = df_encoded.drop(['Batch_ID', 'Batch_Status'], axis=1)
y = df_encoded['Batch_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# 2. Define the XGBoost Base Engine
xgb_baseline = XGBClassifier(random_state=42, eval_metric='logloss')

# 3. Create the Tuning Grid (The Precision Valves)
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    # We test 1 (no penalty), 50 (medium penalty), and 85 (the true mathematical ratio)
    'scale_pos_weight': [1, 50, 85] 
}

# 4. Run GridSearchCV (Optimized for F1-Score to prevent false alarm collapse)
grid_search = GridSearchCV(
    estimator=xgb_baseline, 
    param_grid=param_grid, 
    cv=3,                 
    scoring='f1',         
    verbose=1             
)

grid_search.fit(X_train, y_train)

# 5. Extract and Evaluate the Champion Model
best_xgb = grid_search.best_estimator_

print("\n========== OPTIMIZED XGBOOST RESULTS ==========")
print(f"Best Parameters Found: {grid_search.best_params_}\n")

y_pred_optimized = best_xgb.predict(X_test)

print(classification_report(y_test, y_pred_optimized, zero_division=0))

cm = confusion_matrix(y_test, y_pred_optimized)
print(f"Good Batches Passed (True Negatives): {cm[0][0]}")
print(f"False Alarms (False Positives): {cm[0][1]}")
print(f"Missed Defects (False Negatives): {cm[1][0]}")
print(f"Caught Defects (True Positives): {cm[1][1]}\n")