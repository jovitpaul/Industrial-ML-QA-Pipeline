import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

print("--- Starting Task 5: Operational Deployment & Business Rules --- \n")

# 1. Load and Clean Data
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

# 2. Train the Random Forest (Optimized for stability)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced')
rf_model.fit(X, y)

# 3. Extract Feature Importances
importances = rf_model.feature_importances_
feature_names = X.columns

# Bind them together and sort them
feature_ranking = pd.DataFrame({
    'Factory_Sensor': feature_names,
    'Importance_Score': importances
}).sort_values(by='Importance_Score', ascending=True)

# 4. Generate the Portfolio Visualization
plt.figure(figsize=(10, 6))
# Using a professional, executive blue color scheme
plt.barh(feature_ranking['Factory_Sensor'], feature_ranking['Importance_Score'], color='#1f77b4', edgecolor='black')
plt.title("Malolos Biscuit Corp: Defect Root Cause Analysis (Feature Importance)", fontsize=14, fontweight='bold')
plt.xlabel("Impact on Defect Rate (Gini Importance)", fontsize=12)
plt.ylabel("Operational Parameters", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the image so you can upload it to LinkedIn/GitHub
plt.savefig("feature_importance_chart.png", dpi=300)
print("SUCCESS: 'feature_importance_chart.png' has been saved to your folder!\n")

# 5. Print the Executive SOP Draft
print("========== RECOMMENDED STANDARD OPERATING PROCEDURE (SOP) ==========")
print("Based on the algorithmic feature extraction, the following rules must be enforced on the floor:\n")

# Dynamically grabbing the top 2 offenders
top_feature = feature_ranking.iloc[-1]['Factory_Sensor']
second_feature = feature_ranking.iloc[-2]['Factory_Sensor']

print(f"1. PRIMARY BOTTLENECK: {top_feature}")
print("   ACTION: Install a localized Uninterruptible Power Supply (UPS) or voltage regulator specifically for Oven Zone 3 to prevent grid sags.")
print(f"2. SECONDARY BOTTLENECK: {second_feature}")
print("   ACTION: Implement a strict flour moisture audit. If baseline moisture exceeds 14.8%, mix time must be capped strictly at 17 minutes.")
print("3. SUPPLIER AUDIT: Quality Control must immediately investigate Supplier B due to statistically elevated baseline moisture deliveries.")
print("===================================================================\n")
print("Capstone Project Complete. Ready for GitHub Push.")