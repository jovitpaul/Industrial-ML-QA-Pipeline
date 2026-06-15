import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("--- Starting Task 3: Supervised Model Tournament (Extreme Imbalance) --- \n")

# 1. Load and Clean the Data
df_raw = pd.read_csv("malolos_biscuit_production.csv")

def clean_factory_data(data):
    df_clean = data.copy()
    df_clean['Ambient_Humidity'] = df_clean.groupby('Shift_Supervisor')['Ambient_Humidity'].transform(lambda x: x.fillna(x.median()))
    df_clean['Oven_Zone3_Temp'] = df_clean['Oven_Zone3_Temp'].fillna(df_clean['Oven_Zone3_Temp'].median())
    return df_clean

df_processed = clean_factory_data(df_raw)

# 2. Feature Engineering
df_encoded = pd.get_dummies(df_processed, columns=['Supplier_Name', 'Shift_Supervisor'], drop_first=True)

X = df_encoded.drop(['Batch_ID', 'Batch_Status'], axis=1)
y = df_encoded['Batch_Status']

# 3. Stratified Train-Test Split (CRITICAL for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

print(f"Training Data Size: {X_train.shape[0]} batches")
print(f"Testing Data Size:  {X_test.shape[0]} batches\n")

# 4. Initialize the Three Competitors
models = {
    "1. Baseline Decision Tree": DecisionTreeClassifier(random_state=42),
    "2. Random Forest (Bagging)": RandomForestClassifier(random_state=42),
    "3. XGBoost (Boosting)": XGBClassifier(random_state=42, eval_metric='logloss')
}

# 5. Train and Evaluate Each Model
for name, model in models.items():
    print(f"========== {name} ==========")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Caught Defects (True Positives): {cm[1][1]}")
    print(f"Missed Defects (False Negatives): {cm[1][0]}\n")