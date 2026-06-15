import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

print("--- Starting Task 2: Root-Cause Clustering Pipeline (Imbalanced Data) ---")

# 1. Load the raw data and apply the cleaning function
df_raw = pd.read_csv("malolos_biscuit_production.csv")

def clean_factory_data(data):
    df_clean = data.copy()
    df_clean['Ambient_Humidity'] = df_clean.groupby('Shift_Supervisor')['Ambient_Humidity'].transform(lambda x: x.fillna(x.median()))
    df_clean['Oven_Zone3_Temp'] = df_clean['Oven_Zone3_Temp'].fillna(df_clean['Oven_Zone3_Temp'].median())
    return df_clean

df_processed = clean_factory_data(df_raw)

# 2. Isolate ONLY the Defective Batches (The ~1.18%)
df_defects = df_processed[df_processed['Batch_Status'] == 1].copy()
print(f"Total Defective Batches Isolated for Analysis: {len(df_defects)}\n")

# 3. Select the physical factory features
features_to_cluster = ['Ambient_Humidity', 'Flour_Moisture_Base', 'Mix_Time_Mins', 'Oven_Zone3_Temp']
X_defects = df_defects[features_to_cluster]

# 4. Data Scaling (Crucial for distance-based clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_defects)

# 5. Generate the Dendrogram

plt.figure(figsize=(10, 6))
plt.title("Dendrogram of Malolos Factory Defective Batches (1.18% Crisis)")
plt.xlabel("Individual Defective Batches")
plt.ylabel("Mathematical Distance (Ward's Linkage)")
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.show()

# 6. Apply Agglomerative Clustering
hc_model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
df_defects['Failure_Profile'] = hc_model.fit_predict(X_scaled)

# 7. Print the Business Interpretations
print("--- Factory Floor Failure Profiles Discovered ---")
profile_summary = df_defects.groupby('Failure_Profile')[features_to_cluster].mean().round(2)
print(profile_summary)