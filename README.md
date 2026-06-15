# Malolos Biscuit Corp: AI-Driven Quality Assurance & Predictive Maintenance

## Executive Overview
In Philippine continuous food manufacturing, the **Pareto Chart** is the traditional gold standard for Root Cause Analysis (RCA). However, traditional RCA struggles to identify root causes when dealing with highly imbalanced defect rates and complex, multi-variable interactions (e.g., how a slight grid voltage sag interacts with seasonal monsoon humidity to ruin a batch). 

This project bridges the gap between factory operations and advanced data science. It transitions from reactive, one-dimensional RCA to a **proactive Machine Learning pipeline** capable of analyzing thousands of variables simultaneously to prevent structural moisture failures ("soggy batches") on the production floor.

---

## The Business Problem
* **The Bottleneck:** The factory is experiencing a highly expensive 1.18% defect rate resulting in scrapped raw materials and disrupted shipments.
* **The ML Strategy:** To solve this, the engineering approach moves through two phases: 
  1. **Unsupervised Learning** to mathematically discover the hidden root causes (The "ML Pareto").
  2. **Supervised Learning** to build an early-warning predictive system capable of handling extreme class imbalance (98.8% Pass vs. 1.18% Defect).

---

## Repository Structure & Technical Pipeline

The pipeline is divided into 6 sequential Python scripts, reflecting a true end-to-end Machine Learning lifecycle:

### `task1_data_generator.py` | Data Simulation & Imbalance Engineering
* **Rationale:** Real-world manufacturing data is never perfectly balanced. This script generates 5,500 historical batch logs embedding deterministic engineering physics (e.g., severe power sags in Oven Zone 3, and monsoon humidity interacting with wet flour).
* **Result:** Produced a highly realistic dataset with a true **1.18% minority class defect rate**, perfectly setting up the "needle in a haystack" challenge.

### `task2_clustering.py` | Root-Cause Profiling (Unsupervised ML)
* **Rationale:** Before predicting defects, the business must understand *why* they happen. Agglomerative Hierarchical Clustering was deployed on the defective batches using Euclidean mathematical distance.
* **Result:** Successfully isolated three distinct failure profiles: Voltage sags dropping oven temps, elevated flour moisture interacting with long mix times, and baseline random factory noise.

### `task3_supervised.py` | Supervised Algorithm Tournament
* **Rationale:** A competitive pipeline comparing a Decision Tree, Random Forest, and XGBoost using a Stratified Train-Test split. Global "Accuracy" was strictly ignored to avoid the accuracy trap of imbalanced data, focusing entirely on Class 1 Recall and Precision.
* **Result:** XGBoost proved to be the most capable engine out-of-the-box but required severe calibration to prevent line-stopping false alarms.

### `task4_tuning.py` | XGBoost Hyperparameter Calibration
* **Rationale:** Utilizing `GridSearchCV` optimized for the F1-Score. The `scale_pos_weight` was strictly calibrated to combat the 98.8% vs 1.18% imbalance, forcing the algorithm to financially penalize missed defects much heavier than normal variance.
* **Result:** Achieved the optimal operational sweet spot—catching the vast majority of true predictable defects while triggering near-zero false alarms.

### `task5_deployment.py` | Business Rule Extraction (The "ML Pareto")
* **Rationale:** "Black box" models cannot be handed directly to plant operators. A tuned Random Forest was used to extract "Gini Importance" metrics, translating complex mathematics into a ranked list of the physical sensors driving financial losses.
* **Result:** Outputted a Feature Importance chart proving that `Oven_Zone3_Temp` accounts for nearly 50% of the predictive failure weight, followed by supplier `Flour_Moisture_Base`.

### `task6_export_model.py` | Live SCADA Serialization
* **Rationale:** Preparing the validated model for the factory floor.
* **Result:** The final, optimized XGBoost brain was trained on 100% of historical data and exported as a `.pkl` file, completely ready for live IT/SCADA integration to monitor real-time sensor streams.

---

## Key Findings & Standard Operating Procedure (SOP)

The machine learning pipeline mathematically exonerated the floor operators from "human error." The variance is entirely infrastructure and supply-chain driven. Based on these algorithmic findings, the recommended SOP is:

1. **Capital Expenditure:** Immediately procure and install a localized Uninterruptible Power Supply (UPS) for Oven Zone 3 to eliminate the primary defect trigger (voltage sags).
2. **Supply Chain Audit:** Quality Assurance must enforce stricter baseline moisture testing on Supplier B shipments upon receiving.
3. **Operational Cap:** Implement a hard cap of 17 minutes on mix times during periods of extreme Habagat (monsoon) humidity to prevent dough over-hydration.

---

## How to Run This Project Locally
1. Clone the repository: `git clone. 
2. Ensure you have the required libraries: `pip install pandas numpy scikit-learn xgboost matplotlib scipy`
3. Run the scripts sequentially from `task1` through `task6` to generate the data, models, and visualizations.
