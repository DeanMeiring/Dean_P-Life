import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Set file paths relative to ML_Model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SA_DATA_PATH = os.path.join(BASE_DIR, "..", "South_African_Data", "final_sa_plastics_ml.csv")
OUTPUT_PREDICTIONS_PATH = os.path.join(BASE_DIR, "sa_p_life_impact_predictions.csv")

print("Initializing South Africa P-Life Predictive Modeling Pipeline...\n")

# 1. Load South Africa Dataset
if os.path.exists(SA_DATA_PATH):
    df_sa = pd.read_csv(SA_DATA_PATH)
    print(f"Loaded SA Dataset: {len(df_sa)} rows")
else:
    raise FileNotFoundError(f"Could not locate dataset at {SA_DATA_PATH}")

# Harmonize column names between dataset output and ML script
if "Company_Estimated_Unmanaged_Leakage_Tonnes" in df_sa.columns:
    df_sa["Baseline_Unmanaged_Leakage_Tonnes"] = df_sa["Company_Estimated_Unmanaged_Leakage_Tonnes"]
if "Company_Mechanical_Recycling_Rate_Pct" in df_sa.columns:
    df_sa["Company_Recycled_Input_Pct"] = df_sa["Company_Mechanical_Recycling_Rate_Pct"]

# 2. Benchmark Parameters (JIS K6955 / ASTM D6954 24-month soil tests)
PLIFE_BIOASSIMILATION_EFFICIENCY = 0.85 

# 3. Scenario Matrix Generation (0% to 100% P-Life Adoption)
print("Generating counterfactual simulation matrix across P-Life adoption scenarios (0% to 100%)...")
scenarios = []
adoption_levels = [0.0, 0.15, 0.35, 0.50, 0.75, 1.00]

for adoption in adoption_levels:
    df_scen = df_sa.copy()
    df_scen["P_Life_Adoption_Pct"] = adoption * 100.0
    
    # Calculate P-Life treated virgin polyolefin volume
    df_scen["P_Life_Treated_Packaging_Tonnes"] = df_scen["Company_Virgin_Packaging_Tonnes"] * adoption
    
    # Calculate potential unmanaged leakage from P-Life treated plastics
    df_scen["P_Life_Treated_Unmanaged_Leakage_Tonnes"] = (
        df_scen["Baseline_Unmanaged_Leakage_Tonnes"] * adoption
    )
    
    # Abated Leakage through microbial bioassimilation
    df_scen["Abated_Unmanaged_Leakage_Tonnes"] = (
        df_scen["P_Life_Treated_Unmanaged_Leakage_Tonnes"] * PLIFE_BIOASSIMILATION_EFFICIENCY
    )
    
    # Target 1: Net Persistent Environmental Leakage
    df_scen["Target_Net_Unmanaged_Leakage_Tonnes"] = (
        df_scen["Baseline_Unmanaged_Leakage_Tonnes"] - df_scen["Abated_Unmanaged_Leakage_Tonnes"]
    )
    
    # Target 2: Pollution Reduction Percentage
    # Avoid division by zero for edge cases
    df_scen["Target_Pollution_Reduction_Pct"] = np.where(
        df_scen["Baseline_Unmanaged_Leakage_Tonnes"] > 0,
        (df_scen["Abated_Unmanaged_Leakage_Tonnes"] / df_scen["Baseline_Unmanaged_Leakage_Tonnes"]) * 100.0,
        0.0
    )
    
    scenarios.append(df_scen)

df_full = pd.concat(scenarios, ignore_index=True)

# 4. Feature Engineering & Preprocessing
features_num = [
    "Company_Virgin_Packaging_Tonnes",
    "Company_Recycled_Input_Pct",
    "Municipal_Uncollected_Waste_Pct",
    "Baseline_Unmanaged_Leakage_Tonnes",
    "P_Life_Adoption_Pct"
]
features_cat = ["Company_Name", "Province"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cat = encoder.fit_transform(df_full[features_cat])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(features_cat))

X = pd.concat([df_full[features_num].reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)
y_leakage = df_full["Target_Net_Unmanaged_Leakage_Tonnes"]
y_reduction = df_full["Target_Pollution_Reduction_Pct"]

# 5. Model Training & Evaluation
X_train, X_test, y_train_leak, y_test_leak = train_test_split(X, y_leakage, test_size=0.2, random_state=42)
_, _, y_train_red, y_test_red = train_test_split(X, y_reduction, test_size=0.2, random_state=42)

model_leakage = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
model_leakage.fit(X_train, y_train_leak)

model_reduction = RandomForestRegressor(n_estimators=100, random_state=42)
model_reduction.fit(X_train, y_train_red)

# Evaluate Models
pred_leak = model_leakage.predict(X_test)
pred_red = model_reduction.predict(X_test)

print("--- MODEL PERFORMANCE METRICS ---")
print(f"Net Leakage Model R² Score : {r2_score(y_test_leak, pred_leak):.4f}")
print(f"Net Leakage Model RMSE     : {np.sqrt(mean_squared_error(y_test_leak, pred_leak)):.2f} Tonnes")
print(f"Pollution Reduction R²     : {r2_score(y_test_red, pred_red):.4f}")
print("---------------------------------\n")

# 6. Prediction Function
def predict_company_p_life_impact(company_name, province_name, adoption_pct, monthly_virgin_tonnes=500.0):
    input_dict = {
        "Company_Virgin_Packaging_Tonnes": [monthly_virgin_tonnes],
        "Company_Recycled_Input_Pct": [20.0],
        "Municipal_Uncollected_Waste_Pct": [15.0],
        "Baseline_Unmanaged_Leakage_Tonnes": [(monthly_virgin_tonnes * 0.8) * 0.15],
        "P_Life_Adoption_Pct": [adoption_pct],
        "Company_Name": [company_name],
        "Province": [province_name]
    }
    df_input = pd.DataFrame(input_dict)
    
    enc_cat = encoder.transform(df_input[features_cat])
    enc_cat_df = pd.DataFrame(enc_cat, columns=encoder.get_feature_names_out(features_cat))
    X_single = pd.concat([df_input[features_num].reset_index(drop=True), enc_cat_df.reset_index(drop=True)], axis=1)
    
    pred_net_leak = model_leakage.predict(X_single)[0]
    pred_pct_red = model_reduction.predict(X_single)[0]
    
    baseline_leak = df_input["Baseline_Unmanaged_Leakage_Tonnes"][0]
    abated_tonnes = baseline_leak - pred_net_leak
    
    return {
        "Company": company_name,
        "Province": province_name,
        "P_Life_Adoption": f"{adoption_pct}%",
        "Baseline_Unmanaged_Leakage_Tonnes": round(baseline_leak, 2),
        "Predicted_Net_Leakage_Tonnes": round(pred_net_leak, 2),
        "Predicted_Abated_Pollution_Tonnes": round(abated_tonnes, 2),
        "Predicted_Pollution_Reduction_Pct": f"{round(pred_pct_red, 2)}%"
    }

# 7. Execute Sample Scenario Test
print("--- PREDICTIVE SCENARIO SIMULATIONS (SA COMPANIES) ---")
test_scenarios = [
    ("Coca-Cola Beverages SA", "Gauteng", 50.0, 850.0),
    ("Coca-Cola Beverages SA", "Gauteng", 100.0, 850.0),
    ("Shoprite Group", "Western Cape", 35.0, 620.0),
    ("Shoprite Group", "Western Cape", 75.0, 620.0),
    ("Woolworths", "KwaZulu-Natal", 50.0, 400.0)
]

sim_results = [predict_company_p_life_impact(c, p, a, t) for c, p, a, t in test_scenarios]
df_sim = pd.DataFrame(sim_results)
print(df_sim.to_string(index=False))

# Save output
df_full.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
print(f"\nPipeline complete! Full predictions saved to: '{OUTPUT_PREDICTIONS_PATH}'")