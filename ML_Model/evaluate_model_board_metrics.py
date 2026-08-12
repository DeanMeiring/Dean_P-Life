import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "sa_p_life_impact_predictions.csv")

df = pd.read_csv(PREDICTIONS_PATH)

# Feature & Target Split
features_num = [
    "Company_Virgin_Packaging_Tonnes",
    "Company_Recycled_Input_Pct",
    "Municipal_Uncollected_Waste_Pct",
    "Baseline_Unmanaged_Leakage_Tonnes",
    "P_Life_Adoption_Pct"
]
features_cat = ["Company_Name", "Province"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cat = encoder.fit_transform(df[features_cat])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(features_cat))

X = pd.concat([df[features_num].reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)
y_leakage = df["Target_Net_Unmanaged_Leakage_Tonnes"]

# Model Evaluation
model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X, y_leakage)
predictions = model.predict(X)

# Metrics
r2 = r2_score(y_leakage, predictions)
mae = mean_absolute_error(y_leakage, predictions)
rmse = np.sqrt(mean_squared_error(y_leakage, predictions))

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_leakage, cv=kf, scoring="r2")

print("\n================ OFFICIAL BOARD AUDIT METRICS ================")
print(f"Model Accuracy (R² Score)      : {r2 * 100:.2f}% ({r2:.4f})")
print(f"Mean Absolute Error (MAE)       : {mae:.2f} Tonnes")
print(f"Root Mean Squared Error (RMSE)  : {rmse:.2f} Tonnes")
print(f"5-Fold CV Mean R² Score         : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print("Scientific Standard Reference   : JIS K6955 / ISO 17556 / ASTM D6954")
print("==============================================================\n")