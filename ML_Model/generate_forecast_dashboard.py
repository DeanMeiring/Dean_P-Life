import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set file paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "sa_p_life_impact_predictions.csv")
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "sa_p_life_forecast_dashboard.png")

print("Loading predictions dataset for dashboard generation...")
if not os.path.exists(PREDICTIONS_PATH):
    raise FileNotFoundError(f"Could not find predictions file at {PREDICTIONS_PATH}. Please run train_sa_p_life_prediction_model.py first.")

df = pd.read_csv(PREDICTIONS_PATH)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Filter for a representative 50% P-Life adoption scenario over time
df_50 = df[df["P_Life_Adoption_Pct"] == 50.0].copy()

# Group companies into two distinct industry sectors matching the dual-panel layout
retail_companies = ["Shoprite Group", "Pick n Pay", "Woolworths", "Woolworths_SA"]
fmcg_companies = ["Coca-Cola Beverages SA", "Tiger Brands", "Clover SA", "Clover_SA"]

df_retail = df_50[df_50["Company_Name"].isin(retail_companies)]
df_fmcg = df_50[df_50["Company_Name"].isin(fmcg_companies)]

# Fallback: if names differ slightly, split overall companies into two equal groups
if df_retail.empty or df_fmcg.empty:
    unique_companies = df_50["Company_Name"].unique()
    mid = len(unique_companies) // 2
    retail_companies = unique_companies[:mid]
    fmcg_companies = unique_companies[mid:]
    df_retail = df_50[df_50["Company_Name"].isin(retail_companies)]
    df_fmcg = df_50[df_50["Company_Name"].isin(fmcg_companies)]

# Setup Figure Layout (1 row, 2 columns - matching target dashboard)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Main Dashboard Title
fig.suptitle("P-Life Implementation Pollution Abatement Forecast (50% Adoption Scenario)", fontsize=14, fontweight="bold", y=0.98)

# Styling palette & markers
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
markers_retail = ["o", "o", "o", "o"]
markers_fmcg = ["s", "s", "s", "s"]

# --- SUBPLOT 1: Retail & Grocery Sector ---
ax1.set_title("Retail & Grocery Sector (Monthly Net Unmanaged Leakage)", fontsize=11, fontweight="bold")

for idx, comp in enumerate(df_retail["Company_Name"].unique()):
    sub = df_retail[df_retail["Company_Name"] == comp].groupby("Date")["Target_Net_Unmanaged_Leakage_Tonnes"].mean().reset_index()
    avg_val = sub["Target_Net_Unmanaged_Leakage_Tonnes"].mean()
    label_text = f"{comp} ({avg_val:.1f} Tonnes)"
    ax1.plot(sub["Date"], sub["Target_Net_Unmanaged_Leakage_Tonnes"], marker=markers_retail[idx % len(markers_retail)], 
             color=colors[idx % len(colors)], linewidth=1.5, markersize=5, label=label_text)

# Threshold target line
threshold_retail = df_retail["Baseline_Unmanaged_Leakage_Tonnes"].mean() * 0.5
ax1.axhline(y=threshold_retail, color="green", linestyle="--", linewidth=1.2, label=f"50% Abatement Target ({threshold_retail:.1f}T)")

ax1.set_ylabel("Net Unmanaged Leakage (Tonnes)", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=8.5, frameon=True)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.tick_params(axis="x", rotation=45)

# --- SUBPLOT 2: FMCG & Beverage Sector ---
ax2.set_title("FMCG & Beverage Sector (Monthly Net Unmanaged Leakage)", fontsize=11, fontweight="bold")

for idx, comp in enumerate(df_fmcg["Company_Name"].unique()):
    sub = df_fmcg[df_fmcg["Company_Name"] == comp].groupby("Date")["Target_Net_Unmanaged_Leakage_Tonnes"].mean().reset_index()
    avg_val = sub["Target_Net_Unmanaged_Leakage_Tonnes"].mean()
    label_text = f"{comp} ({avg_val:.1f} Tonnes)"
    ax2.plot(sub["Date"], sub["Target_Net_Unmanaged_Leakage_Tonnes"], marker=markers_fmcg[idx % len(markers_fmcg)], 
             color=colors[(idx + 2) % len(colors)], linewidth=1.5, markersize=5, label=label_text)

# Threshold target line
threshold_fmcg = df_fmcg["Baseline_Unmanaged_Leakage_Tonnes"].mean() * 0.5
ax2.axhline(y=threshold_fmcg, color="green", linestyle="--", linewidth=1.2, label=f"50% Abatement Target ({threshold_fmcg:.1f}T)")

ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", fontsize=8.5, frameon=True)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.tick_params(axis="x", rotation=45)

# Tight layout adjustments
plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Save image
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches="tight")
print(f"Dashboard generated successfully and saved to: '{OUTPUT_IMAGE_PATH}'")