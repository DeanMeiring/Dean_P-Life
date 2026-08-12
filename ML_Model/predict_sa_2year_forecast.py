import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set file paths relative to ML_Model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "sa_p_life_impact_predictions.csv")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "sa_2year_p_life_forecast.csv")
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "sa_2year_p_life_forecast.png")

print("Generating 2-Year South Africa P-Life Adoption Impact Forecast (2026-2028)...")

# 1. Create Next 2 Years Date Range (2026-09 to 2028-08 -> 24 Months)
dates = pd.date_range(start="2026-09-01", periods=24, freq="MS")

companies = ["Coca-Cola Beverages SA", "Shoprite Group", "Pick n Pay", "Tiger Brands", "Clover SA", "Woolworths"]
company_baselines = {
    "Coca-Cola Beverages SA": {"virgin_tonnes": 850.0, "uncollected_pct": 14.5},
    "Shoprite Group": {"virgin_tonnes": 680.0, "uncollected_pct": 12.0},
    "Pick n Pay": {"virgin_tonnes": 520.0, "uncollected_pct": 12.0},
    "Tiger Brands": {"virgin_tonnes": 600.0, "uncollected_pct": 15.0},
    "Clover SA": {"virgin_tonnes": 450.0, "uncollected_pct": 13.5},
    "Woolworths": {"virgin_tonnes": 380.0, "uncollected_pct": 9.5}
}

# Seasonality multiplier for SA consumer/packaging cycles
seasonality = np.array([0.95, 0.98, 1.02, 1.08, 1.15, 0.92, 0.91, 0.94, 0.97, 1.01, 1.04, 1.03])
seasonality /= seasonality.mean()

forecast_rows = []

# Generate S-curve adoption ramp-up over 24 months (0% -> ~70% adoption by end of Year 2)
for idx, d in enumerate(dates):
    # Ramp up factor: gradual onboarding across 24 months
    ramp_factor = 1.0 / (1.0 + np.exp(-0.25 * (idx - 10))) 
    adoption_pct = float(ramp_factor * 75.0) # Peaks at 75% market adoption
    
    month_season = seasonality[d.month - 1]
    
    for comp in companies:
        cfg = company_baselines[comp]
        monthly_virgin = cfg["virgin_tonnes"] * month_season
        baseline_leakage = monthly_virgin * (cfg["uncollected_pct"] / 100.0)
        
        # P-Life 85% bioassimilation impact on treated portion
        treated_leakage = baseline_leakage * (adoption_pct / 100.0)
        abated_leakage = treated_leakage * 0.85
        net_leakage = baseline_leakage - abated_leakage
        
        forecast_rows.append({
            "Date": d,
            "Year": d.year,
            "Month": d.month,
            "Company_Name": comp,
            "P_Life_Adoption_Pct": round(adoption_pct, 2),
            "Virgin_Packaging_Tonnes": round(monthly_virgin, 2),
            "Baseline_Unmanaged_Leakage_Tonnes": round(baseline_leakage, 2),
            "Abated_Pollution_Tonnes": round(abated_leakage, 2),
            "Net_Persistent_Leakage_Tonnes": round(net_leakage, 2)
        })

df_forecast = pd.DataFrame(forecast_rows)
df_forecast.to_csv(OUTPUT_CSV_PATH, index=False)

# 2. Build Dashboard Forecast Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

fig.suptitle("2-Year Forecast: SA Company P-Life Onboarding Impact on Plastic Leakage (2026 - 2028)", fontsize=13, fontweight="bold", y=0.98)

# Panel 1: Company-Level Net Leakage Trajectories
colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]
ax1.set_title("Company Monthly Persistent Plastic Pollution (Tonnes)", fontsize=11, fontweight="bold")

for idx, comp in enumerate(companies):
    sub = df_forecast[df_forecast["Company_Name"] == comp]
    ax1.plot(sub["Date"], sub["Net_Persistent_Leakage_Tonnes"], marker="o", color=colors[idx], 
             linewidth=1.5, markersize=3, label=f"{comp}")

ax1.set_ylabel("Net Environmental Leakage (Tonnes)", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=8, frameon=True)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.tick_params(axis="x", rotation=45)

# Panel 2: Total SA Benchmark (Baseline vs. P-Life Abated Trajectory)
nat_df = df_forecast.groupby("Date").agg({
    "Baseline_Unmanaged_Leakage_Tonnes": "sum",
    "Net_Persistent_Leakage_Tonnes": "sum",
    "Abated_Pollution_Tonnes": "sum",
    "P_Life_Adoption_Pct": "mean"
}).reset_index()

ax2.set_title("Total Combined Monthly Pollution Abatement across all 6 SA Corporates", fontsize=11, fontweight="bold")

ax2.plot(nat_df["Date"], nat_df["Baseline_Unmanaged_Leakage_Tonnes"], color="red", linestyle="--", linewidth=1.5, label="Baseline Pollution (Without P-Life)")
ax2.plot(nat_df["Date"], nat_df["Net_Persistent_Leakage_Tonnes"], color="green", linewidth=2.0, label="Projected Pollution (With P-Life Onboarding)")

# Highlight total abated gap
ax2.fill_between(nat_df["Date"], nat_df["Baseline_Unmanaged_Leakage_Tonnes"], nat_df["Net_Persistent_Leakage_Tonnes"], color="green", alpha=0.15, label="Abated Plastic Accumulation")

ax2.set_ylabel("Total Monthly Unmanaged Plastic (Tonnes)", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower left", fontsize=8.5, frameon=True)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches="tight")

# Print Summary Metrics
tot_baseline_2yr = nat_df["Baseline_Unmanaged_Leakage_Tonnes"].sum()
tot_net_2yr = nat_df["Net_Persistent_Leakage_Tonnes"].sum()
tot_abated_2yr = nat_df["Abated_Pollution_Tonnes"].sum()
pct_overall_red = (tot_abated_2yr / tot_baseline_2yr) * 100.0

print("\n================ 2-YEAR SA P-LIFE FORECAST SUMMARY ================")
print(f"Forecast Window                    : Sept 2026 – Aug 2028")
print(f"Cumulative Baseline Leakage        : {tot_baseline_2yr:,.2f} Tonnes")
print(f"Cumulative Net Persistent Leakage  : {tot_net_2yr:,.2f} Tonnes")
print(f"Total Plastic Prevented / Abated   : {tot_abated_2yr:,.2f} Tonnes")
print(f"Overall Net Pollution Reduction    : {pct_overall_red:.2f}%")
print("====================================================================\n")
print(f"Dashboard saved to: '{OUTPUT_IMAGE_PATH}'")