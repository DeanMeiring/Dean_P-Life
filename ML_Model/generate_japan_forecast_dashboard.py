import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set file paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JAPAN_DATA_PATH = os.path.join(BASE_DIR, "..", "International_Data_Implemented_P-Life", "final_japan_p_life_impact_ml.csv")
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "japan_p_life_historical_declines.png")

print("Loading Japan P-Life implementation dataset...")
if not os.path.exists(JAPAN_DATA_PATH):
    raise FileNotFoundError(f"Could not find Japan dataset at {JAPAN_DATA_PATH}. Please run build_international_p_life_database.py first.")

df = pd.read_csv(JAPAN_DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# Create 1x2 dashboard matching your target style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

fig.suptitle("Japan Historical Plastic Pollution Decline Post-P-Life Rollout (2010 - 2025)", fontsize=14, fontweight="bold", y=0.98)

# --- SUBPLOT 1: Sector-Specific Persistent Leakage ---
sectors = df["Sector"].unique()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
markers = ["o", "s", "^"]

ax1.set_title("Sector-Wise Persistent Plastic Leakage (Tonnes/Month)", fontsize=11, fontweight="bold")

for idx, sec in enumerate(sectors):
    sub = df[df["Sector"] == sec].sort_values("Date")
    label_text = sec.replace("_", " ")
    ax1.plot(sub["Date"], sub["Net_Persistent_Environmental_Leakage_Tonnes"], 
             marker=markers[idx], color=colors[idx], linewidth=1.5, markersize=3, markevery=12, label=label_text)

ax1.set_ylabel("Net Persistent Environmental Leakage (Tonnes)", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=8.5, frameon=True)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.tick_params(axis="x", rotation=45)

# --- SUBPLOT 2: National Total Baseline vs. P-Life Abated Pollution ---
national_df = df.groupby("Date").agg({
    "Raw_Unmanaged_Leakage_Tonnes": "sum",
    "Net_Persistent_Environmental_Leakage_Tonnes": "sum"
}).reset_index()

ax2.set_title("Japan National Monthly Unmanaged Plastic Accumulation", fontsize=11, fontweight="bold")

ax2.plot(national_df["Date"], national_df["Raw_Unmanaged_Leakage_Tonnes"], color="red", linestyle="--", linewidth=1.5, label="Baseline Unmanaged Leakage (Without P-Life)")
ax2.plot(national_df["Date"], national_df["Net_Persistent_Environmental_Leakage_Tonnes"], color="green", linewidth=2.0, label="Actual Persistent Residue (With P-Life)")

# Milestones
ax2.axvline(pd.to_datetime("2016-01-01"), color="gray", linestyle=":", linewidth=1.2, label="2016: Beverage Sector Rollout")
ax2.axvline(pd.to_datetime("2018-01-01"), color="purple", linestyle=":", linewidth=1.2, label="2018: Konbini Packaging Rollout")

ax2.set_ylabel("Total National Leakage (Tonnes)", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower left", fontsize=8.5, frameon=True)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Save chart image
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches="tight")
print(f"Japan pollution decline dashboard saved to: '{OUTPUT_IMAGE_PATH}'")