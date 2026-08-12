import os
import io
import requests
import pandas as pd
import numpy as np

# Automatically targets the folder relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Executing South Africa Plastics Monthly, Provincial & Corporate ML Data Pipeline...")

# 1. Generate Monthly Date Range (2010-01 to 2025-12 -> 192 Months)
dates = pd.date_range(start="2010-01-01", end="2025-12-01", freq="MS")
df_monthly = pd.DataFrame({"Date": dates})
df_monthly["Year"] = df_monthly["Date"].dt.year
df_monthly["Month"] = df_monthly["Date"].dt.month

# Annual target baselines (National totals)
annual_cons_data = {
    "Year": list(range(2010, 2026)),
    "Annual_Virgin_Tonnes": [
        1320000, 1350000, 1380000, 1410000, 1450000, 1490000, 1510000, 1540000,
        1560000, 1580000, 1420000, 1510000, 1550000, 1590000, 1620000, 1650000
    ],
    "Annual_Recycled_Tonnes": [
        241000, 256000, 268000, 285000, 292000, 308000, 329000, 334000,
        352000, 337000, 296000, 312000, 338000, 355000, 368000, 380000
    ],
    "Per_Capita_Kg": [
        25.8, 26.1, 26.4, 26.8, 27.2, 27.8, 28.0, 28.3,
        28.5, 28.7, 25.6, 27.0, 27.5, 28.0, 28.4, 28.8
    ],
    "Packaging_Share_Pct": [
        52.1, 52.5, 53.0, 52.8, 53.2, 53.5, 53.8, 54.0,
        54.2, 53.9, 52.5, 53.1, 53.6, 54.0, 54.3, 54.5
    ]
}
df_annual = pd.DataFrame(annual_cons_data)

# Merge annual targets into monthly frame
df_cons = pd.merge(df_monthly, df_annual, on="Year", how="left")

# Monthly seasonality multiplier (Q4 consumer/packaging surge)
monthly_seasonality = np.array([0.94, 0.93, 0.98, 0.97, 0.99, 0.98, 0.99, 1.01, 1.02, 1.04, 1.08, 1.07])
monthly_seasonality /= monthly_seasonality.mean()

df_cons["Seasonal_Factor"] = df_cons["Month"].map(lambda m: monthly_seasonality[m - 1])

df_cons["National_Virgin_Tonnes"] = (df_cons["Annual_Virgin_Tonnes"] / 12) * df_cons["Seasonal_Factor"]
df_cons["National_Recycled_Tonnes"] = (df_cons["Annual_Recycled_Tonnes"] / 12) * df_cons["Seasonal_Factor"]
df_cons["Per_Capita_Consumption_Kg"] = df_cons["Per_Capita_Kg"].interpolate(method="linear")
df_cons["Packaging_Share_Percent"] = df_cons["Packaging_Share_Pct"].interpolate(method="linear")

df_cons = df_cons.drop(columns=["Annual_Virgin_Tonnes", "Annual_Recycled_Tonnes", "Per_Capita_Kg", "Packaging_Share_Pct", "Seasonal_Factor"])

# 2. Provincial Profiles & Cross-Join
provincial_profiles = {
    "Province": [
        "Gauteng", "KwaZulu-Natal", "Western Cape", "Eastern Cape",
        "Mpumalanga", "Limpopo", "Free State", "North West", "Northern Cape"
    ],
    "Province_Code": ["GP", "KZN", "WC", "EC", "MP", "LP", "FS", "NW", "NC"],
    "Province_Consumption_Share": [0.35, 0.18, 0.15, 0.08, 0.06, 0.06, 0.05, 0.05, 0.02],
    "Recycling_Efficiency_Multiplier": [1.10, 0.95, 1.25, 0.90, 0.85, 0.60, 0.80, 0.75, 0.70],
    "Municipal_Uncollected_Waste_Pct": [22.0, 32.0, 18.0, 38.0, 41.0, 52.0, 39.0, 44.0, 35.0]
}
df_prov = pd.DataFrame(provincial_profiles)
df_cons = df_cons.merge(df_prov, how="cross")

df_cons["Provincial_Virgin_Tonnes"] = df_cons["National_Virgin_Tonnes"] * df_cons["Province_Consumption_Share"]
df_cons["Provincial_Recycled_Tonnes"] = (
    df_cons["National_Recycled_Tonnes"] * df_cons["Province_Consumption_Share"] * df_cons["Recycling_Efficiency_Multiplier"]
)

# Overwrite raw_sa_consumption.csv
df_cons.to_csv(os.path.join(SCRIPT_DIR, "raw_sa_consumption.csv"), index=False)

# 3. Corporate Contributor Profiles & Cross-Join
company_profiles = {
    "Company_Name": [
        "Coca-Cola Beverages SA", "PepsiCo SA", "Tiger Brands",
        "Shoprite Group", "Pick n Pay", "Woolworths", "Unilever SA"
    ],
    "Sector": ["Beverage", "Food & Beverage", "FMCG", "Retail", "Retail", "Retail", "Home & Personal Care"],
    "Company_Packaging_Market_Share_Pct": [18.5, 14.2, 12.0, 11.5, 9.0, 6.5, 5.8],
    "Primary_Polymer_Focus": ["PET_1", "LDPE_LLDPE_4", "HDPE_2", "LDPE_LLDPE_4", "LDPE_LLDPE_4", "PET_1", "HDPE_2"],
    "EPR_Target_Recycled_Content_Pct": [30.0, 25.0, 20.0, 35.0, 30.0, 40.0, 25.0],
    "SA_Plastics_Pact_Signatory": [1, 1, 1, 1, 1, 1, 1]
}
df_company = pd.DataFrame(company_profiles)
df_company.to_csv(os.path.join(SCRIPT_DIR, "raw_sa_company_shares.csv"), index=False)

# Cross-join Company Profiles (192 months x 9 provinces x 7 companies = 12,096 rows)
df_merged = df_cons.merge(df_company, how="cross")

# Calculate Company-Specific Packaging & Recycled Tonnes
df_merged["Company_Virgin_Packaging_Tonnes"] = (
    df_merged["Provincial_Virgin_Tonnes"] 
    * (df_merged["Packaging_Share_Percent"] / 100.0) 
    * (df_merged["Company_Packaging_Market_Share_Pct"] / 100.0)
)

df_merged["Company_Recycled_Input_Tonnes"] = (
    df_merged["Provincial_Recycled_Tonnes"] 
    * (df_merged["Company_Packaging_Market_Share_Pct"] / 100.0) 
    * (df_merged["EPR_Target_Recycled_Content_Pct"] / 25.0)
)

# 4. Mismanaged Waste Bounds
pollution_data = {
    "Metric_Category": [
        "Total_Annual_Plastic_Waste_Generated",
        "Estimated_Mismanaged_Plastic_Waste",
        "Land_Based_Plastic_Leakage",
        "Riverine_Coastal_Ocean_Input",
        "Informal_Reclaimer_Collection_Share",
        "Municipal_Waste_Uncollected_Pct"
    ],
    "Value_Lower_Bound": [2300000, 480000, 300000, 15000, 80.0, 31.2],
    "Value_Upper_Bound": [2500000, 720000, 550000, 40000, 90.0, 37.8],
    "Unit": ["Tonnes/Year", "Tonnes/Year", "Tonnes/Year", "Tonnes/Year", "Percent", "Percent"]
}
df_poll = pd.DataFrame(pollution_data)
df_poll.to_csv(os.path.join(SCRIPT_DIR, "raw_sa_pollution.csv"), index=False)

# 5. Polymer Classification Profiles
polymer_data = {
    "Polymer_Code": ["PET_1", "HDPE_2", "PVC_3", "LDPE_LLDPE_4", "PP_5", "PS_EPS_6", "OTHER_7"],
    "Primary_Application": [
        "Beverage_Bottles", "Rigid_Containers", "Pipes_Cables",
        "Films_Bags", "Tubs_Automotive", "Food_Trays_Insulation", "Multi_Layer"
    ],
    "Market_Share_Percent": [22.4, 18.1, 8.5, 24.6, 18.3, 4.2, 3.9],
    "Recycling_Efficiency_Score": [0.85, 0.80, 0.20, 0.65, 0.50, 0.15, 0.05]
}
df_polym = pd.DataFrame(polymer_data)
df_polym.to_csv(os.path.join(SCRIPT_DIR, "raw_sa_polymers.csv"), index=False)

# 6. Beach Litter Trends
beach_litter_data = {
    "Date": pd.to_datetime(["1985-01-01", "1995-01-01", "2005-01-01", "2015-01-01", "2025-01-01"]),
    "Observed_Density_Items_Per_Meter": [12.4, 28.6, 54.2, 89.1, 118.2],
    "Single_Use_Plastic_Ratio_Percent": [45.0, 58.2, 67.5, 74.8, 79.1],
    "Microplastic_Mean_Pellets_M2": [120, 240, 510, 890, 1210]
}
df_beach = pd.DataFrame(beach_litter_data)

# Merge Beach Litter on Date
df_merged = pd.merge(df_merged, df_beach, on="Date", how="left")
df_merged['Observed_Density_Items_Per_Meter'] = df_merged['Observed_Density_Items_Per_Meter'].interpolate(method='linear').bfill()
df_merged['Single_Use_Plastic_Ratio_Percent'] = df_merged['Single_Use_Plastic_Ratio_Percent'].interpolate(method='linear').bfill()
df_merged['Microplastic_Mean_Pellets_M2'] = df_merged['Microplastic_Mean_Pellets_M2'].interpolate(method='linear').bfill()

# 7. Live OWID Dataset Fetch
try:
    owid_url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Plastic%20waste%20by%20country%20-%20OWID/Plastic%20waste%20by%20country%20-%20OWID.csv"
    res = requests.get(owid_url, timeout=10)
    df_owid = pd.read_csv(io.StringIO(res.text))
    df_owid_sa = df_owid[df_owid['Entity'].str.contains('South Africa', case=False, na=False)]
    df_owid_sa.to_csv(os.path.join(SCRIPT_DIR, "raw_sa_owid.csv"), index=False)
except Exception as e:
    print(f"OWID fetch warning: {e}")

# 8. Feature Engineering (Grouped by Province & Company)
df_merged = df_merged.sort_values(by=["Province_Code", "Company_Name", "Date"]).reset_index(drop=True)

df_merged['Company_Mechanical_Recycling_Rate_Pct'] = (
    df_merged['Company_Recycled_Input_Tonnes'] / df_merged['Company_Virgin_Packaging_Tonnes']
) * 100

# Month-over-Month (MoM) Growth per Company in Province
group_cols = ['Province_Code', 'Company_Name']
df_merged['Company_Virgin_MoM_Growth'] = df_merged.groupby(group_cols)['Company_Virgin_Packaging_Tonnes'].pct_change().fillna(0)
df_merged['Company_Recycling_MoM_Growth'] = df_merged.groupby(group_cols)['Company_Recycled_Input_Tonnes'].pct_change().fillna(0)

# Year-over-Year (YoY 12-Month Lag) Growth
df_merged['Company_Virgin_YoY_Growth'] = df_merged.groupby(group_cols)['Company_Virgin_Packaging_Tonnes'].pct_change(12).fillna(0)
df_merged['Company_Recycling_YoY_Growth'] = df_merged.groupby(group_cols)['Company_Recycled_Input_Tonnes'].pct_change(12).fillna(0)

# Company Estimated Unmanaged Leakage Tonnes
df_merged['Company_Estimated_Unmanaged_Leakage_Tonnes'] = (
    (df_merged['Company_Virgin_Packaging_Tonnes'] - df_merged['Company_Recycled_Input_Tonnes'])
    * (df_merged['Municipal_Uncollected_Waste_Pct'] / 100.0)
)

# Overwrite final_sa_plastics_ml.csv
final_csv_path = os.path.join(SCRIPT_DIR, "final_sa_plastics_ml.csv")
df_merged.to_csv(final_csv_path, index=False)

print(f"\nPipeline complete! 12,096 panel records (192 months x 9 provinces x 7 companies) saved to: '{final_csv_path}'")