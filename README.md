# P-Life: Plastic Pollution Impact Modeling

A personal data science project that simulates and forecasts the environmental impact of **P-Life** — a hypothetical bio-additive that accelerates microbial biodegradation of polyolefin plastics (PP/PE) — on plastic pollution leakage in South Africa and Japan.

The project builds synthetic, feature-rich monthly panel datasets (province/company level for South Africa, sector level for Japan), trains regression models to predict "net persistent environmental leakage" under different P-Life adoption scenarios, and renders board-style dashboards summarizing the projected pollution abatement.

> **Note on the data:** all consumption, pollution, company market-share, and biodegradation figures in this repo are **synthetically generated** for modeling purposes (hardcoded baselines + seasonality/growth curves + cross-joins), not sourced from real corporate disclosures, government statistics, or lab studies. Company names (Coca-Cola Beverages SA, Shoprite, Pick n Pay, ITO EN, etc.) and standards references (JIS K6955, ISO 17556, Keio University field trials) are used as realistic scenario labels, not verified data sources.

## What it does

1. **Builds synthetic datasets**
   - `South_African_Data/build_sa_plastic_database.py` — generates a 2010–2025 monthly panel (192 months × 9 provinces × 7 major SA companies) covering virgin/recycled plastic tonnage, provincial recycling efficiency, mismanaged waste bounds, polymer mix, and beach-litter trends. Also pulls a live reference dataset from [Our World in Data](https://github.com/owid/owid-datasets) for South Africa.
   - `International_Data_Implemented_P-Life/build_international_p_life_database.py` — generates a comparable 2010–2025 monthly panel for Japan (192 months × 3 sectors), plus lab-style biodegradation benchmark data.

2. **Trains predictive models** (`ML_Model/train_sa_p_life_prediction_model.py`)
   - Simulates a counterfactual adoption matrix (0%–100% P-Life adoption) over the SA dataset.
   - Trains a `GradientBoostingRegressor` to predict net persistent leakage and a `RandomForestRegressor` to predict percentage pollution reduction, with one-hot encoded company/province features.
   - Runs sample scenario predictions for named companies and saves full predictions to CSV.

3. **Evaluates model quality** (`ML_Model/evaluate_model_board_metrics.py`) — R², MAE, RMSE, and 5-fold cross-validated R² for the leakage model.

4. **Forecasts and visualizes** — several scripts turn the trained model's output into `matplotlib` dashboards:
   - `generate_forecast_dashboard.py` — retail vs. FMCG sector pollution trajectories at a 50% adoption scenario.
   - `generate_board_accuracy_dashboard.py` — an executive-style model accuracy/reliability summary.
   - `predict_sa_2year_forecast.py` + `generate_japan_forecast_dashboard.py` — a 24-month (Sept 2026–Aug 2028) forward forecast for SA companies, and a historical (2010–2025) decline chart for Japan.

## Repository layout

```
South_African_Data/                     Synthetic SA plastics dataset + generator script
International_Data_Implemented_P-Life/  Synthetic Japan dataset + generator script
ML_Model/                                Model training, evaluation, forecasting, and dashboard scripts
```

## Tech stack

- **Python** 3
- **pandas** / **numpy** — data generation and feature engineering
- **scikit-learn** — `GradientBoostingRegressor`, `RandomForestRegressor`, `OneHotEncoder`, cross-validation
- **matplotlib** — dashboard/report generation
- **requests** — fetching the OWID reference dataset

## Setup & running

No `requirements.txt` is included yet; install the dependencies directly:

```bash
pip install pandas numpy scikit-learn matplotlib requests
```

Then run the pipeline in order (each script reads/writes CSVs relative to its own folder):

```bash
# 1. Build the synthetic datasets
python South_African_Data/build_sa_plastic_database.py
python International_Data_Implemented_P-Life/build_international_p_life_database.py

# 2. Train the SA prediction model (writes ML_Model/sa_p_life_impact_predictions.csv)
python ML_Model/train_sa_p_life_prediction_model.py

# 3. Evaluate model accuracy
python ML_Model/evaluate_model_board_metrics.py

# 4. Generate dashboards (each writes a .png into ML_Model/)
python ML_Model/generate_forecast_dashboard.py
python ML_Model/generate_board_accuracy_dashboard.py
python ML_Model/predict_sa_2year_forecast.py
python ML_Model/generate_japan_forecast_dashboard.py
```

No environment variables or credentials are required — the only network call is an unauthenticated fetch of a public CSV from the OWID GitHub dataset repo.

## Project status

This is a personal/portfolio data science project exploring simulation and forecasting techniques for environmental impact modeling. It is not connected to a real product, company, or dataset provider.
