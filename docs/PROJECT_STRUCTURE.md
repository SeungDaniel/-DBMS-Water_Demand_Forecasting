# Project Structure

This document explains the organization of the Seoul Water Usage Prediction project.

## Directory Overview

- **`src/`**: Source code for data processing, model training, and the Streamlit dashboard.
- **`data/`**: Raw and processed datasets.
- **`results/`**: Generated plots, reports, and prediction files.
- **`docs/`**: Project documentation and reports.

## Key Files

### `src/` (Source Code)
- **`app.py`**: The main Streamlit dashboard application. Run with `streamlit run src/app.py`.
- **`data_prep.py`**: Handles data loading, cleaning, and feature engineering (including COVID discount logic).
- **`train.py`**: Implementation of the ANFIS model.
- **`train_xgboost.py`**: Implementation of the XGBoost model for comparison.
- **`train_hybrid.py`**: Implementation of the Hybrid Model (XGBoost + OLS Elasticity).
- **`scenario_analysis.py`**: Script for running price change scenarios.

### `data/` (Data)
- **`anfis_dataset_with_covid.csv`**: The final integrated dataset used for training and testing.
- *Original raw files*: `demand_monthly.csv`, `ComTempPrec.xlsx`, etc.

### `results/` (Outputs)
- **`xgboost_results.png`**: Visualization of XGBoost model performance.
- **`scenario_analysis_2025.png`**: Plot showing demand predictions under different price scenarios.
- **`anfis_results_with_covid.png`**: Visualization of the tuned ANFIS model performance.

### `docs/` (Documentation)
- **`walkthrough.md`**: Detailed English report covering the project workflow, model tuning, and final hybrid solution.
- **`walkthrough_KR.md`**: Korean version of the walkthrough report.
- **`PROJECT_SUMMARY.md`**: High-level summary of the project goals and initial analysis.

## How to Run
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run Dashboard**: `streamlit run src/app.py`
