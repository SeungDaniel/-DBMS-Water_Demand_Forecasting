# Seoul Water Usage Prediction Project

## Overview
This project aims to predict monthly water demand in Seoul using advanced machine learning models. It addresses the challenge of balancing predictive accuracy with economic interpretability, especially in the context of policy simulations (e.g., price changes).

## Key Features
- **Hybrid Model**: Combines **XGBoost** (for high accuracy) and **OLS Regression** (for price elasticity) to ensure both precise predictions and logical policy analysis.
- **Scenario Analysis**: Simulates future water demand under various price change scenarios (e.g., +10%, -10%).
- **Interactive Dashboard**: A Streamlit-based web app for real-time visualization and simulation.

## Project Structure
- `src/`: Source code for models and dashboard.
- `data/`: Datasets (including COVID-19 discount logic).
- `results/`: Generated plots and reports.
- `docs/`: Detailed documentation and reports.

## Model Performance
| Model | Test R² | Test RMSE | Policy Sensitivity |
| :--- | :--- | :--- | :--- |
| **ANFIS** | 0.68 | 6.2M ㎥ | Unstable |
| **XGBoost** | **0.93** | **3.0M ㎥** | None (0%) |
| **Hybrid** | **0.93** | **3.0M ㎥** | **Correct (-0.062)** |

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SeungDaniel/-DBMS-Water_Demand_Forecasting.git
   cd -DBMS-Water_Demand_Forecasting
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Dashboard**:
   ```bash
   streamlit run src/app.py
   ```

## Documentation
For more details, please refer to the `docs/` folder:
- [Walkthrough Report (Eng)](docs/walkthrough.md)
- [Project Structure (Eng)](docs/PROJECT_STRUCTURE.md)
