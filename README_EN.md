# Water Demand Forecasting System - ANFIS Model

## Project Overview

ANFIS-based water demand forecasting system for Seoul's monthly water usage.

### Key Features
- **Model**: ANFIS (Gaussian Membership Functions + Sugeno consequents)
- **Input Features**: Temperature, Precipitation, Population, Previous Demand, Price Level, Seasonality
- **Target**: Monthly Water Usage (㎥)
- **Data Period**: 2018-2024 (84 months)

## Performance

### Training Set (2018-2022)
- **RMSE**: 222,528 ㎥
- **MAE**: 115,377 ㎥
- **R²**: 0.9996 (99.96%)
- **MAPE**: 0.13%

### Test Set (2023-2024)
- **RMSE**: 6,201,036 ㎥
- **MAE**: 5,478,672 ㎥
- **R²**: 0.6828 (68.28%)
- **MAPE**: 6.13%

## How to Run

### 1. Data Analysis
```bash
cd scripts
python 01_analyze_data.py
```

### 2. Data Integration
```bash
python 02_integrate_data.py
```

### 3. Model Training
```bash
python 03_train_anfis.py
```

### 4. Generate Report
```bash
python 04_generate_report.py
```

## File Structure

```
anfis_water_forecasting/
├── README.md / README_EN.md
├── scripts/
│   ├── 01_analyze_data.py      # Data analysis
│   ├── 02_integrate_data.py     # Data integration
│   ├── 03_train_anfis.py        # Model training
│   └── 04_generate_report.py    # Report generation
├── data/
│   ├── anfis_dataset.csv        # Integrated dataset
│   └── anfis_predictions.csv    # Prediction results
└── results/
    ├── data_analysis.png        # Data analysis plots
    ├── anfis_results.png        # Model results
    ├── policy_analysis.png      # Policy analysis plots
    └── policy_report.txt        # Analysis report
```

## Key Findings

### Price Policy Impact
- **Price Increases**: 2018.7(+3.5%), 2021.7(+4.4%), 2022.1(+16.1%), 2023.1(+14.2%)
- **Price Elasticity**: -0.064 (inelastic)
- **Demand Change**: -2.51% decrease when price increased

### Seasonality Pattern
- **Summer Peak (Jul-Aug)**: 108,749,600 ㎥
- **Winter Low (Jan-Feb)**: 76,182,650 ㎥

### COVID-19 Impact
- COVID period demand: -0.2% decrease
- Minimal effect

## Requirements

```
numpy>=1.20.0
pandas>=1.3.0
torch>=1.9.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Installation

```bash
pip install -r requirements.txt
```

## License

MIT License

## Citation

Based on: Forecasting municipal water demands (EPCOR, 2019)


