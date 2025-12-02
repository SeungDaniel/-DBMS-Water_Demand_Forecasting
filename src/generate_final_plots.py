import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os

# Set style for professional presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300

def load_and_train():
    # Load Data
    data_path = 'data/anfis_dataset_with_covid.csv'
    if not os.path.exists(data_path):
        # Fallback for different CWD
        data_path = '../data/anfis_dataset_with_covid.csv'
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    feature_cols = ['Temperature', 'Precipitation', 'population_norm', 
                   'Prev_Demand', 'effective_fee_adjusted', 'month_sin', 'month_cos']
    df = df.dropna(subset=feature_cols + ['Current_Demand'])
    
    # Train/Test Split (Train until 2022)
    train_mask = df.index.year <= 2022
    df_train = df[train_mask].copy()
    
    # Scaler
    scaler_X = StandardScaler()
    scaler_X.fit(df_train[feature_cols])
    
    # Train XGBoost
    X_train = scaler_X.transform(df_train[feature_cols])
    y_train = df_train['Current_Demand'].values
    
    model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model, scaler_X, df, feature_cols

def plot_long_term_forecast(model, scaler_X, df, feature_cols):
    print("Generating Long-term Forecast Plot...")
    
    # Generate Future Data (2025-2034)
    last_year_data = df[df.index.year == 2024].copy()
    future_data_list = []
    
    for i in range(1, 11): # 10 years
        temp_df = last_year_data.copy()
        temp_df.index = temp_df.index + pd.DateOffset(years=i)
        temp_df['year'] = 2024 + i
        future_data_list.append(temp_df)
        
    future_df = pd.concat(future_data_list)
    
    # Predict
    X_base = scaler_X.transform(future_df[feature_cols])
    pred = model.predict(X_base)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 2 years for context)
    history_df = df[df.index.year >= 2023]
    plt.plot(history_df.index, history_df['Current_Demand'], 'k-', label='Historical (2023-2024)', alpha=0.6)
    
    # Plot forecast
    plt.plot(future_df.index, pred, 'b-', label='Forecast (2025-2034)', linewidth=2)
    
    # Styling
    plt.title('Seoul Water Demand Forecast (2025-2034)', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Water Demand ($m^3$)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Highlight the trend
    z = np.polyfit(range(len(pred)), pred, 1)
    p = np.poly1d(z)
    plt.plot(future_df.index, p(range(len(pred))), "r--", alpha=0.8, label='Trend Line')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/Long_Term_Forecast_2025_2034.png')
    plt.close()

def plot_model_comparison():
    print("Generating Model Comparison Plot...")
    
    # Data from walkthrough.md
    models = ['ANFIS', 'XGBoost', 'Hybrid (Final)']
    r2_scores = [0.68, 0.93, 0.93]
    rmse_scores = [6.2, 3.0, 3.0] # Million tons
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Bar 1: R2 Score
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='Test R² (Accuracy)', color='#2ecc71', alpha=0.8)
    ax1.set_ylabel('R² Score', fontsize=12, color='#2ecc71', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    
    # Bar 2: RMSE
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label='Test RMSE (Error)', color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('RMSE (Million $m^3$)', fontsize=12, color='#e74c3c', fontweight='bold')
    ax2.set_ylim(0, 8)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison: Accuracy vs Error', fontsize=16, pad=20, fontweight='bold')
    
    # Add value labels
    def add_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    add_labels(bars1, ax1)
    add_labels(bars2, ax2)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/Model_Performance_Comparison.png')
    plt.close()

def plot_scenario_response():
    print("Generating Scenario Response Plot...")
    
    # Scenario: Price +10%
    models = ['ANFIS', 'XGBoost', 'Hybrid (Final)']
    changes = [7.8, 0.0, -0.62] # % Change in Demand
    colors = ['#e74c3c', '#95a5a6', '#3498db'] # Red (Wrong), Grey (None), Blue (Correct)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, changes, color=colors, alpha=0.8, width=0.6)
    
    # Add baseline line
    plt.axhline(0, color='black', linewidth=1)
    
    # Styling
    plt.title('Demand Response to +10% Price Increase', fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('% Change in Demand', fontsize=12)
    plt.ylim(-2, 10)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height >=0 else -0.5),
                 f'{height:+.2f}%',
                 ha='center', va='bottom' if height >=0 else 'top', 
                 fontsize=12, fontweight='bold')
        
    # Annotations
    plt.text(0, 8.5, 'Extrapolation Failure\n(Illogical Increase)', ha='center', color='#c0392b', fontsize=10)
    plt.text(1, 1.0, 'Insensitive\n(No Response)', ha='center', color='#7f8c8d', fontsize=10)
    plt.text(2, -1.5, 'Economic Logic\n(Law of Demand)', ha='center', color='#2980b9', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/Scenario_Response_Comparison.png')
    plt.close()

if __name__ == "__main__":
    # 1. Train & Forecast
    model, scaler_X, df, feature_cols = load_and_train()
    plot_long_term_forecast(model, scaler_X, df, feature_cols)
    
    # 2. Comparisons
    plot_model_comparison()
    plot_scenario_response()
    
    print("All plots generated in results/ folder.")
