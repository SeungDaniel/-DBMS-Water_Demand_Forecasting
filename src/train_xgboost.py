# -*- coding: utf-8 -*-
"""
src/train_xgboost.py - XGBoost 모델 학습 및 비교 분석
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def evaluate(y_true, y_pred, label):
    """성능 평가"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n[{label}] Performance:")
    print(f"  RMSE: {rmse:,.0f}")
    print(f"  MAE:  {mae:,.0f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2}

def run_scenario_analysis(model, scaler_X, df, feature_cols):
    """시나리오 분석 (요금 탄력성 테스트)"""
    print("\n[Scenario Analysis] Price Elasticity Test (2025)")
    
    # 2024년 12월 데이터 가져오기
    last_row = df.iloc[[-1]].copy()
    base_demand = last_row['Current_Demand'].values[0]
    
    scenarios = {
        'Status Quo (0%)': 0.0,
        'Price Up (+10%)': 0.10,
        'Price Down (-10%)': -0.10
    }
    
    print(f"  Base Demand (2024-12): {base_demand:,.0f}")
    
    for name, rate in scenarios.items():
        # 데이터 복사
        scenario_row = last_row.copy()
        
        # 요금 조정
        scenario_row['effective_fee_adjusted'] = scenario_row['effective_fee_adjusted'] * (1 + rate)
        
        # 예측
        X_scenario = scaler_X.transform(scenario_row[feature_cols])
        y_pred = model.predict(X_scenario)[0]
        
        change_pct = (y_pred - base_demand) / base_demand * 100
        print(f"  {name}: {y_pred:,.0f} ({change_pct:+.2f}%)")

def main():
    print("=" * 60)
    print("XGBoost Model Training & Evaluation")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1] Loading data...")
    df = pd.read_csv('../data/anfis_dataset_with_covid.csv', index_col=0, parse_dates=True)
    
    feature_cols = ['Temperature', 'Precipitation', 'population_norm', 
                   'Prev_Demand', 'effective_fee_adjusted', 'month_sin', 'month_cos']
    
    df = df.dropna(subset=feature_cols + ['Current_Demand'])
    
    # 2. Train/Test Split
    train_mask = df.index.year <= 2022
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()
    
    print(f"Training: {len(df_train)} months, Test: {len(df_test)} months")
    
    # 3. Scaling (XGBoost는 스케일링이 필수는 아니지만, 비교를 위해 동일하게 적용)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(df_train[feature_cols])
    y_train = df_train['Current_Demand'].values
    
    X_test = scaler_X.transform(df_test[feature_cols])
    y_test = df_test['Current_Demand'].values
    
    # 4. Model Training
    print("\n[2] Training XGBoost...")
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluation
    print("\n[3] Evaluation...")
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    
    evaluate(y_train, y_pred_tr, "Training Set")
    evaluate(y_test, y_pred_te, "Test Set")
    
    # 6. Feature Importance
    print("\n[4] Feature Importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance)
    
    # 7. Scenario Analysis
    run_scenario_analysis(model, scaler_X, df, feature_cols)
    
    # 8. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df_train.index, y_train, 'b-', label='Train True')
    plt.plot(df_train.index, y_pred_tr, 'b--', alpha=0.5, label='Train Pred')
    plt.plot(df_test.index, y_test, 'r-', label='Test True')
    plt.plot(df_test.index, y_pred_te, 'r--', alpha=0.5, label='Test Pred')
    plt.title('XGBoost Prediction Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/xgboost_results.png')
    print("\n[OK] Graph saved: ../results/xgboost_results.png")

if __name__ == "__main__":
    main()
