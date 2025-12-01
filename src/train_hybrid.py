# -*- coding: utf-8 -*-
"""
src/train_hybrid.py - Hybrid Model (XGBoost + Elasticity) Implementation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def calculate_elasticity(df):
    """경제학적 모델(Log-Log Regression)로 가격 탄력성 계산"""
    print("\n[1] Calculating Price Elasticity...")
    
    # Log 변환 (탄력성 계산을 위해)
    # ln(Demand) = a + b * ln(Price) + c * ln(Temp) ...
    # 여기서 b가 바로 탄력성(Elasticity)임
    
    df_log = df.copy()
    # 0이나 음수가 있을 수 있는 컬럼 처리 (Temp는 섭씨라 음수 가능 -> Kelvin으로 변환하거나 그냥 선형으로 둠)
    # 여기서는 간단히 Price와 Demand만 로그 변환하여 분석
    
    df_log['ln_demand'] = np.log(df_log['Current_Demand'])
    df_log['ln_price'] = np.log(df_log['effective_fee_adjusted'])
    
    # 통제 변수 (계절성 등) 추가
    X = df_log[['ln_price', 'Temperature', 'Precipitation', 'month_sin', 'month_cos']]
    X = sm.add_constant(X)
    y = df_log['ln_demand']
    
    model = sm.OLS(y, X).fit()
    
    elasticity = model.params['ln_price']
    p_value = model.pvalues['ln_price']
    
    print(model.summary())
    print(f"\n>>> Calculated Price Elasticity: {elasticity:.4f} (P-value: {p_value:.4f})")
    
    # 탄력성이 양수(비정상)거나 통계적으로 유의하지 않으면, 
    # 문헌값(서울시 연구보고서 등)을 차용하거나 0으로 가정해야 함.
    # 여기서는 만약 양수라면 -0.1 (일반적인 상수도 탄력성)로 강제 조정하는 로직 추가 가능.
    
    if elasticity > 0:
        print(">>> Warning: Positive elasticity detected. Forcing to -0.1 (Literature value).")
        elasticity = -0.1
        
    return elasticity

def train_xgboost(df_train, df_test, feature_cols):
    """XGBoost 모델 학습 (Base Demand 예측용)"""
    print("\n[2] Training XGBoost for Base Demand...")
    
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(df_train[feature_cols])
    y_train = df_train['Current_Demand'].values
    
    X_test = scaler_X.transform(df_test[feature_cols])
    y_test = df_test['Current_Demand'].values
    
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
    
    # 평가
    y_pred_te = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_te)
    print(f"  XGBoost Test R²: {r2:.4f}")
    
    return model, scaler_X

def run_hybrid_scenario(xgb_model, scaler_X, elasticity, df, feature_cols):
    """하이브리드 시나리오 분석"""
    print("\n[3] Hybrid Scenario Analysis (2025)...")
    
    # 2024년 12월 데이터 기준
    last_row = df.iloc[[-1]].copy()
    base_price = last_row['effective_fee_adjusted'].values[0]
    
    # XGBoost가 예측한 2025년 1월의 Base Demand (가격 변동 없음 가정)
    # 실제로는 2025년 전체를 해야 하지만, 여기서는 1개 시점만 예시로 보여줌
    X_base = scaler_X.transform(last_row[feature_cols])
    base_demand_pred = xgb_model.predict(X_base)[0]
    
    scenarios = {
        'Status Quo (0%)': 0.0,
        'Price Up (+10%)': 0.10,
        'Price Down (-10%)': -0.10
    }
    
    print(f"  Base Demand (XGBoost Pred): {base_demand_pred:,.0f}")
    print(f"  Applied Elasticity: {elasticity:.4f}")
    
    results = {}
    
    for name, rate in scenarios.items():
        # Hybrid Formula:
        # Final Demand = Base Demand * (1 + Elasticity * Price Change %)
        
        demand_change_pct = elasticity * rate
        final_demand = base_demand_pred * (1 + demand_change_pct)
        
        results[name] = final_demand
        print(f"  {name}: {final_demand:,.0f} (Change: {demand_change_pct*100:+.2f}%)")
        
    return results

def main():
    print("=" * 60)
    print("Hybrid Model (XGBoost + Econometrics)")
    print("=" * 60)
    
    # 1. 데이터 로드
    df = pd.read_csv('../data/anfis_dataset_with_covid.csv', index_col=0, parse_dates=True)
    feature_cols = ['Temperature', 'Precipitation', 'population_norm', 
                   'Prev_Demand', 'effective_fee_adjusted', 'month_sin', 'month_cos']
    df = df.dropna(subset=feature_cols + ['Current_Demand'])
    
    # 2. 탄력성 계산 (전체 데이터 사용)
    elasticity = calculate_elasticity(df)
    
    # 3. XGBoost 학습
    train_mask = df.index.year <= 2022
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()
    
    xgb_model, scaler_X = train_xgboost(df_train, df_test, feature_cols)
    
    # 4. 하이브리드 시나리오 실행
    run_hybrid_scenario(xgb_model, scaler_X, elasticity, df, feature_cols)

if __name__ == "__main__":
    main()
