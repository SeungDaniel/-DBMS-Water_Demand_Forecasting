# -*- coding: utf-8 -*-
"""
src/scenario_analysis.py - 시나리오별 물 수요 예측 분석
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from train import ANFIS, GaussianMF, init_rules  # train.py에서 모델 클래스 재사용

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_data():
    """학습된 모델과 데이터 로드"""
    print("[1] 모델 및 데이터 로드...")
    
    # 데이터 로드 (스케일러 적합을 위해 전체 데이터 필요)
    df = pd.read_csv('../data/anfis_dataset_with_covid.csv', index_col=0, parse_dates=True)
    feature_cols = ['Temperature', 'Precipitation', 'population_norm', 
                   'Prev_Demand', 'effective_fee_adjusted', 'month_sin', 'month_cos']
    df = df.dropna(subset=feature_cols + ['Current_Demand'])
    
    # 스케일러 학습
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # 2022년까지 데이터로 스케일러 학습 (Train set 기준)
    train_mask = df.index.year <= 2022
    df_train = df[train_mask]
    
    scaler_X.fit(df_train[feature_cols])
    scaler_y.fit(df_train[['Current_Demand']])
    
    # 모델 재구성 및 학습 (저장된 가중치가 없으므로 다시 학습 - 실제로는 저장된 .pth를 불러와야 함)
    # 여기서는 편의상 train.py의 로직을 빌려 빠르게 재학습하거나, 
    # train.py에서 모델을 저장하도록 수정하고 불러오는 것이 정석임.
    # 시간 관계상 train.py를 import해서 재학습 함수를 호출하는 방식으로 진행.
    from train import train_model
    
    X_train = scaler_X.transform(df_train[feature_cols])
    y_train = scaler_y.transform(df_train[['Current_Demand']]).ravel()
    
    # Test set (2023-2024)
    df_test = df[~train_mask]
    X_test = scaler_X.transform(df_test[feature_cols])
    y_test = scaler_y.transform(df_test[['Current_Demand']]).ravel()
    
    print("  모델 재학습 중 (규칙=6, Epochs=1000)...")
    model, _, _ = train_model(X_train, y_train, X_test, y_test, n_rules=6, epochs=1000, lr=0.01)
    
    return model, scaler_X, scaler_y, df, feature_cols

def run_scenarios(model, scaler_X, scaler_y, df, feature_cols):
    """2025년 시나리오 분석"""
    print("\n[2] 2025년 시나리오 분석 시작...")
    
    # 2025년 가상 데이터 생성 (2024년 데이터 기반)
    last_year_data = df[df.index.year == 2024].copy()
    base_2025 = last_year_data.copy()
    base_2025.index = base_2025.index + pd.DateOffset(years=1)
    base_2025['year'] = 2025
    
    # 시나리오 정의: 요금 변동률
    scenarios = {
        '현상 유지 (0%)': 0.0,
        '요금 인상 (+10%)': 0.10,
        '요금 인상 (+20%)': 0.20,
        '요금 인하 (-10%)': -0.10
    }
    
    results = {}
    
    for name, rate in scenarios.items():
        # 시나리오 데이터 준비 (기본 데이터 복사)
        scenario_df = base_2025.copy()
        
        # 요금 조정
        scenario_df['effective_fee_adjusted'] = scenario_df['effective_fee_adjusted'] * (1 + rate)
        
        # 결과 저장용 리스트
        y_preds = []
        
        # 초기 Prev_Demand 설정 (2024년 12월의 실제 수요)
        # 주의: 2024년 12월 데이터가 df에 있어야 함
        last_demand = df.loc[df.index[-1], 'Current_Demand']
        
        # 월별 순차 예측 (Recursive Prediction)
        for i in range(len(scenario_df)):
            # 현재 월의 입력 데이터 가져오기
            current_row = scenario_df.iloc[[i]].copy()
            
            # Prev_Demand 업데이트
            current_row['Prev_Demand'] = last_demand
            
            # 전처리
            X_scenario = scaler_X.transform(current_row[feature_cols])
            X_scenario_t = torch.tensor(X_scenario, dtype=torch.float32, device=device)
            
            # 예측
            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_scenario_t).cpu().numpy()
            
            # 역정규화
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).item()
            y_preds.append(y_pred)
            
            # 다음 달을 위해 last_demand 업데이트
            last_demand = y_pred
            
        results[name] = np.array(y_preds)
        
    return results, base_2025.index

def visualize_scenarios(results, dates):
    """시나리오 결과 시각화"""
    print("\n[3] 결과 시각화...")
    
    plt.figure(figsize=(12, 6))
    
    colors = ['black', 'red', 'darkred', 'blue']
    styles = ['-', '--', ':', '-.']
    
    for (name, pred), color, style in zip(results.items(), colors, styles):
        total_demand = pred.sum()
        plt.plot(dates, pred, label=f"{name} (연간: {total_demand/1e6:.1f}백만㎥)", 
                 color=color, linestyle=style, linewidth=2)
    
    plt.title('2025년 요금 시나리오별 물 수요 예측', fontsize=14)
    plt.xlabel('날짜')
    plt.ylabel('예측 수요 (㎥)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = '../results/scenario_analysis_2025.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  그래프 저장 완료: {save_path}")

def main():
    model, scaler_X, scaler_y, df, feature_cols = load_model_and_data()
    results, dates = run_scenarios(model, scaler_X, scaler_y, df, feature_cols)
    visualize_scenarios(results, dates)
    
    # 결과 요약 출력
    print("\n[4] 시나리오별 연간 총 수요 예측:")
    base_demand = results['현상 유지 (0%)'].sum()
    
    for name, pred in results.items():
        total = pred.sum()
        diff = total - base_demand
        pct = (diff / base_demand) * 100
        print(f"  {name}: {total:,.0f} ㎥ ({pct:+.2f}%)")

if __name__ == "__main__":
    main()
