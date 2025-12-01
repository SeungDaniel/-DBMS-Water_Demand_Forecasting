import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# 한글 폰트 설정
import platform
system_name = platform.system()

if system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
elif system_name == 'Windows': # Windows
    plt.rc('font', family='Malgun Gothic')
else: # Linux (Streamlit Cloud)
    # 리눅스에서는 나눔글꼴 설치가 필요할 수 있음
    # 우분투: sudo apt-get install fonts-nanum
    # Streamlit Cloud: packages.txt에 fonts-nanum 추가
    try:
        plt.rc('font', family='NanumGothic')
    except:
        plt.rc('font', family='DejaVu Sans') # Fallback

plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

st.set_page_config(page_title="서울시 물 수요 예측 대시보드", layout="wide")

@st.cache_resource
def load_data_and_model():
    # 데이터 로드
    df = pd.read_csv('data/anfis_dataset_with_covid.csv', index_col=0, parse_dates=True)
    feature_cols = ['Temperature', 'Precipitation', 'population_norm', 
                   'Prev_Demand', 'effective_fee_adjusted', 'month_sin', 'month_cos']
    df = df.dropna(subset=feature_cols + ['Current_Demand'])
    
    # Train/Test Split (2022년까지 학습)
    train_mask = df.index.year <= 2022
    df_train = df[train_mask].copy()
    
    # Scaler 학습
    scaler_X = StandardScaler()
    scaler_X.fit(df_train[feature_cols])
    
    # XGBoost 모델 학습 (미리 저장된 모델을 불러오는 게 좋지만, 여기선 즉석 학습)
    X_train = scaler_X.transform(df_train[feature_cols])
    y_train = df_train['Current_Demand'].values
    
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
    
    return model, scaler_X, df, feature_cols

def main():
    st.title("💧 서울시 물 수요 예측 & 정책 시뮬레이터")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("정책 시나리오 설정")
    price_change = st.sidebar.slider("요금 변동률 (%)", -30, 30, 0, 5)
    
    st.sidebar.header("예측 기간 설정")
    forecast_years = st.sidebar.slider("향후 예측 기간 (년)", 1, 10, 1)
    
    # 모델 로드
    model, scaler_X, df, feature_cols = load_data_and_model()
    
    # 탄력성 (Hybrid Model Logic)
    elasticity = -0.0621
    
    # 미래 데이터 생성 (2024년 데이터 반복 사용 - 단순 가정)
    # 실제로는 기후 변화 시나리오 등을 반영해야 하지만, 여기서는 패턴 반복으로 가정
    last_year_data = df[df.index.year == 2024].copy()
    future_data_list = []
    
    for i in range(1, forecast_years + 1):
        temp_df = last_year_data.copy()
        temp_df.index = temp_df.index + pd.DateOffset(years=i)
        temp_df['year'] = 2024 + i
        future_data_list.append(temp_df)
        
    future_df = pd.concat(future_data_list)
    
    # Base Demand 예측 (XGBoost)
    X_base = scaler_X.transform(future_df[feature_cols])
    base_demand_pred = model.predict(X_base)
    
    # Hybrid Demand 계산
    demand_change_pct = elasticity * (price_change / 100)
    final_demand = base_demand_pred * (1 + demand_change_pct)
    
    # 결과 시각화
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"향후 {forecast_years}년 물 수요 예측")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 2024년 실제값 (참고용)
        ax.plot(last_year_data.index, last_year_data['Current_Demand'], 
                'k:', label='2024년 (실제)', alpha=0.5)
        
        # 미래 예측값
        ax.plot(future_df.index, final_demand, 
                'r-', label=f'예측 (요금 {price_change:+.0f}%)', linewidth=2)
        
        ax.set_title("장기 물 수요 예측")
        ax.set_ylabel("수요량 (㎥)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with col2:
        st.subheader("기간 총 수요 예측")
        total_demand = final_demand.sum()
        base_total = base_demand_pred.sum()
        
        delta = total_demand - base_total
        delta_pct = (delta / base_total) * 100
        
        st.metric(
            label=f"향후 {forecast_years}년 총 수요",
            value=f"{total_demand/1e8:.2f} 억톤",
            delta=f"{delta/1e4:,.0f} 만톤 ({delta_pct:+.2f}%)",
            delta_color="inverse"
        )
        
        st.info(f"""
        **적용된 모델 로직 (Hybrid)**
        1. **XGBoost**: 계절/날씨 기반 기본 수요 예측
        2. **Econometrics**: 요금 탄력성({elasticity}) 적용
        
        요금을 **{price_change}%** 조정하면,
        수요는 **{demand_change_pct*100:.2f}%** 변동합니다.
        """)

if __name__ == "__main__":
    main()
