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
    import matplotlib.font_manager as fm
    import os
    
    # 폰트 파일 경로 리스트 (가능한 경로들)
    font_paths = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
    ]
    
    font_found = False
    for path in font_paths:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_prop = fm.FontProperties(fname=path)
            plt.rc('font', family=font_prop.get_name())
            font_found = True
            print(f"Font found and set: {path}")
            break
            
    if not font_found:
        print("Nanum font not found. Using fallback.")
        plt.rc('font', family='DejaVu Sans')

plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

st.set_page_config(page_title="서울시 물 수요 예측 대시보드", layout="wide")

@st.cache_resource
def load_data_and_model():
    # 데이터 로드 (절대 경로 사용 - 상위 폴더로 이동)
    import os
    
    # 데이터 로드
    import os
    
    # 로그 분석 결과:
    # Current working directory: /mount/src/-dbms-water_demand_forecasting
    # Files in current directory: ['README.md', 'src', 'README_KR.md', '.git', 'docs', '.gitignore', 'results', 'requirements.txt', 'packages.txt', 'data', 'README_EN.md']
    
    # 결론: 작업 디렉토리가 이미 프로젝트 루트입니다.
    # 따라서 'data/anfis_dataset_with_covid.csv'로 바로 접근하면 됩니다.
    
    data_path = 'data/anfis_dataset_with_covid.csv'
    
    # 디버깅용 출력
    print(f"Loading data from: {data_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    if not os.path.exists(data_path):
        # 만약 그래도 없다면 절대 경로로 시도 (비상용)
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        data_path = os.path.join(project_root, 'data', 'anfis_dataset_with_covid.csv')
        print(f"Retry with absolute path: {data_path}")
        
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
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

# 다국어 지원 딕셔너리
TEXT = {
    'KR': {
        'page_title': "서울시 물 수요 예측 대시보드",
        'main_title': "💧 서울시 물 수요 예측 & 정책 시뮬레이터",
        'sidebar_lang': "언어 선택 (Language)",
        'sidebar_scenario': "정책 시나리오 설정",
        'slider_price': "요금 변동률 (%)",
        'sidebar_period': "예측 기간 설정",
        'slider_period': "향후 예측 기간 (년)",
        'subheader_forecast': "향후 {}년 물 수요 예측",
        'plot_title': "장기 물 수요 예측",
        'plot_ylabel': "수요량 (㎥)",
        'legend_actual': "2024년 (실제)",
        'legend_pred': "예측 (요금 {:+.0f}%)",
        'subheader_total': "기간 총 수요 예측",
        'metric_label': "향후 {}년 총 수요",
        'unit_100m': " 억톤",
        'unit_10k': " 만톤",
        'info_title': "**적용된 모델 로직 (Hybrid)**",
        'info_item1': "1. **XGBoost**: 계절/날씨 기반 기본 수요 예측",
        'info_item2': "2. **Econometrics**: 요금 탄력성({}) 적용",
        'info_result': "요금을 **{}%** 조정하면,\n수요는 **{:.2f}%** 변동합니다."
    },
    'EN': {
        'page_title': "Seoul Water Demand Forecasting Dashboard",
        'main_title': "💧 Seoul Water Demand Forecasting & Policy Simulator",
        'sidebar_lang': "Language Selection",
        'sidebar_scenario': "Policy Scenario Settings",
        'slider_price': "Price Change Rate (%)",
        'sidebar_period': "Forecast Period Settings",
        'slider_period': "Forecast Horizon (Years)",
        'subheader_forecast': "Water Demand Forecast for Next {} Years",
        'plot_title': "Long-term Water Demand Forecast",
        'plot_ylabel': "Demand (㎥)",
        'legend_actual': "2024 (Actual)",
        'legend_pred': "Forecast (Price {:+.0f}%)",
        'subheader_total': "Total Demand for Period",
        'metric_label': "Total Demand for Next {} Years",
        'unit_100m': "00M tons",
        'unit_10k': "0k tons",
        'info_title': "**Applied Model Logic (Hybrid)**",
        'info_item1': "1. **XGBoost**: Baseline demand prediction based on seasonality/weather",
        'info_item2': "2. **Econometrics**: Price elasticity ({}) applied",
        'info_result': "Adjusting price by **{}%** results in\na **{:.2f}%** change in demand."
    }
}

def main():
    # 언어 선택 (기본값: KR)
    if 'lang' not in st.session_state:
        st.session_state.lang = 'KR'
        
    st.set_page_config(page_title=TEXT[st.session_state.lang]['page_title'], layout="wide")
    
    # 사이드바에서 언어 선택
    lang_option = st.sidebar.selectbox(
        TEXT[st.session_state.lang]['sidebar_lang'],
        ('한국어 (Korean)', 'English'),
        index=0 if st.session_state.lang == 'KR' else 1
    )
    
    if lang_option == '한국어 (Korean)':
        st.session_state.lang = 'KR'
    else:
        st.session_state.lang = 'EN'
        
    t = TEXT[st.session_state.lang] # 현재 언어 팩
    
    st.title(t['main_title'])
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header(t['sidebar_scenario'])
    price_change = st.sidebar.slider(t['slider_price'], -30, 30, 0, 5)
    
    st.sidebar.header(t['sidebar_period'])
    forecast_years = st.sidebar.slider(t['slider_period'], 1, 10, 1)
    
    # 모델 로드
    model, scaler_X, df, feature_cols = load_data_and_model()
    
    # 탄력성 (Hybrid Model Logic)
    elasticity = -0.0621
    
    # 미래 데이터 생성 (2024년 데이터 반복 사용 - 단순 가정)
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
        st.subheader(t['subheader_forecast'].format(forecast_years))
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 2024년 실제값 (참고용)
        ax.plot(last_year_data.index, last_year_data['Current_Demand'], 
                'k:', label=t['legend_actual'], alpha=0.5)
        
        # 미래 예측값
        ax.plot(future_df.index, final_demand, 
                'r-', label=t['legend_pred'].format(price_change), linewidth=2)
        
        ax.set_title(t['plot_title'])
        ax.set_ylabel(t['plot_ylabel'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with col2:
        st.subheader(t['subheader_total'])
        total_demand = final_demand.sum()
        base_total = base_demand_pred.sum()
        
        delta = total_demand - base_total
        delta_pct = (delta / base_total) * 100
        
        st.metric(
            label=t['metric_label'].format(forecast_years),
            value=f"{total_demand/1e8:.2f}{t['unit_100m']}",
            delta=f"{delta/1e4:,.0f}{t['unit_10k']} ({delta_pct:+.2f}%)",
            delta_color="inverse"
        )
        
        st.info(f"""
        {t['info_title']}
        {t['info_item1']}
        {t['info_item2'].format(elasticity)}
        
        {t['info_result'].format(price_change, demand_change_pct*100)}
        """)

if __name__ == "__main__":
    main()
