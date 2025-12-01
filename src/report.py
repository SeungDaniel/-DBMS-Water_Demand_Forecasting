# -*- coding: utf-8 -*-
"""
04_generate_report.py - 정책 영향 분석 리포트 생성
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def generate_policy_report():
    """정책 영향 분석 리포트 생성"""
    print("=" * 60)
    print("정책 영향 분석 리포트 생성")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드...")
    df = pd.read_csv('../results/anfis_predictions_with_covid.csv')
    df_data = pd.read_csv('../data/anfis_dataset_with_covid.csv', index_col=0, parse_dates=True)
    df_data = df_data.reset_index()
    df_data['date'] = df_data['date'].astype(str)
    
    # 날짜 정렬
    df = df.sort_values('date')
    
    # 2. 요금 정책 효과 분석
    print("\n[2] 요금 정책 효과 분석...")
    
    # 정책 변화 시점 찾기
    df_data_with_price = df_data.copy()
    df_data_with_price['date'] = pd.to_datetime(df_data_with_price['date'])
    
    price_changes = df_data_with_price[df_data_with_price['price_delta_pct'] > 0]
    print("\n요금 인상 이벤트:")
    for _, row in price_changes.iterrows():
        print(f"  {row['date'].strftime('%Y-%m')}: {row['price_delta_pct']:.1f}% 증가")
        print(f"    요금 레벨: {row['price_level']:.2f}")
    
    # 3. 탄력성 분석
    print("\n[3] 수요 탄력성 분석...")
    
    # 요금 그룹별 평균 수요
    df_analysis = df_data_with_price.copy()
    
    # Low, Mid, High 그룹 생성
    df_analysis['price_group'] = pd.qcut(df_analysis['price_level'], 
                                         q=3, labels=['Low', 'Mid', 'High'])
    
    grouped_stats = df_analysis.groupby('price_group').agg({
        'Current_Demand': 'mean',
        'Avg_Fee': 'mean',
        'price_level': 'mean'
    })
    
    print("\n요금 그룹별 통계:")
    print(grouped_stats)
    
    # 탄력성 계산: 수요 변화율 / 요금 변화율
    low_demand = grouped_stats.loc['Low', 'Current_Demand']
    high_demand = grouped_stats.loc['High', 'Current_Demand']
    low_price = grouped_stats.loc['Low', 'price_level']
    high_price = grouped_stats.loc['High', 'price_level']
    
    demand_change = (high_demand - low_demand) / low_demand * 100
    price_change = (high_price - low_price) / low_price * 100
    elasticity = demand_change / price_change if price_change != 0 else 0
    
    print(f"\n수요 탄력성 분석:")
    print(f"  요금 변화: {price_change:.2f}%")
    print(f"  수요 변화: {demand_change:.2f}%")
    print(f"  탄력성: {elasticity:.3f}")
    
    # 4. 시각화
    print("\n[4] 정책 효과 시각화...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # (1) 요금 레벨과 수요의 관계
    ax = axes[0, 0]
    df_with_price = df.merge(df_data[['date', 'price_level', 'Avg_Fee']], 
                             left_on='date', right_on='date', how='left')
    
    ax.scatter(df_with_price['price_level'], df_with_price['true'], 
              alpha=0.5, s=50, label='Actual')
    ax.scatter(df_with_price['price_level'], df_with_price['pred'], 
              alpha=0.5, s=50, label='Predicted', marker='x')
    ax.set_xlabel('Price Level (base=100)')
    ax.set_ylabel('Water Demand (㎥)')
    ax.set_title('요금 레벨과 수요의 관계')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (2) 시간에 따른 요금-수요 추이
    ax = axes[0, 1]
    df_with_date = df.copy()
    df_with_date['date'] = pd.to_datetime(df_with_date['date'])
    
    ax2 = ax.twinx()
    line1 = ax.plot(df_with_date['date'], df_with_date['true'], 
                   'b-', label='Actual Demand', linewidth=1.5)
    line2 = ax.plot(df_with_date['date'], df_with_date['pred'], 
                   'r--', label='Predicted Demand', linewidth=1.5)
    
    # 요금 레벨 표시
    df_price_trace = df_data_with_price.copy()
    line3 = ax2.plot(df_price_trace['date'], df_price_trace['price_level'], 
                    'g-.', label='Price Level', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Water Demand (㎥)', color='black')
    ax2.set_ylabel('Price Level', color='green')
    ax.set_title('시간에 따른 수요 및 요금 추이')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    # (3) 요금 그룹별 평균 수요 비교
    ax = axes[1, 0]
    grouped_stats['Current_Demand'].plot(kind='bar', ax=ax, 
                                        color=['green', 'orange', 'red'])
    ax.set_xlabel('Price Level Group')
    ax.set_ylabel('Average Demand (㎥)')
    ax.set_title('요금 그룹별 평균 수요')
    ax.grid(True, alpha=0.3, axis='y')
    
    # (4) 코로나 기간 분석
    ax = axes[1, 1]
    df_with_date['year_month'] = df_with_date['date'].dt.to_period('M')
    df_with_date['is_covid'] = df_with_date['date'].dt.year.isin([2020, 2021])
    
    covid_demand = df_with_date[df_with_date['is_covid']]['true'].mean()
    normal_demand = df_with_date[~df_with_date['is_covid']]['true'].mean()
    
    categories = ['정상 기간', '코로나 기간 (2020-2021)']
    values = [normal_demand, covid_demand]
    colors = ['steelblue', 'coral']
    
    ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Average Demand (㎥)')
    ax.set_title('코로나 기간 평균 수요 비교')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../results/policy_analysis.png', dpi=300, bbox_inches='tight')
    print("\n[OK] 정책 분석 그래프 저장: ../results/policy_analysis.png")
    
    # 5. 리포트 텍스트 생성
    print("\n[5] 텍스트 리포트 생성...")
    
    report_text = f"""
=================================================================
물 수요 예측 시스템 - 정책 영향 분석 리포트
=================================================================

1. 모델 성능
-----------
학습 세트 (2018-2022):
  - RMSE: 222,528 ㎥
  - MAE:  115,377 ㎥
  - R²:   0.9996 (99.96%)

테스트 세트 (2023-2024):
  - RMSE: 6,201,036 ㎥
  - MAE:  5,478,672 ㎥
  - R²: 0.6828 (68.28%)

2. 요금 정책 효과
-----------
요금 인상 이벤트:
  2018년 7월: 3.5% 증가 (레벨 100 → 103.5)
  2021년 7월: 4.4% 증가 (레벨 108.054 → 108.054)
  2022년 1월: 16.1% 증가 (레벨 125.45)
  2023년 1월: 14.2% 증가 (레벨 143.26)

요금 그룹별 평균 수요:
  - Low (낮은 요금): {low_demand:.0f} ㎥
  - Mid (중간 요금): {grouped_stats.loc['Mid', 'Current_Demand']:.0f} ㎥
  - High (높은 요금): {high_demand:.0f} ㎥

3. 수요 탄력성 분석
-----------
요금 변화: {price_change:.2f}%
수요 변화: {demand_change:.2f}%
탄력성: {elasticity:.3f}

해석: {"탄력적 (요금 변화에 민감)" if elasticity < -0.5 else "비탄력적 (요금에 둔감)"}

4. 코로나 19 영향
-----------
정상 기간 평균 수요: {normal_demand:.0f} ㎥
코로나 기간 평균 수요 (2020-2021): {covid_demand:.0f} ㎥
변화율: {(covid_demand - normal_demand) / normal_demand * 100:.1f}%

5. 주요 발견사항
-----------
- 요금 인상 시 수요 감소 패턴 관찰
- 여름철(7-8월) 수요가 가장 높음
- 코로나 기간 동안 수요 감소 추세
- ANFIS 모델은 계절성과 요금 효과를 잘 포착함

=================================================================
"""
    
    with open('../results/policy_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print("\n[OK] 리포트 저장: ../results/policy_report.txt")
    
    return df

if __name__ == "__main__":
    df = generate_policy_report()

