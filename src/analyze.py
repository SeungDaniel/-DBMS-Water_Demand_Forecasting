# -*- coding: utf-8 -*-
"""
01_analyze_data.py - 데이터 분석 및 구조 확인
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_and_analyze():
    """메인 데이터 로드 및 분석"""
    print("=" * 60)
    print("물 수요 예측 데이터 분석")
    print("=" * 60)
    
    # 1. 메인 데이터 로드
    df = pd.read_csv('../data/demand_monthly_with_price.csv')
    
    # 첫 번째 컬럼이 빈 컬럼이고 두 번째부터 데이터
    print("\n원본 컬럼:", df.columns.tolist())
    print("\n원본 데이터 샘플:")
    print(df.head())
    
    # 날짜 컬럼 처리 - 첫 번째 컬럼이 날짜
    df.rename(columns={df.columns[0]: 'date_str'}, inplace=True)
    df['date'] = pd.to_datetime(df['date_str'], format='%Y.%m.%d')
    df = df.set_index('date')
    
    # 컬럼명 정리
    if 'date_str' in df.columns:
        df = df.drop(columns=['date_str'])
    
    print("\n컬럼명:", df.columns.tolist())
    
    print("\n[1] 데이터 구조:")
    print(df.head(10))
    
    print("\n[2] 데이터 요약 통계:")
    print(df.describe())
    
    print("\n[3] 결측치:")
    print(df.isnull().sum())
    
    print("\n[4] 데이터 범위:")
    print(f"시작: {df.index.min()}")
    print(f"종료: {df.index.max()}")
    print(f"총 기간: {len(df)}개월")
    
    # 계절성 파악
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    print("\n[5] 월별 평균 수요:")
    monthly_avg = df.groupby('month')['Current_Demand'].mean()
    print(monthly_avg)
    
    print("\n[6] 연도별 요금 변화:")
    yearly_price = df.groupby('year').agg({
        'Avg_Fee': 'mean',
        'price_level': 'last',
        'price_delta_pct': lambda x: x[x > 0].sum() if (x > 0).any() else 0
    })
    print(yearly_price)
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # (1) 시계열 수요
    axes[0, 0].plot(df.index, df['Current_Demand'], linewidth=1)
    axes[0, 0].set_title('월별 물 수요 추이', fontsize=12)
    axes[0, 0].set_xlabel('기간')
    axes[0, 0].set_ylabel('수요 (㎥)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # (2) 월별 패턴
    df.boxplot(column='Current_Demand', by='month', ax=axes[0, 1])
    axes[0, 1].set_title('월별 수요 분포')
    axes[0, 1].set_xlabel('월')
    axes[0, 1].set_ylabel('수요 (㎥)')
    
    # (3) 요금 레벨과 수요 관계
    axes[1, 0].scatter(df['price_level'], df['Current_Demand'], alpha=0.6)
    axes[1, 0].set_title('요금 레벨 vs 수요')
    axes[1, 0].set_xlabel('요금 지수 (100 기준)')
    axes[1, 0].set_ylabel('수요 (㎥)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # (4) 요금 인상 이벤트
    price_changes = df[df['price_delta_pct'] > 0]
    if len(price_changes) > 0:
        axes[1, 1].bar(price_changes.index, price_changes['price_delta_pct'])
        axes[1, 1].set_title('요금 인상률')
        axes[1, 1].set_xlabel('날짜')
        axes[1, 1].set_ylabel('인상률 (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, '요금 인상 이벤트 없음', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig('../results/data_analysis.png', dpi=300, bbox_inches='tight')
    print("\n[OK] 데이터 분석 그래프 저장: ../results/data_analysis.png")
    
    return df

if __name__ == "__main__":
    df = load_and_analyze()

