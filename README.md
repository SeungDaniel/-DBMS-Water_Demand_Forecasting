# 물 수요 예측 시스템 - Water Demand Forecasting System

## 프로젝트 개요

서울시 월별 물 사용량을 예측하는 ANFIS(Adaptive Neuro-Fuzzy Inference System) 기반 모델입니다.

### 주요 특징
- **모델**: ANFIS (가우시안 멤버십 함수 + Sugeno consequents)
- **입력 특성**: 온도, 강수량, 인구, 이전 수요, 요금 레벨, 계절성
- **예측 대상**: 월별 물 사용량 (㎥)
- **데이터 기간**: 2018-2024 (84개월)

## 성능

### 학습 세트 (2018-2022)
- **RMSE**: 222,528 ㎥
- **MAE**: 115,377 ㎥
- **R²**: 0.9996 (99.96%)
- **MAPE**: 0.13%

### 테스트 세트 (2023-2024)
- **RMSE**: 6,201,036 ㎥
- **MAE**: 5,478,672 ㎥
- **R²**: 0.6828 (68.28%)
- **MAPE**: 6.13%

## 실행 방법

### 1. 데이터 준비
```bash
python scripts/01_analyze_data.py
```

### 2. 데이터 통합
```bash
python scripts/02_integrate_data.py
```

### 3. 모델 학습
```bash
python scripts/03_train_anfis.py
```

### 4. 리포트 생성
```bash
python scripts/04_generate_report.py
```

## 파일 구조

```
anfis_water_forecasting/
├── README.md
├── scripts/
│   ├── 01_analyze_data.py      # 데이터 분석
│   ├── 02_integrate_data.py     # 데이터 통합
│   ├── 03_train_anfis.py        # 모델 학습
│   └── 04_generate_report.py    # 리포트 생성
├── data/
│   ├── anfis_dataset.csv        # 통합 데이터셋
│   └── anfis_predictions.csv    # 예측 결과
└── results/
    ├── data_analysis.png        # 데이터 분석 그래프
    ├── anfis_results.png        # 모델 결과 그래프
    ├── policy_analysis.png      # 정책 분석 그래프
    └── policy_report.txt        # 분석 리포트
```

## 주요 발견사항

### 요금 정책 효과
- **요금 인상**: 2018.7(+3.5%), 2021.7(+4.4%), 2022.1(+16.1%), 2023.1(+14.2%)
- **요금 탄력성**: -0.064 (비탄력적)
- **수요 변화**: 요금 인상 시 -2.51% 감소

### 계절성 패턴
- 여름철(7-8월) 수요 최대: 108,749,600 ㎥
- 겨울철(1-2월) 수요 최소: 76,182,650 ㎥

### 코로나 영향
- 코로나 기간 수요: -0.2% 감소
- 효과 미미

## 환경 요구사항

```
numpy>=1.20.0
pandas>=1.3.0
torch>=1.9.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## 라이선스

MIT License

## 참고문헌

- Forecasting municipal water demands (EPCOR, 2019)


