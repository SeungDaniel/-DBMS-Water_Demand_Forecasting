# 프로젝트 구조 (Project Structure)

이 문서는 서울시 물 수요 예측 프로젝트의 폴더 구조와 주요 파일을 설명합니다.

## 폴더 개요 (Directory Overview)

- **`src/`**: 데이터 처리, 모델 학습, 대시보드 실행을 위한 소스 코드
- **`data/`**: 원본 데이터 및 전처리된 데이터셋
- **`results/`**: 생성된 그래프, 리포트, 예측 결과 파일
- **`docs/`**: 프로젝트 문서 및 결과 보고서

## 주요 파일 (Key Files)

### `src/` (소스 코드)
- **`app.py`**: 메인 대시보드 애플리케이션. 실행: `streamlit run src/app.py`
- **`data_prep.py`**: 데이터 로드, 정제, 특성 공학 (코로나 감면 로직 포함)
- **`train.py`**: ANFIS 모델 구현 및 학습
- **`train_xgboost.py`**: 비교용 XGBoost 모델 구현
- **`train_hybrid.py`**: 하이브리드 모델 (XGBoost + OLS 탄력성) 구현
- **`scenario_analysis.py`**: 요금 변동 시나리오 분석 스크립트

### `data/` (데이터)
- **`anfis_dataset_with_covid.csv`**: 학습 및 테스트에 사용되는 최종 통합 데이터셋
- *기타 원본 파일*: `demand_monthly.csv`, `ComTempPrec.xlsx` 등

### `results/` (결과물)
- **`xgboost_results.png`**: XGBoost 모델 성능 시각화
- **`scenario_analysis_2025.png`**: 요금 시나리오별 수요 예측 그래프
- **`anfis_results_with_covid.png`**: 튜닝된 ANFIS 모델 성능 시각화

### `docs/` (문서)
- **`walkthrough.md`**: 프로젝트 전체 흐름, 모델 튜닝, 하이브리드 솔루션을 설명하는 영문 리포트
- **`walkthrough_KR.md`**: 위 리포트의 한글 버전
- **`PROJECT_STRUCTURE.md`**: 영문 프로젝트 구조 설명서
- **`PROJECT_STRUCTURE_KR.md`**: 한글 프로젝트 구조 설명서 (본 문서)

## 실행 방법 (How to Run)
1. **패키지 설치**: `pip install -r requirements.txt`
2. **대시보드 실행**: `streamlit run src/app.py`
