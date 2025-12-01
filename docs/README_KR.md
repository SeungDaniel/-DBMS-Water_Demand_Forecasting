# 서울시 물 수요 예측 프로젝트

## 개요 (Overview)
본 프로젝트는 머신러닝 모델을 활용하여 서울시의 월별 물 수요를 예측하는 것을 목표로 합니다. 특히 높은 예측 정확도와 경제학적 설명력(요금 탄력성)을 동시에 달성하기 위해 하이브리드 모델을 구축했습니다.

## 주요 특징 (Key Features)
- **하이브리드 모델 (Hybrid Model)**: **XGBoost**(정확도)와 **회귀분석(OLS)**(탄력성)을 결합하여, 정확한 예측과 논리적인 정책 시뮬레이션을 모두 구현했습니다.
- **시나리오 분석**: 요금 인상/인하(+10%, -10% 등)에 따른 미래 물 수요 변화를 시뮬레이션합니다.
- **인터랙티브 대시보드**: Streamlit을 활용하여 사용자가 직접 변수를 조정하고 결과를 확인할 수 있는 웹 대시보드를 제공합니다.

## 프로젝트 구조 (Project Structure)
- `src/`: 모델 학습 및 대시보드 소스 코드
- `data/`: 데이터셋 (코로나 감면 로직 적용됨)
- `results/`: 결과 그래프 및 리포트
- `docs/`: 상세 프로젝트 문서 및 보고서

## 모델 성능 비교
| 모델 | Test R² (정확도) | Test RMSE (오차) | 정책 민감도 |
| :--- | :--- | :--- | :--- |
| **ANFIS** | 0.68 | 620만 톤 | 불안정 (역방향) |
| **XGBoost** | **0.93** | **300만 톤** | 없음 (0%) |
| **Hybrid** | **0.93** | **300만 톤** | **정상 (-0.062)** |

## 실행 방법 (How to Run)
1. **저장소 복제 (Clone)**:
   ```bash
   git clone https://github.com/SeungDaniel/-DBMS-Water_Demand_Forecasting.git
   cd -DBMS-Water_Demand_Forecasting
   ```
2. **패키지 설치**:
   ```bash
   pip install -r requirements.txt
   ```
3. **대시보드 실행**:
   ```bash
   streamlit run src/app.py
   ```

## 문서 (Documentation)
더 자세한 내용은 `docs/` 폴더를 참고하세요:
- [상세 리포트 (한글)](docs/walkthrough_KR.md)
- [프로젝트 구조 설명 (한글)](docs/PROJECT_STRUCTURE_KR.md)
