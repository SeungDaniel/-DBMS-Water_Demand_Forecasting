# ANFIS Model Tuning & Scenario Analysis Walkthrough

## 1. Model Tuning Results
We successfully tuned the ANFIS model by:
- **Reducing Rules**: 10 → 6 (Reduced complexity)
- **Increasing Regularization**: Weight Decay 1e-5 → 1e-3
- **Realistic Data**: Applied weighted Covid discount (23% for commercial sector)

### Performance Comparison
| Metric | Previous (Overfitted) | **Tuned (Final)** |
| :--- | :--- | :--- |
| **Test R²** | 0.57 | **0.68** (+11%p) |
| **Test RMSE** | 7.2M ㎥ | **6.2M ㎥** (-1.0M) |

> [!NOTE]
> The tuned model shows significantly better generalization on unseen data (2023-2024).

## 2. Scenario Analysis (The "Stress Test")
We simulated 2025 water demand under different price scenarios to test the model's policy logic.

### Results (Counter-Intuitive)
| Scenario | Price Change | Demand Change | Elasticity |
| :--- | :--- | :--- | :--- |
| **Status Quo** | 0% | - | - |
| **Price Up** | +10% | **+7.8%** 🔺 | Positive (Wrong) |
| **Price Down** | -10% | **+1.4%** 🔺 | Negative (Wrong) |

### Why did this happen?
The ANFIS model failed this "Stress Test" because of **Extrapolation Failure**:
1.  **Local Validity**: Fuzzy rules are valid only within the range of data seen during training.
2.  **Out-of-Distribution**: When we artificially increase price by 10%, it pushes the input into a "grey zone" where no rule fires strongly, leading to unpredictable behavior (defaulting to a higher baseline).

## 3. Conclusion & Next Steps
- **ANFIS is good for Interpolation**: It predicts well when conditions are similar to the past (R² 0.68).
- **ANFIS is bad for Extrapolation**: It fails to capture consistent economic laws (Law of Demand) when inputs go beyond historical ranges.

> [!IMPORTANT]
> **Recommendation**: To build a robust policy simulator, we must introduce a model that handles extrapolation better, such as **XGBoost** (Tree-based) or **Linear Regression** (for elasticity baseline). This justifies the "Model Enhancement" phase of the final project.

## 4. XGBoost Implementation Results
We implemented XGBoost to overcome ANFIS's limitations.

### Performance Comparison
| Metric | ANFIS (Tuned) | **XGBoost** | Improvement |
| :--- | :--- | :--- | :--- |
| **Test R²** | 0.68 | **0.93** | **+25%p** (Huge!) |
| **Test RMSE** | 6.2M ㎥ | **3.0M ㎥** | **-50%** (Halved) |

### Scenario Analysis (Price Elasticity)
| Scenario | Price Change | ANFIS Prediction | XGBoost Prediction |
| :--- | :--- | :--- | :--- |
| **Price Up** | +10% | +7.8% (Wrong direction) | **0.0%** (Insensitive) |
| **Price Down** | -10% | +1.4% (Wrong direction) | **0.0%** (Insensitive) |

### Critical Insight for Final Report
- **ANFIS**: Sensitive to price but gets the direction wrong (unstable extrapolation).
- **XGBoost**: Extremely accurate at predicting demand based on **Seasonality** (Month, Temp), but ignores **Price** because historical price changes had little impact compared to weather.
- **Conclusion**: For a "Policy Simulator", we might need a **Hybrid Approach** or a simpler **Econometric Model (Elasticity-based)** to enforce the law of demand, while using XGBoost for the baseline demand prediction. This is a perfect "Future Work" or "Discussion" point for the final project.

## 5. Hybrid Model (The Final Solution)
We implemented a **Hybrid Model** that combines:
1.  **Regression Analysis (OLS)**: To calculate "Price Elasticity" (-0.0621, P-value 0.028).
2.  **XGBoost**: To predict "Base Demand" with high accuracy (R² 0.93).

### Final Scenario Analysis
| Scenario | Price Change | ANFIS | XGBoost | **Hybrid (Final)** |
| :--- | :--- | :--- | :--- | :--- |
| **Price Up** | +10% | +7.8% (Wrong) | 0.0% (No Change) | **-0.62%** (Correct!) |
| **Price Down** | -10% | +1.4% (Wrong) | 0.0% (No Change) | **+0.62%** (Correct!) |

### Conclusion
- **Accuracy**: Maintained XGBoost's high performance (R² 0.93).
- **Logic**: Enforced economic laws (Law of Demand) via the hybrid structure.
- **Value**: Satisfies both the **Operations Manager** (Accuracy) and the **Policy Maker** (Causality).
