# Fraud Detection Pipeline

Real-time fraud detection model for banking transactions with extreme class imbalance (0.17% fraud rate).

## Results

| Model | ROC AUC | Fraud Precision | Fraud Recall |
|-------|---------|-----------------|--------------|
| Logistic Regression | 0.961 | 77% | 52% |
| Logistic Regression (balanced) | 0.978 | 5% | 94% |
| CatBoost | 0.986 | 95% | 79% |
| Isolation Forest (unsupervised) | — | 25% | 29% |
| **CatBoost Pipeline (Optuna)** | **0.992** | **99%** | **80%** |

## Key Findings

- Extreme class imbalance (0.17% fraud) — accuracy of 99.9% is meaningless, baseline model misses 48% of fraud
- CatBoost with Optuna-tuned hyperparameters achieved 99% precision with 80% recall
- Isolation Forest as unsupervised anomaly detector captures 29% of fraud without labels — useful as a first-layer filter
- StandardScaler on Amount/Time features resolved convergence issues and improved model stability
- Balanced class weights boost recall to 94% but collapse precision to 5% — threshold tuning needed for production

## Stack

Python, Pandas, Scikit-learn, CatBoost, Optuna, Isolation Forest

## Data

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284K transactions, 31 features (PCA-anonymized).

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb
```