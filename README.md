# Customer Churn Prediction with Ensemble Learning

A reproducible machine-learning project that identifies customers at risk of churn and combines churn probability with customer lifetime value (CLTV) to support targeted retention campaigns.

## Problem it solves

Retention teams need to focus limited outreach on customers who are both likely to leave and valuable to retain. This project builds an AdaBoost classification pipeline, evaluates it with ROC-AUC and classification metrics, and creates prioritized retention segments using churn risk and CLTV.

> This is a sanitized portfolio demonstration using synthetic data. It contains no employer code, customer records, or confidential business data.

## Highlights

- Reproducible synthetic telecom-style dataset generation
- Data preprocessing with imputation, scaling, and one-hot encoding
- AdaBoost model training with stratified train/test splitting
- ROC-AUC, accuracy, precision, recall, and F1 evaluation
- CLTV calculation and high-value/high-risk retention prioritization
- Saved model and machine-readable evaluation report
- Unit tests for data generation and CLTV logic

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --rows 2500 --output artifacts
pytest
```

## Outputs

- `artifacts/churn_model.joblib` — trained preprocessing and AdaBoost pipeline
- `artifacts/metrics.json` — evaluation metrics and dataset details
- `artifacts/retention_candidates.csv` — ranked high-risk customer list

## Responsible use

Predictions should support—not replace—human decisions. Before production use, evaluate subgroup performance, monitor drift, establish intervention policies, and verify that retention actions do not create unfair outcomes.

## Author

Nagendra — AI/ML Engineer experienced in machine learning, model evaluation, GenAI, and production-oriented Python development.
