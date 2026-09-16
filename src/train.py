"""Train and evaluate an AdaBoost churn model."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import add_cltv, generate_customers


def build_pipeline() -> Pipeline:
    numeric = ["tenure_months", "monthly_charges", "support_calls"]
    categorical = ["contract_type", "internet_service", "paperless_billing"]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", AdaBoostClassifier(n_estimators=120, learning_rate=0.6, random_state=42)),
    ])


def train(rows: int, output: Path) -> dict[str, float]:
    frame = add_cltv(generate_customers(rows=rows))
    features = frame.drop(columns=["customer_id", "churn", "cltv"])
    target = frame["churn"]
    train_x, test_x, train_y, test_y, train_idx, test_idx = train_test_split(
        features, target, frame.index, test_size=0.25, random_state=42, stratify=target
    )

    pipeline = build_pipeline()
    pipeline.fit(train_x, train_y)
    probability = pipeline.predict_proba(test_x)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(test_y, probability), 4),
        "accuracy": round(accuracy_score(test_y, prediction), 4),
        "precision": round(precision_score(test_y, prediction, zero_division=0), 4),
        "recall": round(recall_score(test_y, prediction, zero_division=0), 4),
        "f1": round(f1_score(test_y, prediction, zero_division=0), 4),
        "training_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
    }

    candidates = frame.loc[test_idx, ["customer_id", "cltv"]].copy()
    candidates["churn_probability"] = probability
    candidates["priority_score"] = candidates["churn_probability"] * candidates["cltv"]
    candidates = candidates.sort_values("priority_score", ascending=False)

    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "churn_model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    candidates.head(100).to_csv(output / "retention_candidates.csv", index=False)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    arguments = parser.parse_args()
    print(json.dumps(train(arguments.rows, arguments.output), indent=2))
