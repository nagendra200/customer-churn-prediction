"""Synthetic data generation and customer value helpers."""

import numpy as np
import pandas as pd


def generate_customers(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("rows must be at least 100")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    monthly = rng.uniform(20, 130, rows).round(2)
    support_calls = rng.poisson(2.2, rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["fiber", "dsl", "none"], rows, p=[0.52, 0.38, 0.10])
    paperless = rng.choice(["yes", "no"], rows, p=[0.7, 0.3])

    logits = (
        -1.5
        + 1.0 * (contract == "month-to-month")
        + 0.35 * (internet == "fiber")
        + 0.18 * support_calls
        + 0.008 * monthly
        - 0.025 * tenure
    )
    probability = 1 / (1 + np.exp(-logits))
    churn = rng.binomial(1, probability)

    return pd.DataFrame({
        "customer_id": [f"CUST-{value:06d}" for value in range(1, rows + 1)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "support_calls": support_calls,
        "contract_type": contract,
        "internet_service": internet,
        "paperless_billing": paperless,
        "churn": churn,
    })


def add_cltv(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    expected_remaining_months = np.maximum(6, 48 - result["tenure_months"] * 0.35)
    result["cltv"] = (result["monthly_charges"] * expected_remaining_months).round(2)
    return result
