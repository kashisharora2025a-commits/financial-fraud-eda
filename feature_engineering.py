"""Feature helpers for fields that exist in the ULB dataset.

The ULB benchmark has no cardholder/account ID or geographic location. Do not
invent a single dummy user to calculate transaction velocity features.
"""
import numpy as np
import pandas as pd


def compute_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "Amount" in result.columns:
        result["Log_Amount"] = np.log1p(result["Amount"].clip(lower=0))
    return result


def compute_velocity_features(df: pd.DataFrame, user_col: str = "User_ID") -> pd.DataFrame:
    if user_col not in df.columns:
        raise ValueError(
            f"Cannot calculate customer velocity: {user_col!r} is absent. "
            "The ULB dataset has no account identifier; do not create a dummy shared user."
        )
    raise NotImplementedError(
        "Customer velocity must be engineered with a causal, per-account rolling window "
        "and computed separately after a time-aware split."
    )


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Only safe basic transforms; model uses original Time, Amount, and V features."""
    return compute_log_amount(df)
