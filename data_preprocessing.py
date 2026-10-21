"""Data-quality inspection helpers. No automatic imputation or global preprocessing."""
import pandas as pd


def inspect_data(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_by_column": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "class_counts": df["Class"].value_counts().sort_index().to_dict() if "Class" in df else {},
        "column_types": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exact duplicates removed; fail rather than hide missingness."""
    if df.isna().any().any():
        raise ValueError("Missing values found. Review them in the EDA notebook; no silent imputation is applied.")
    return df.drop_duplicates().copy()
