from __future__ import annotations

from typing import Optional
import pandas as pd


def load_returns(path: str) -> pd.DataFrame:
    """
    Load daily returns (prices already transformed).
    Supported formats: .csv, .parquet
    Index must be parseable as dates.
    """
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unsupported returns format: {path}")

    # Robust datetime parsing (handles empty rows or weird strings gracefully)
    idx = pd.to_datetime(df.index, errors="coerce", dayfirst=True)
    df.index = idx

    # Drop rows with unparsed dates
    if df.index.isna().any():
        df = df.loc[~df.index.isna()].copy()

    # If duplicate dates exist, keep the last occurrence
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].copy()
    
    df = df.sort_index()
    return df


def validate_returns(df: pd.DataFrame) -> None:
    """
    Basic validation: monotonic dates, numeric columns.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("Returns index must be sorted ascending by date.")
    if df.shape[0] < 10:
        raise ValueError("Returns dataframe seems too short.")
    # Ensure numeric
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric return columns: {non_numeric[:10]}")
