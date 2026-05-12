"""
Minimal artifact I/O helpers for the Polymodel pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Union

import numpy as np
import pandas as pd


def save_parquet(df: Union[pd.DataFrame, pd.Series], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.to_parquet(path, index=True)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(array, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_npy(path: str):
    return np.load(path, allow_pickle=False)
