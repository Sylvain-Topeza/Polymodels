from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Optional, Union

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_id(config_hash: str, label: Optional[str] = None) -> str:
    """
    Run identifier used in artifact paths.
    """
    if label:
        safe = "".join(c for c in label if c.isalnum() or c in ("-", "_"))
        return f"{safe}_{config_hash}"
    return config_hash


def artifact_dir(artifacts_root: str, run_id: str) -> str:
    return os.path.join(artifacts_root, run_id)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parquet(df: Union[pd.DataFrame, pd.Series], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.to_parquet(path, index=True)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def exists(path: str) -> bool:
    return os.path.exists(path)


def save_pickle(obj: Any, path: str) -> None:
    import pickle
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def save_npy(array, path: str) -> None:
    import numpy as np
    ensure_dir(os.path.dirname(path))
    np.save(path, array)


def load_npy(path: str):
    import numpy as np
    return np.load(path, allow_pickle=False)