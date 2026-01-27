from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple
import pandas as pd


TickerType = Literal["price", "rate"]


@dataclass(frozen=True)
class DictionaryColumns:
    """
    Column mapping for the ticker dictionary file.
    Adjust these if your Excel differs.
    """
    symbol: str = "SYMBOL"
    nature: str = "Nature"  # expected values include "Price", "Rate", etc.


def load_ticker_dictionary(path: str, cols: DictionaryColumns = DictionaryColumns()) -> pd.DataFrame:
    """
    Load the ticker dictionary from Excel or CSV.
    Keeps all columns, but validates presence of required ones.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported dictionary format: {path}")

    missing = [c for c in (cols.symbol, cols.nature) if c not in df.columns]
    if missing:
        raise KeyError(f"Dictionary is missing required columns: {missing}. Found: {list(df.columns)}")

    # Clean
    df = df.copy()
    df[cols.symbol] = df[cols.symbol].astype(str).str.strip()
    df[cols.nature] = df[cols.nature].astype(str).str.strip()
    df = df[df[cols.symbol] != ""]
    return df


def build_ticker_type_map(
    dictionary_df: pd.DataFrame,
    cols: DictionaryColumns = DictionaryColumns(),
    rate_regex: str = r"(?i)\b(rate|yield)\b",
) -> Dict[str, TickerType]:
    """
    Build a mapping: ticker -> "price" or "rate".

    By default, any row whose 'Nature' contains "Rate" or "Yield" (case-insensitive)
    is treated as a rate series. Everything else defaults to "price".

    You can change the rule later if your dictionary has a cleaner categorical column.
    """
    nature = dictionary_df[cols.nature].astype(str)
    is_rate = nature.str.contains(rate_regex, regex=True, na=False)
    symbols = dictionary_df[cols.symbol].astype(str)

    mapping: Dict[str, TickerType] = {}
    for sym, flag in zip(symbols, is_rate):
        if sym and sym not in mapping:
            mapping[sym] = "rate" if bool(flag) else "price"
    return mapping


def validate_ticker_map_against_returns(ticker_map: Dict[str, TickerType], returns_df: pd.DataFrame) -> Tuple[Sequence[str], Sequence[str]]:
    """
    Returns (missing_in_map, missing_in_returns).
    Missing tickers are not fatal by themselves: you may have extra tickers in either side.
    """
    returns_cols = set(map(str, returns_df.columns))
    map_cols = set(map(str, ticker_map.keys()))
    missing_in_map = sorted(list(returns_cols - map_cols))
    missing_in_returns = sorted(list(map_cols - returns_cols))
    return missing_in_map, missing_in_returns


def subset_map(ticker_map: Dict[str, TickerType], tickers: Iterable[str]) -> Dict[str, TickerType]:
    tickers = list(tickers)
    return {t: ticker_map[t] for t in tickers if t in ticker_map}