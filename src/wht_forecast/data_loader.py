"""
CSV data loading and preprocessing for time series.

Supports Yahoo Finance, MetaTrader, TradingView, and generic CSV exports.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Minimum values required for forecasting (2 blocks minimum)
MIN_SERIES_LENGTH = 64

# Price column priority for financial datasets
PRICE_COLUMN_PRIORITY = ["Close", "close", "Adj Close", "Adj Close*", "price", "value"]

# Date column names for sorting
DATE_COLUMN_NAMES = ["Date", "date", "datetime", "DateTime", "time", "Timestamp"]


def _clean_numeric_string(s: str) -> str:
    """
    Clean string for European number format conversion.
    - Remove quotes
    - Remove spaces (thousands separator)
    - Convert comma to decimal point
    """
    if not isinstance(s, str):
        return str(s)
    s = s.strip().strip('"\'')
    s = s.replace(" ", "").replace("\u00a0", "")  # Remove spaces and nbsp
    s = s.replace(",", ".")  # European decimal
    return s


def _to_numeric_robust(col: pd.Series) -> pd.Series:
    """
    Convert column to numeric, handling:
    - European format: "19403,9" -> 19403.9
    - With spaces: "19 403,9" -> 19403.9
    - Quotes and extra spaces
    """
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(np.float64)
    cleaned = col.astype(str).apply(_clean_numeric_string)
    return pd.to_numeric(cleaned, errors="coerce")


def _find_date_column(df: pd.DataFrame) -> str | None:
    """Find date column for sorting. Case-insensitive match."""
    cols_lower = {c.lower(): c for c in df.columns}
    for name in ["date", "datetime", "time", "timestamp"]:
        if name in cols_lower:
            return cols_lower[name]
    return None


def _parse_and_sort_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse date column and sort dataframe chronologically."""
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        return df
    except Exception:
        return df


def _select_price_column(df: pd.DataFrame, value_column: str | None) -> tuple[str, pd.Series]:
    """
    Select price column by priority.
    Priority: Close, close, Adj Close, price, value. Else last numeric column.
    """
    # Get all columns that convert to numeric
    convertible: dict[str, pd.Series] = {}
    for c in df.columns:
        converted = _to_numeric_robust(df[c])
        valid = converted.dropna()
        if len(valid) > 0:
            convertible[c] = converted

    if not convertible:
        raise ValueError(
            "No numeric columns found. Check decimal format (comma vs period) "
            "and ensure at least one column contains numeric values."
        )

    if value_column is not None:
        if value_column not in df.columns:
            raise ValueError(
                f"Column '{value_column}' not found. Available: {list(df.columns)}"
            )
        return value_column, _to_numeric_robust(df[value_column])

    # Priority order
    for preferred in PRICE_COLUMN_PRIORITY:
        if preferred in convertible:
            return preferred, convertible[preferred]

    # Fallback: last numeric column (often Close in OHLCV)
    last_col = list(convertible.keys())[-1]
    return last_col, convertible[last_col]


def load_time_series_from_csv(
    path: str,
    value_column: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load and clean time series from CSV for forecasting.

    Performs:
    - Auto-detect price column (Close, close, Adj Close, price, value)
    - European number format ("19403,9", "19 403,9")
    - Remove quotes and spaces
    - Parse Date column and sort chronologically
    - Drop NaN rows
    - Validate minimum 64 values

    Parameters
    ----------
    path : str
        Path to CSV file.
    value_column : str | None
        Override column selection. If None, auto-detect by priority.
    verbose : bool
        Print diagnostics (default: True).

    Returns
    -------
    np.ndarray
        Clean 1D float array, sorted by date if Date column exists.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If no numeric data, or fewer than 64 values.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Try reading with different decimal separators
    df: pd.DataFrame
    for decimal in [".", ","]:
        try:
            df = pd.read_csv(path, decimal=decimal)
            break
        except Exception:
            continue
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError("CSV file is empty")

    # Sort by date if Date column exists
    date_col = _find_date_column(df)
    if date_col is not None:
        df = _parse_and_sort_by_date(df, date_col)

    # Select price column
    col_name, values = _select_price_column(df, value_column)

    # Drop NaN
    values = values.dropna()
    if len(values) == 0:
        raise ValueError(
            f"Column '{col_name}' has no valid numeric values after cleaning. "
            "Check for non-numeric or corrupted data."
        )

    # Convert to numpy
    series = values.astype(np.float64).values

    # Validation: minimum 64 values
    if len(series) < MIN_SERIES_LENGTH:
        raise ValueError(
            f"Series has {len(series)} values. Need at least {MIN_SERIES_LENGTH} for forecasting. "
            f"Add more data or use a smaller block_size."
        )

    if verbose:
        first_vals = series[:3].tolist()
        print("Loaded CSV successfully")
        print(f"Rows: {len(series)}")
        print(f"Column used for forecasting: {col_name}")
        print(f"First values: {first_vals}")

    return series


def load_csv_series(
    path: str,
    value_column: str | None = None,
    decimal: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Load time series from CSV file (legacy API).

    Delegates to load_time_series_from_csv for robust cleaning.
    Use load_time_series_from_csv directly for diagnostics.

    Parameters
    ----------
    path : str
        Path to CSV file.
    value_column : str | None
        Column name for values. If None, auto-detect.
    decimal : str | None
        Ignored; European format is auto-detected.
    verbose : bool
        Print diagnostics (default: False).

    Returns
    -------
    np.ndarray
        1D float array of time series values.
    """
    return load_time_series_from_csv(
        path, value_column=value_column, verbose=verbose
    )


def normalize_series(
    series: np.ndarray,
    method: Literal["zscore", "minmax", "none"] = "none",
) -> np.ndarray:
    """
    Normalize time series for forecasting.

    Parameters
    ----------
    series : np.ndarray
        Input 1D time series.
    method : Literal["zscore", "minmax", "none"]
        Normalization method:
        - zscore: (x - mean) / std
        - minmax: (x - min) / (max - min)
        - none: return copy unchanged

    Returns
    -------
    np.ndarray
        Normalized series (copy).
    """
    out = series.copy()
    if method == "none":
        return out

    if method == "zscore":
        mean = np.mean(out)
        std = np.std(out)
        if std == 0:
            return out - mean
        return (out - mean) / std

    if method == "minmax":
        lo, hi = np.min(out), np.max(out)
        if hi == lo:
            return np.zeros_like(out)
        return (out - lo) / (hi - lo)

    raise ValueError(f"Unknown normalization method: {method}")


def validate_series_for_forecasting(
    series: np.ndarray,
    block_size: int,
) -> None:
    """
    Validate that series has enough data for forecasting.

    Requires at least 3 full blocks (2 for training/deltas, 1 for holdout).

    Parameters
    ----------
    series : np.ndarray
        Time series to validate.
    block_size : int
        Block size used in forecasting.

    Raises
    ------
    ValueError
        If series is too short.
    """
    min_length = block_size * 3
    if len(series) < min_length:
        raise ValueError(
            f"Series length ({len(series)}) is too short for forecasting. "
            f"Need at least {min_length} values (3 blocks of size {block_size})."
        )
