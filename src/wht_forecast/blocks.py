"""
Time series block splitting utilities.
"""

from typing import List, Tuple

import numpy as np


def pad_series_to_blocks(
    series: np.ndarray,
    block_size: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Append samples so the series length is an exact multiple of ``block_size``.

    Does not modify the input array.

    Parameters
    ----------
    series : np.ndarray
        Input time series (1D).
    block_size : int
        Target block length.
    pad_value : float
        Value used for appended samples (default: 0.0).

    Returns
    -------
    padded : np.ndarray
        Copy of the series with trailing padding, ``dtype`` float64.
    n_pad : int
        Number of values appended (0 if already aligned).
    """
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    r = int(x.size % block_size)
    if r == 0:
        return x.copy(), 0
    n_pad = block_size - r
    tail = np.full(n_pad, pad_value, dtype=np.float64)
    return np.concatenate([x, tail]), n_pad


def split_into_blocks(series: np.ndarray, block_size: int) -> List[np.ndarray]:
    """
    Split time series into non-overlapping blocks of fixed size.

    The last incomplete block is discarded.

    Parameters
    ----------
    series : np.ndarray
        Input time series (1D array).
    block_size : int
        Length of each block (recommended: power of 2, e.g. 32).

    Returns
    -------
    List[np.ndarray]
        List of blocks, each of length block_size.
    """
    n_blocks = len(series) // block_size
    blocks: List[np.ndarray] = []

    for k in range(n_blocks):
        start = k * block_size
        end = (k + 1) * block_size
        block = series[start:end].astype(np.float64)
        blocks.append(block)

    return blocks
