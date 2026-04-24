"""
Time series block splitting utilities.
"""

from typing import List

import numpy as np


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
