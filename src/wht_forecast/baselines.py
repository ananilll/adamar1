"""
Baseline forecasting methods for comparison.
"""

import numpy as np

from wht_forecast.blocks import split_into_blocks


def naive_forecast(series: np.ndarray, block_size: int) -> np.ndarray:
    """
    Naive forecast: next block equals the last known block.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    block_size : int
        Block length.

    Returns
    -------
    np.ndarray
        Forecast block (copy of last block).
    """
    blocks = split_into_blocks(series, block_size)
    return blocks[-1].copy()


def moving_average_forecast(
    series: np.ndarray, block_size: int, window: int = 3
) -> np.ndarray:
    """
    Moving average forecast: element-wise mean over last window blocks.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    block_size : int
        Block length.
    window : int
        Number of recent blocks to average (default: 3).

    Returns
    -------
    np.ndarray
        Forecast block.
    """
    blocks = split_into_blocks(series, block_size)
    recent = blocks[-window:] if len(blocks) >= window else blocks
    return np.mean(recent, axis=0)


def linear_extrapolation_forecast(series: np.ndarray, block_size: int) -> np.ndarray:
    """
    Linear extrapolation: extrapolate using delta between last two blocks.

    forecast = blocks[-1] + (blocks[-1] - blocks[-2])

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    block_size : int
        Block length.

    Returns
    -------
    np.ndarray
        Forecast block.
    """
    blocks = split_into_blocks(series, block_size)
    if len(blocks) < 2:
        return blocks[-1].copy()

    delta = blocks[-1] - blocks[-2]
    return blocks[-1] + delta
