"""
Synthetic time series generation for testing and experiments.
"""

import numpy as np


def generate_synthetic_series(
    n: int = 512,
    seed: int = 42,
    noise_level: float = 0.2,
) -> np.ndarray:
    """
    Generate synthetic time series for testing.

    Components:
    - Slow trend (low-frequency sinusoid)
    - Seasonal oscillations (medium frequency)
    - High-frequency component
    - Gaussian noise

    Parameters
    ----------
    n : int
        Series length (default: 512).
    seed : int
        Random seed for reproducibility (default: 42).
    noise_level : float
        Standard deviation of Gaussian noise (default: 0.2).

    Returns
    -------
    np.ndarray
        Synthetic time series of length n.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)

    trend = 2.0 * np.sin(0.3 * t)
    seasonal = 1.5 * np.sin(2.0 * t) + 0.8 * np.cos(3.5 * t)
    high_freq = 0.5 * np.sin(8.0 * t)
    noise = noise_level * rng.standard_normal(n)

    return trend + seasonal + high_freq + noise
