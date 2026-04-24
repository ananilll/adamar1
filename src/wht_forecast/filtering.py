"""
Spectral coefficient filtering and energy computation.
"""

from typing import Tuple

import numpy as np


def compute_energy(coeffs: np.ndarray) -> np.ndarray:
    """
    Compute energy (squared magnitude) of each coefficient.

    E_i = C_i^2

    Parameters
    ----------
    coeffs : np.ndarray
        Vector of spectral coefficients.

    Returns
    -------
    np.ndarray
        Vector of energies.
    """
    return coeffs**2


def select_top_coefficients(
    coeffs: np.ndarray, top_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top-k coefficients by energy; zero out the rest.

    Procedure:
    1. Compute energy E_i = C_i^2 for each coefficient.
    2. Select indices of top_k coefficients with maximum energy.
    3. Zero out all other coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        Vector of spectral coefficients.
    top_k : int
        Number of coefficients to retain.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - filtered: Vector with only top-k coefficients retained.
        - indices: Indices of selected coefficients.
    """
    energy = coeffs**2
    top_indices = np.argsort(energy)[::-1][:top_k]
    filtered = np.zeros_like(coeffs)
    filtered[top_indices] = coeffs[top_indices]
    return filtered, top_indices
