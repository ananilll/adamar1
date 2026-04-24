"""
Forward and inverse Walsh-Hadamard transform.

Forward:  C = A @ X
Inverse:  X = A.T @ C
"""

import numpy as np


def wht_forward(block: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Apply forward Walsh-Hadamard transform to a data block.

    C = A @ X

    Parameters
    ----------
    block : np.ndarray
        Input block of length n (1D array).
    A : np.ndarray
        Normalized Walsh-Hadamard matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        Vector of spectral coefficients C.
    """
    return A @ block


def wht_inverse(coeffs: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Apply inverse Walsh-Hadamard transform.

    Since A is orthogonal: A^{-1} = A.T
    X_hat = A.T @ C_hat

    Parameters
    ----------
    coeffs : np.ndarray
        Vector of spectral coefficients.
    A : np.ndarray
        Normalized Walsh-Hadamard matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        Reconstructed block in the original domain.
    """
    return A.T @ coeffs
