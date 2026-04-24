"""
Walsh-Hadamard matrix construction.

Implements the recursive construction:
    H_1 = [1]
    H_{2n} = [ H_n   H_n ]
             [ H_n  -H_n ]

Normalized matrix: A = H / sqrt(n) with A @ A.T = I.
"""

import numpy as np


def build_hadamard_matrix(n: int) -> np.ndarray:
    """
    Build Walsh-Hadamard matrix of order n using recursive construction.

    Parameters
    ----------
    n : int
        Order of the matrix (must be a power of 2).

    Returns
    -------
    np.ndarray
        Walsh-Hadamard matrix of shape (n, n).

    Raises
    ------
    ValueError
        If n is not a power of 2.

    Notes
    -----
    Recursive formula:
        H_1 = [1]
        H_{2k} = [[H_k, H_k],
                  [H_k, -H_k]]
    """
    if n == 1:
        return np.array([[1.0]])

    if (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got n={n}")

    H: np.ndarray = np.array([[1.0]])
    size = 1

    while size < n:
        H = np.block([[H, H], [H, -H]])
        size *= 2

    return H


def build_normalized_hadamard(n: int) -> np.ndarray:
    """
    Build normalized Walsh-Hadamard matrix.

    A = (1 / sqrt(n)) * H_n

    The normalized matrix is orthogonal: A @ A.T = I.
    For real data, the inverse equals the transpose: A^{-1} = A.T.

    Parameters
    ----------
    n : int
        Order of the matrix (must be a power of 2).

    Returns
    -------
    np.ndarray
        Normalized matrix A of shape (n, n).
    """
    H = build_hadamard_matrix(n)
    return H / np.sqrt(n)
