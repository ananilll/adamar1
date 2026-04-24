"""
WHT-based forecasting algorithm.

Pipeline:
1. Split series into blocks
2. Apply WHT to each block
3. Select top-k coefficients
4. Compute deltas between consecutive blocks
5. Smooth deltas
6. Forecast next block coefficients
7. Apply inverse WHT
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from wht_forecast.blocks import split_into_blocks
from wht_forecast.filtering import select_top_coefficients
from wht_forecast.hadamard import build_normalized_hadamard
from wht_forecast.trace_log import log_trace
from wht_forecast.transform import wht_forward, wht_inverse


def _array_summary(name: str, a: np.ndarray) -> str:
    """Compact numeric summary for console trace."""
    a = np.asarray(a)
    return (
        f"{name}: shape={a.shape}, dtype={a.dtype}, "
        f"min={float(np.min(a)):.6g}, max={float(np.max(a)):.6g}, "
        f"mean={float(np.mean(a)):.6g}, L2={float(np.linalg.norm(a)):.6g}"
    )


def compute_deltas(coeffs_history: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute coefficient deltas between consecutive blocks.

    delta_C^{(k)} = C_filtered^{(k)} - C_filtered^{(k-1)}

    Parameters
    ----------
    coeffs_history : List[np.ndarray]
        History of filtered coefficients per block.

    Returns
    -------
    List[np.ndarray]
        List of delta vectors.
    """
    deltas: List[np.ndarray] = []
    for i in range(1, len(coeffs_history)):
        deltas.append(coeffs_history[i] - coeffs_history[i - 1])
    return deltas


def smooth_deltas(deltas: List[np.ndarray], m: int) -> np.ndarray:
    """
    Smooth deltas over the last m blocks.

    delta_avg = mean(delta^{(k)}, delta^{(k-1)}, ..., delta^{(k-m+1)})

    Parameters
    ----------
    deltas : List[np.ndarray]
        List of coefficient deltas.
    m : int
        Smoothing window size.

    Returns
    -------
    np.ndarray
        Averaged delta vector.

    Raises
    ------
    ValueError
        If deltas is empty (requires at least 2 blocks).
    """
    if len(deltas) == 0:
        raise ValueError("Need at least 2 blocks to compute deltas")

    window = deltas[-m:] if len(deltas) >= m else deltas
    return np.mean(window, axis=0)


def forecast_next_block(
    series: np.ndarray,
    A: Optional[np.ndarray] = None,
    block_size: int = 32,
    top_k: int = 8,
    smooth_window: int = 3,
    trace: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Forecast the next block of a time series using WHT.

    Algorithm:
    1. Split series into blocks of block_size.
    2. Apply WHT to each block.
    3. Select top_k coefficients by energy.
    4. Compute deltas between consecutive blocks.
    5. Smooth deltas over last smooth_window blocks.
    6. Forecast: C_hat^{(K+1)} = C_filtered^{(K)} + delta_avg
    7. Apply inverse WHT to obtain forecast block.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    A : np.ndarray
        Normalized Walsh-Hadamard matrix.
    block_size : int
        Block length (default: 32).
    top_k : int
        Number of coefficients to retain (default: 8).
    smooth_window : int
        Smoothing window for deltas (default: 3).
    trace : bool
        If True (default), print each pipeline stage and a compact summary to stdout.
        Set False to suppress pipeline logging (e.g. batch runs or tests).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, object]]
        - forecast: Predicted next block in original domain.
        - info: Dict with blocks, coeffs, deltas, etc. for analysis.

    Raises
    ------
    ValueError
        If series is too short for forecasting (need at least 2 blocks).
    """
    if A is None:
        A = build_normalized_hadamard(block_size)
        log_trace(
            trace,
            pipeline="WHT",
            step=1,
            total_steps=8,
            title="Hadamard operator",
            detail=(
                f"constructed normalized matrix; shape={A.shape}, dtype={A.dtype}, "
                f"min={float(np.min(A)):.6g}, max={float(np.max(A)):.6g}, "
                f"Frobenius={float(np.linalg.norm(A, ord='fro')):.6g}"
            ),
        )
    else:
        log_trace(
            trace,
            pipeline="WHT",
            step=1,
            total_steps=8,
            title="Hadamard operator",
            detail=(
                f"using supplied matrix A; shape={A.shape}, dtype={A.dtype}, "
                f"min={float(np.min(A)):.6g}, max={float(np.max(A)):.6g}, "
                f"Frobenius={float(np.linalg.norm(A, ord='fro')):.6g}"
            ),
        )

    blocks = split_into_blocks(series, block_size)

    if len(blocks) < 2:
        raise ValueError("Series too short for forecasting (need at least 2 blocks)")

    lens = [len(b) for b in blocks]
    log_trace(
        trace,
        pipeline="WHT",
        step=2,
        total_steps=8,
        title="Non-overlapping block split",
        detail=(
            f"num_blocks={len(blocks)}, block_lengths={lens}, "
            f"first_block {_array_summary('x', blocks[0])}, "
            f"last_block {_array_summary('x', blocks[-1])}"
        ),
    )

    all_raw_coeffs: List[np.ndarray] = []
    all_filtered_coeffs: List[np.ndarray] = []
    energy_ratios: List[float] = []

    for block in blocks:
        raw = wht_forward(block, A)
        filtered, _ = select_top_coefficients(raw, top_k)
        e_raw = float(np.sum(raw**2))
        e_f = float(np.sum(filtered**2))
        energy_ratios.append(e_f / e_raw if e_raw > 0 else 0.0)
        all_raw_coeffs.append(raw)
        all_filtered_coeffs.append(filtered)

    raw_l2 = [float(np.linalg.norm(c)) for c in all_raw_coeffs]
    log_trace(
        trace,
        pipeline="WHT",
        step=3,
        total_steps=8,
        title="Forward WHT",
        detail=f"per-block coefficient L2 norms: {raw_l2}",
    )
    log_trace(
        trace,
        pipeline="WHT",
        step=4,
        total_steps=8,
        title="Top-k spectral masking",
        detail=(
            f"k={top_k}, retained_energy_ratio_per_block="
            f"{[round(r, 6) for r in energy_ratios]}, "
            f"mean_ratio={float(np.mean(energy_ratios)):.6g}"
        ),
    )

    deltas = compute_deltas(all_filtered_coeffs)
    delta_l2 = [float(np.linalg.norm(d)) for d in deltas]
    log_trace(
        trace,
        pipeline="WHT",
        step=5,
        total_steps=8,
        title="Inter-block coefficient deltas",
        detail=f"num_deltas={len(deltas)}, L2_norms={delta_l2}",
    )

    avg_delta = smooth_deltas(deltas, smooth_window)
    used_window = min(smooth_window, len(deltas))
    log_trace(
        trace,
        pipeline="WHT",
        step=6,
        total_steps=8,
        title="Delta smoothing",
        detail=(
            f"window={smooth_window}, effective_last_blocks={used_window}; "
            + _array_summary("avg_delta", avg_delta)
        ),
    )

    last_filtered = all_filtered_coeffs[-1]
    forecast_coeffs = last_filtered + avg_delta
    log_trace(
        trace,
        pipeline="WHT",
        step=7,
        total_steps=8,
        title="Forecast spectrum (C_hat = C_last + avg_delta)",
        detail=(
            _array_summary("C_last", last_filtered)
            + "; "
            + _array_summary("C_hat", forecast_coeffs)
        ),
    )

    forecast_block = wht_inverse(forecast_coeffs, A)
    log_trace(
        trace,
        pipeline="WHT",
        step=8,
        total_steps=8,
        title="Inverse WHT to time domain",
        detail=_array_summary("forecast_block", forecast_block),
    )

    info: Dict[str, object] = {
        "blocks": blocks,
        "all_raw_coeffs": all_raw_coeffs,
        "all_filtered_coeffs": all_filtered_coeffs,
        "deltas": deltas,
        "avg_delta": avg_delta,
        "forecast_coeffs": forecast_coeffs,
        "last_filtered": last_filtered,
    }

    return forecast_block, info
