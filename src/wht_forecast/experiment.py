"""
Full experiment orchestration: data, forecasting, baselines, metrics, visualization.
"""

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

from wht_forecast.baselines import (
    linear_extrapolation_forecast,
    moving_average_forecast,
    naive_forecast,
)
from wht_forecast.data_loader import (
    load_time_series_from_csv,
    normalize_series,
    validate_series_for_forecasting,
)
from wht_forecast.filtering import select_top_coefficients
from wht_forecast.forecasting import forecast_next_block
from wht_forecast.hadamard import build_hadamard_matrix, build_normalized_hadamard
from wht_forecast.metrics import compute_metrics
from wht_forecast.synthetic_data import generate_synthetic_series
from wht_forecast.trace_log import log_trace
from wht_forecast.transform import wht_forward, wht_inverse
from wht_forecast.visualization import (
    plot_method_comparison,
    plot_results,
    plot_topk_analysis,
)


def run_experiment(
    block_size: int = 32,
    top_k: int = 8,
    smooth_window: int = 3,
    series_length: int = 512,
    noise_level: float = 0.3,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    csv_path: Optional[str] = None,
    value_column: Optional[str] = None,
    normalize: Literal["zscore", "minmax", "none"] = "none",
    trace_pipeline: bool = True,
) -> Dict[str, object]:
    """
    Run full WHT forecasting experiment.

    Parameters
    ----------
    block_size : int
        Block length (default: 32).
    top_k : int
        Number of coefficients to retain (default: 8).
    smooth_window : int
        Smoothing window for deltas (default: 3).
    series_length : int
        Length of synthetic series when csv_path is None (default: 512).
    noise_level : float
        Noise level for synthetic data (default: 0.3).
    seed : int
        Random seed for synthetic data (default: 42).
    output_dir : Optional[Path]
        Directory for saving plots. If None, plots are not saved.
    csv_path : Optional[str]
        Path to CSV file. If provided, load series from CSV instead of synthetic.
    value_column : Optional[str]
        Column name for values in CSV. Auto-detected if None.
    normalize : Literal["zscore", "minmax", "none"]
        Preprocessing normalization (default: "none").
    trace_pipeline : bool
        If True (default), print experiment stages and WHT step-by-step trace to stdout.
        Set False for quiet runs (matches ``forecast_next_block(..., trace=False)``).

    Returns
    -------
    Dict[str, object]
        Results dict with forecasts, metrics, info.
    """
    output_dir = output_dir or Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is not None:
        series = load_time_series_from_csv(
            csv_path,
            value_column=value_column,
            verbose=True,
        )
    else:
        series = generate_synthetic_series(n=series_length, seed=seed, noise_level=noise_level)

    if trace_pipeline:
        s0 = np.asarray(series, dtype=float)
        log_trace(
            trace_pipeline,
            pipeline="EXP",
            title="Series after ingest",
            detail=(
                f"n={s0.size}, min={float(np.min(s0)):.6g}, max={float(np.max(s0)):.6g}, "
                f"mean={float(np.mean(s0)):.6g}"
            ),
        )

    validate_series_for_forecasting(series, block_size)
    series = normalize_series(series, method=normalize)

    if trace_pipeline:
        s1 = np.asarray(series, dtype=float)
        log_trace(
            trace_pipeline,
            pipeline="EXP",
            title="Series after validation and normalization",
            detail=(
                f"method={normalize}, n={s1.size}, min={float(np.min(s1)):.6g}, "
                f"max={float(np.max(s1)):.6g}, mean={float(np.mean(s1)):.6g}"
            ),
        )

    A = build_normalized_hadamard(block_size)

    n_blocks = len(series) // block_size
    train_series = series[: (n_blocks - 1) * block_size]
    actual_next = series[(n_blocks - 1) * block_size : n_blocks * block_size]

    if trace_pipeline:
        log_trace(
            trace_pipeline,
            pipeline="EXP",
            title="Train / holdout split",
            detail=(
                f"full_blocks_in_series={n_blocks}, train_len={train_series.size}, "
                f"holdout_len={actual_next.size}, block_size={block_size}"
            ),
        )

    forecast_wht, info = forecast_next_block(
        train_series,
        A=A,
        block_size=block_size,
        top_k=top_k,
        smooth_window=smooth_window,
        trace=trace_pipeline,
    )

    forecast_naive = naive_forecast(train_series, block_size)
    forecast_ma = moving_average_forecast(train_series, block_size, window=3)
    forecast_linear = linear_extrapolation_forecast(train_series, block_size)

    if trace_pipeline:
        log_trace(
            trace_pipeline,
            pipeline="EXP",
            title="Baseline forecasts (L2 norms)",
            detail=(
                f"actual={float(np.linalg.norm(actual_next)):.6g}, "
                f"wht={float(np.linalg.norm(forecast_wht)):.6g}, "
                f"naive={float(np.linalg.norm(forecast_naive)):.6g}, "
                f"moving_avg={float(np.linalg.norm(forecast_ma)):.6g}, "
                f"linear={float(np.linalg.norm(forecast_linear)):.6g}"
            ),
        )

    metrics_wht = compute_metrics(actual_next, forecast_wht)
    metrics_naive = compute_metrics(actual_next, forecast_naive)
    metrics_ma = compute_metrics(actual_next, forecast_ma)
    metrics_linear = compute_metrics(actual_next, forecast_linear)

    if trace_pipeline:
        log_trace(
            trace_pipeline,
            pipeline="EXP",
            title="Holdout metrics summary",
            detail=(
                f"wht MAE={metrics_wht['MAE']:.6g} RMSE={metrics_wht['RMSE']:.6g}; "
                f"naive MAE={metrics_naive['MAE']:.6g}; "
                f"moving_avg MAE={metrics_ma['MAE']:.6g}; "
                f"linear MAE={metrics_linear['MAE']:.6g}"
            ),
        )

    results: Dict[str, object] = {
        "forecast_wht": forecast_wht,
        "forecast_naive": forecast_naive,
        "forecast_ma": forecast_ma,
        "forecast_linear": forecast_linear,
        "actual_next": actual_next,
        "metrics_wht": metrics_wht,
        "metrics_naive": metrics_naive,
        "metrics_ma": metrics_ma,
        "metrics_linear": metrics_linear,
        "info": info,
        "series": series,
        "train_series": train_series,
    }

    if output_dir:
        plot_results(
            train_series,
            forecast_wht,
            actual_next,
            block_size,
            info,
            save_path=str(output_dir / "forecast.png"),
        )
        plot_topk_analysis(
            train_series,
            A,
            actual_next,
            block_size=block_size,
            topk_values=[2, 4, 6, 8, 12, 16],
            save_path=str(output_dir / "topk_analysis.png"),
        )
        plot_method_comparison(
            actual_next,
            forecast_wht,
            forecast_naive,
            forecast_ma,
            forecast_linear,
            save_path=str(output_dir / "method_comparison.png"),
        )

    return results


def print_metrics_table(results: Dict[str, object]) -> None:
    """Print formatted metrics table to stdout."""
    methods = [
        ("WHT (proposed)", results["metrics_wht"]),
        ("Naive", results["metrics_naive"]),
        ("Moving average", results["metrics_ma"]),
        ("Linear extrapolation", results["metrics_linear"]),
    ]
    print(f"{'Method':<25} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
    print("-" * 50)
    for name, m in methods:
        print(f"{name:<25} {m['MAE']:8.4f} {m['RMSE']:8.4f} {m['MAPE']:8.2f}")
    print("=" * 50)


def run_small_numerical_example() -> Dict[str, object]:
    """
    Small numerical example for thesis illustration (block size 8).

    Demonstrates: Hadamard matrix, WHT, top-k selection, delta, forecast.
    """
    n = 8
    H = build_hadamard_matrix(n)
    A = build_normalized_hadamard(n)

    X1 = np.array([1.0, 3.0, 2.0, 4.0, 1.5, 2.5, 3.5, 1.0])
    X2 = np.array([1.2, 3.3, 2.1, 4.2, 1.6, 2.6, 3.7, 1.1])
    X3_actual = np.array([1.4, 3.6, 2.3, 4.5, 1.7, 2.8, 3.9, 1.2])

    top_k = 3
    C1 = wht_forward(X1, A)
    C1_filtered, top_idx = select_top_coefficients(C1, top_k)
    C2 = wht_forward(X2, A)
    C2_filtered, _ = select_top_coefficients(C2, top_k)

    delta = C2_filtered - C1_filtered
    C3_forecast = C2_filtered + delta
    X3_forecast = wht_inverse(C3_forecast, A)

    metrics = compute_metrics(X3_actual, X3_forecast)

    return {
        "H": H,
        "A": A,
        "X1": X1,
        "X2": X2,
        "X3_actual": X3_actual,
        "X3_forecast": X3_forecast,
        "C1": C1,
        "C2": C2,
        "C1_filtered": C1_filtered,
        "C2_filtered": C2_filtered,
        "delta": delta,
        "C3_forecast": C3_forecast,
        "metrics": metrics,
    }
