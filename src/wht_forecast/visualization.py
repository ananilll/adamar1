"""
Visualization utilities for WHT forecasting analysis.
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wht_forecast.forecasting import forecast_next_block
from wht_forecast.metrics import compute_metrics


def plot_time_series_forecast(
    series: np.ndarray,
    forecast: np.ndarray,
    actual_next: Optional[np.ndarray],
    block_size: int,
    n_blocks_used: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot time series with forecast and optional actual next block.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    forecast : np.ndarray
        Forecast block.
    actual_next : Optional[np.ndarray]
        Actual next block (if available).
    block_size : int
        Block length.
    n_blocks_used : int
        Number of blocks used for forecasting.
    save_path : Optional[str]
        Path to save figure.
    """
    n_used = n_blocks_used * block_size
    t_series = np.arange(n_used)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_series, series[:n_used], color="steelblue", linewidth=1.2, label="Series", alpha=0.8)

    last_block_start = (n_blocks_used - 1) * block_size
    ax.axvspan(last_block_start, n_used, alpha=0.12, color="orange", label="Last block")

    t_forecast = np.arange(n_used, n_used + block_size)
    ax.plot(t_forecast, forecast, color="red", linewidth=2.0, linestyle="--", label="Forecast (WHT)", marker="o", markersize=3)

    if actual_next is not None:
        ax.plot(t_forecast, actual_next, color="green", linewidth=1.5, linestyle="-", label="Actual", alpha=0.8)

    ax.set_title("Time Series and Forecast")
    ax.set_xlabel("Time (index)")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_spectral_energy(
    last_raw: np.ndarray,
    last_filtered: np.ndarray,
    block_size: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot spectral coefficient energy (raw vs filtered).

    Parameters
    ----------
    last_raw : np.ndarray
        Raw coefficients of last block.
    last_filtered : np.ndarray
        Filtered (top-k) coefficients.
    block_size : int
        Block length.
    save_path : Optional[str]
        Path to save figure.
    """
    idx = np.arange(block_size)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(idx - 0.2, last_raw**2, width=0.4, color="steelblue", alpha=0.7, label="Raw spectrum (energy)")
    ax.bar(idx + 0.2, last_filtered**2, width=0.4, color="crimson", alpha=0.7, label="Filtered (top-k)")
    ax.set_title("Spectral Coefficient Energy (Last Block)")
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Energy C_i^2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_topk_analysis(
    series: np.ndarray,
    A: np.ndarray,
    actual_next: np.ndarray,
    block_size: int = 32,
    topk_values: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Analyze effect of top_k on forecast quality.

    Parameters
    ----------
    series : np.ndarray
        Time series.
    A : np.ndarray
        Normalized Hadamard matrix.
    actual_next : np.ndarray
        Actual next block.
    block_size : int
        Block length.
    topk_values : Optional[List[int]]
        Values of top_k to test (default: [2, 4, 8, 12, 16]).
    save_path : Optional[str]
        Path to save figure.
    """
    if topk_values is None:
        topk_values = [2, 4, 8, 12, 16]

    rmse_vals: List[float] = []
    mae_vals: List[float] = []
    forecasts: List[np.ndarray] = []

    for k in topk_values:
        forecast, _ = forecast_next_block(
            series, A, block_size=block_size, top_k=k, trace=False
        )
        m = compute_metrics(actual_next, forecast)
        rmse_vals.append(m["RMSE"])
        mae_vals.append(m["MAE"])
        forecasts.append(forecast)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(topk_values, rmse_vals, "o-", color="crimson", linewidth=2, markersize=8, label="RMSE")
    ax1.plot(topk_values, mae_vals, "s--", color="steelblue", linewidth=2, markersize=8, label="MAE")
    ax1.set_title("Forecast Error vs Number of Coefficients (top_k)")
    ax1.set_xlabel("top_k")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for k, r in zip(topk_values, rmse_vals):
        ax1.annotate(f"k={k}\nRMSE={r:.3f}", (k, r), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax2 = axes[1]
    t = np.arange(block_size)
    ax2.plot(t, actual_next, color="black", linewidth=2.5, label="Actual", zorder=5)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(topk_values)))
    for i, (k, fc) in enumerate(zip(topk_values, forecasts)):
        ax2.plot(t, fc, color=colors[i], linewidth=1.5, linestyle="--", label=f"top_k={k}", alpha=0.8)
    ax2.set_title("Forecasts at Different top_k vs Actual")
    ax2.set_xlabel("Position in block")
    ax2.set_ylabel("Value")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_method_comparison(
    actual_next: np.ndarray,
    forecast_wht: np.ndarray,
    forecast_naive: np.ndarray,
    forecast_ma: np.ndarray,
    forecast_linear: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare WHT forecast with baseline methods.

    Parameters
    ----------
    actual_next : np.ndarray
        Actual next block.
    forecast_wht : np.ndarray
        WHT forecast.
    forecast_naive : np.ndarray
        Naive forecast.
    forecast_ma : np.ndarray
        Moving average forecast.
    forecast_linear : np.ndarray
        Linear extrapolation forecast.
    save_path : Optional[str]
        Path to save figure.
    """
    block_size = len(actual_next)
    t = np.arange(block_size)

    methods: Dict[str, tuple] = {
        "WHT (proposed)": (forecast_wht, "crimson", "-", 2.5),
        "Naive": (forecast_naive, "steelblue", "--", 1.5),
        "Moving average": (forecast_ma, "orange", "-.", 1.5),
        "Linear extrapolation": (forecast_linear, "green", ":", 1.5),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(t, actual_next, color="black", linewidth=2.5, label="Actual", zorder=5)
    for name, (fc, color, ls, lw) in methods.items():
        ax1.plot(t, fc, color=color, linestyle=ls, linewidth=lw, label=name, alpha=0.85)
    ax1.set_title("Method Comparison")
    ax1.set_xlabel("Position in block")
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    names = list(methods.keys())
    rmse_vals = [compute_metrics(actual_next, fc)["RMSE"] for (fc, *_) in methods.values()]
    mae_vals = [compute_metrics(actual_next, fc)["MAE"] for (fc, *_) in methods.values()]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, rmse_vals, 0.4, label="RMSE", color="crimson", alpha=0.8)
    ax2.bar(x + 0.2, mae_vals, 0.4, label="MAE", color="steelblue", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2.set_title("Error Metrics by Method")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_results(
    series: np.ndarray,
    forecast: np.ndarray,
    actual_next: Optional[np.ndarray],
    block_size: int,
    info: Dict[str, object],
    save_path: Optional[str] = None,
) -> None:
    """
    Comprehensive results plot: time series, spectral energy, block comparison, coefficient dynamics.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    forecast : np.ndarray
        Forecast block.
    actual_next : Optional[np.ndarray]
        Actual next block.
    block_size : int
        Block length.
    info : Dict[str, object]
        Info dict from forecast_next_block.
    save_path : Optional[str]
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("WHT Time Series Analysis and Forecasting", fontsize=14, fontweight="bold", y=1.01)

    n_blocks_used = len(info["blocks"])
    n_used = n_blocks_used * block_size
    t_series = np.arange(n_used)

    ax1 = axes[0, 0]
    ax1.plot(t_series, series[:n_used], color="steelblue", linewidth=1.2, label="Series", alpha=0.8)
    last_block_start = (n_blocks_used - 1) * block_size
    ax1.axvspan(last_block_start, n_used, alpha=0.12, color="orange", label="Last block")
    t_forecast = np.arange(n_used, n_used + block_size)
    ax1.plot(t_forecast, forecast, color="red", linewidth=2.0, linestyle="--", label="Forecast (WHT)", marker="o", markersize=3)
    if actual_next is not None:
        ax1.plot(t_forecast, actual_next, color="green", linewidth=1.5, linestyle="-", label="Actual", alpha=0.8)
    ax1.set_title("Time Series and Forecast")
    ax1.set_xlabel("Time (index)")
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    last_raw = info["all_raw_coeffs"][-1]
    last_filtered = info["last_filtered"]
    idx = np.arange(block_size)
    ax2.bar(idx - 0.2, last_raw**2, width=0.4, color="steelblue", alpha=0.7, label="Raw spectrum")
    ax2.bar(idx + 0.2, last_filtered**2, width=0.4, color="crimson", alpha=0.7, label="Filtered (top-k)")
    ax2.set_title("Spectral Energy (Last Block)")
    ax2.set_xlabel("Coefficient index")
    ax2.set_ylabel("Energy C_i^2")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    t_block = np.arange(block_size)
    ax3.plot(t_block, info["blocks"][-1], color="steelblue", linewidth=1.5, label="Last block", marker="s", markersize=4)
    ax3.plot(t_block, forecast, color="red", linewidth=2.0, linestyle="--", label="Forecast", marker="o", markersize=4)
    if actual_next is not None:
        ax3.plot(t_block, actual_next, color="green", linewidth=1.5, label="Actual next", marker="^", markersize=4)
        metrics = compute_metrics(actual_next, forecast)
        ax3.set_title(f"Block Comparison\nMAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
    else:
        ax3.set_title("Block Comparison")
    ax3.set_xlabel("Position in block")
    ax3.set_ylabel("Value")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    n_blocks = len(info["all_filtered_coeffs"])
    n_show = min(5, block_size)
    colors = plt.cm.Set1(np.linspace(0, 1, n_show))
    for i in range(n_show):
        vals = [info["all_filtered_coeffs"][k][i] for k in range(n_blocks)]
        ax4.plot(range(n_blocks), vals, color=colors[i], linewidth=1.5, marker="o", markersize=4, label=f"C[{i}]")
    ax4.set_title("Leading Coefficient Dynamics Across Blocks")
    ax4.set_xlabel("Block index")
    ax4.set_ylabel("Coefficient value")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
