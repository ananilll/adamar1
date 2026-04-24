"""
WHT Forecast: Walsh-Hadamard Transform for time series forecasting.

A research-grade implementation for scientific use.
"""

from wht_forecast.hadamard import build_hadamard_matrix, build_normalized_hadamard
from wht_forecast.transform import wht_forward, wht_inverse
from wht_forecast.blocks import split_into_blocks
from wht_forecast.filtering import select_top_coefficients, compute_energy
from wht_forecast.forecasting import forecast_next_block
from wht_forecast.metrics import compute_metrics
from wht_forecast.baselines import (
    naive_forecast,
    moving_average_forecast,
    linear_extrapolation_forecast,
)
from wht_forecast.synthetic_data import generate_synthetic_series
from wht_forecast.data_loader import (
    load_csv_series,
    load_time_series_from_csv,
    normalize_series,
)

__all__ = [
    "build_hadamard_matrix",
    "build_normalized_hadamard",
    "wht_forward",
    "wht_inverse",
    "split_into_blocks",
    "select_top_coefficients",
    "compute_energy",
    "forecast_next_block",
    "compute_metrics",
    "naive_forecast",
    "moving_average_forecast",
    "linear_extrapolation_forecast",
    "generate_synthetic_series",
    "load_csv_series",
    "load_time_series_from_csv",
    "normalize_series",
]

__version__ = "0.1.0"
