# WHT Forecast: Walsh-Hadamard Transform for Time Series Forecasting

A research-grade Python implementation of time series forecasting using the Walsh-Hadamard Transform (WHT). Designed for scientific research and thesis work. Current package version: **0.1.0** (see `pyproject.toml` and `wht_forecast.__version__`).

## Algorithm Overview

The forecasting pipeline:

1. **Build** normalized Walsh-Hadamard matrix
2. **Split** time series into non-overlapping blocks
3. **Apply** WHT transform to each block
4. **Select** top-k spectral coefficients by energy
5. **Compute** coefficient deltas between consecutive blocks
6. **Smooth** deltas over a sliding window
7. **Forecast** next block coefficients
8. **Apply** inverse WHT to obtain forecast
9. **Evaluate** the holdout block using MAE, MSE, RMSE, MAPE (WHT forecast only; `run_experiment` and the CLI print metrics for the proposed method only)

**Baseline-style helpers** (naive, moving average, linear extrapolation) live in `baselines.py` for ad-hoc use or custom scripts; they are **not** computed or shown by `run_experiment` or `python -m wht_forecast.cli run-experiment`.

By default, the library prints **structured, color-highlighted traces** for each pipeline stage (WHT steps and experiment milestones). Disable with `trace=False` / `trace_pipeline=False`, or use `--quiet` on the CLI.

## Mathematical Foundation

**Hadamard matrix** (recursive construction):

```
H_1 = [1]
H_{2n} = [ H_n   H_n ]
         [ H_n  -H_n ]
```

**Normalized matrix**: `A = H / sqrt(n)` with property `A @ A.T = I`

**Forward transform**: `C = A @ X`  
**Inverse transform**: `X = A.T @ C`

## Algorithms and methods beyond the Hadamard transform

The **Walsh–Hadamard** core is: recursive matrix construction in `hadamard.py` and the forward/inverse WHT in `transform.py` (orthogonal mapping in the `A` / `A.T` form above). The rest of the library layers **heuristics, baselines, and utilities** on top; none of it replaces the Hadamard basis with another transform.

| Area | What it is | Where |
|------|------------|--------|
| Forecast pipeline | Non-overlapping **block split**; **top-k** by coefficient energy (masking); **deltas** between filtered spectra; **sliding mean** of deltas; spectrum forecast `C_hat = C_last + avg_delta`; **inverse WHT** back to time | `blocks.py`, `filtering.py`, `forecasting.py` |
| Baselines (optional, not in `run_experiment`) | Same three methods, for manual / programmatic comparison | `baselines.py` (not used by the default CLI experiment) |
| Quality metrics | **MAE, MSE, RMSE, MAPE** (evaluation, not a transform) | `metrics.py` |
| Preprocessing | **Z-score** and **min–max** scaling; CSV loading and number cleaning (e.g. European decimals) | `data_loader.py` |
| Test data | **Synthetic series**: sinusoids (trend, seasonal, high-frequency) + **Gaussian noise** | `synthetic_data.py` |

*If you need only the Hadamard / WHT math*, use `build_hadamard_matrix` / `build_normalized_hadamard` and `wht_forward` / `wht_inverse` from the public API. The forecasting pipeline and CSV-backed `run_experiment` use the WHT method only for holdout evaluation; import `baselines` yourself if you want those comparisons.

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Quick Start

### CLI

```bash
# Run full experiment with synthetic data (default)
python -m wht_forecast.cli run-experiment

# Run on CSV file (use any path; examples below)
python -m wht_forecast.cli run-experiment --csv path/to/your_series.csv

# With custom parameters
python -m wht_forecast.cli run-experiment \
    --csv data.csv \
    --value-column value \
    --block-size 32 \
    --top-k 8 \
    --smooth-window 3 \
    --normalize zscore \
    --output-dir ./outputs

# Suppress stage-by-stage console traces (they are on by default)
python -m wht_forecast.cli run-experiment --quiet
```

**Output plots** (written to `--output-dir`, default `./outputs`):

- `forecast.png` — time series, spectral energy, block comparison, coefficient dynamics
- `topk_analysis.png` — error vs `top_k` and curves vs actual
- `wht_vs_actual.png` — **holdout** actual vs WHT forecast and WHT-only RMSE/MAE bars (replaces the older multi-method comparison figure)

### Console logging

- **Default:** `forecast_next_block` and `run_experiment` print English trace lines to stdout: `[WHT] Stage k/8 …` for the core algorithm and `[EXP] …` for ingest, train/holdout split, holdout L2 norms (actual vs WHT), and WHT holdout metrics.
- **Highlighting:** ANSI colors when stdout is a TTY. Set `NO_COLOR` (see [no-color.org](https://no-color.org/)) to force plain text; set `FORCE_COLOR=1` to keep colors when output is piped.
- **Quiet mode:** pass `trace=False` to `forecast_next_block`, `trace_pipeline=False` to `run_experiment`, or `--quiet` on `run-experiment`.

### Python API

```python
from wht_forecast import (
    build_normalized_hadamard,
    forecast_next_block,
    generate_synthetic_series,
    compute_metrics,
)

# Generate synthetic data
series = generate_synthetic_series(n=512, seed=42, noise_level=0.2)

# Build Hadamard matrix and forecast (A optional - built internally if omitted).
# trace=True by default; use trace=False for silent calls.
forecast, info = forecast_next_block(
    series, block_size=32, top_k=8, smooth_window=3
)
# forecast, info = forecast_next_block(..., trace=False)

# Evaluate (with holdout)
actual = series[480:512]  # last block
metrics = compute_metrics(actual, forecast)
print(f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
```

## Project Structure

```
project_root/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── wht_forecast/
│       ├── __init__.py
│       ├── hadamard.py      # Hadamard matrix construction
│       ├── transform.py     # Forward/inverse WHT
│       ├── blocks.py        # Block splitting
│       ├── filtering.py     # Top-k coefficient selection
│       ├── forecasting.py   # Main forecasting algorithm
│       ├── metrics.py       # MAE, MSE, RMSE, MAPE
│       ├── baselines.py     # Naive, MA, linear baselines
│       ├── data_loader.py   # CSV loading, normalization
│       ├── synthetic_data.py
│       ├── visualization.py
│       ├── trace_log.py     # Colored stdout traces (NO_COLOR / FORCE_COLOR)
│       ├── experiment.py    # run_experiment (WHT holdout), small example
│       └── cli.py           # Command-line interface
├── experiments/
│   └── run_experiment.py
└── notebooks/
    └── exploration.ipynb
```

`notebooks/` may be missing in some checkouts. The `data/` directory is often gitignored; place CSVs locally (see [Example Data](#example-data)).

## Running on Real Datasets

Load time series from CSV files. Supported formats:

**Option A: timestamp,value**
```csv
timestamp,value
2023-01-01,10
2023-01-02,11
2023-01-03,9
```

**Option B: date,value**
```csv
date,value
2023-01-01,10
2023-01-02,11
```

**Option C: single column**
```csv
value
10
11
9
```

### CLI Usage

```bash
# Auto-detect value column
python -m wht_forecast.cli run-experiment --csv data.csv

# Specify value column explicitly
python -m wht_forecast.cli run-experiment --csv data.csv --value-column sales

# With preprocessing (z-score or min-max normalization)
python -m wht_forecast.cli run-experiment --csv data.csv --normalize zscore
```

### Python API

```python
from wht_forecast import load_time_series_from_csv, normalize_series, forecast_next_block, compute_metrics

# Load with full cleaning (prints diagnostics)
series = load_time_series_from_csv("data.csv")

# Or load silently
series = load_time_series_from_csv("data.csv", verbose=False)
series = normalize_series(series, method="zscore")

# Forecast (requires at least 64 values)
forecast, info = forecast_next_block(series, block_size=32, top_k=8)
```

### Data Loader Features

- **Price column priority**: Close → close → Adj Close → price → value (else last numeric)
- **European format**: `"19403,9"` and `"19 403,9"` (comma decimal, space thousands)
- **Date sorting**: If Date column exists, parses and sorts chronologically
- **Validation**: Loader requires at least **64** values; `run_experiment` with default `block_size=32` further requires **3 full blocks** (96 points) for train/holdout split—use a longer series or a smaller `--block-size`
- **Compatible with**: Yahoo Finance, MetaTrader, TradingView, generic CSV

### Example Data

The `data/` directory is listed in `.gitignore`, so **sample CSVs are not guaranteed to be in a fresh clone**; add your own files locally or use any path in `--csv`.

If you keep examples under `data/`, typical names are:
- `data/example_timestamp_value.csv` — timestamp,value format
- `data/example_single_column.csv` — single value column

## Performance

Validated for:
- Series length up to 100,000
- Block sizes: 8, 16, 32, 64, 128

## Architecture

The project follows a modular design with clear separation of concerns:

| Module | Responsibility |
|--------|----------------|
| `hadamard.py` | Pure math: Walsh-Hadamard matrix construction (H, A) |
| `transform.py` | Forward/inverse WHT (C = A@X, X = A.T@C) |
| `blocks.py` | Time series splitting into non-overlapping blocks |
| `filtering.py` | Top-k coefficient selection by energy |
| `forecasting.py` | Full pipeline: blocks → WHT → top-k → deltas → smooth → forecast |
| `metrics.py` | MAE, MSE, RMSE, MAPE |
| `baselines.py` | Naive, moving average, linear extrapolation |
| `data_loader.py` | CSV loading, normalization (zscore, minmax) |
| `synthetic_data.py` | Reproducible synthetic series generation |
| `visualization.py` | Matplotlib plots; includes `plot_wht_vs_actual` (holdout actual vs WHT and WHT error bars) and related helpers |
| `experiment.py` | Orchestrates CSV/synthetic `run_experiment` (WHT holdout metrics only) + small numerical example |
| `trace_log.py` | Optional ANSI-colored trace lines for pipeline visibility |
| `cli.py` | Argument parsing and CLI entry point (`--quiet` suppresses traces) |

**Data flow**: `series` → `blocks` → `WHT` → `top-k` → `deltas` → `smooth` → `forecast_coeffs` → `inverse_WHT` → `forecast_block`

## License

MIT
