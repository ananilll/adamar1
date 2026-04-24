#!/usr/bin/env python3
"""
Run WHT forecasting experiment from the experiments directory.

Usage:
    python experiments/run_experiment.py
    python experiments/run_experiment.py --csv data/dataset.csv
    python experiments/run_experiment.py --quiet
    python -m wht_forecast.cli run-experiment --csv data.csv --block-size 32
"""

import argparse
import sys
from pathlib import Path

# Add src to path when run as script (before pip install)
_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from wht_forecast.experiment import print_metrics_table, run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WHT forecasting experiment")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file. If omitted, use synthetic data.",
    )
    parser.add_argument(
        "--value-column",
        type=str,
        default=None,
        help="Column name for values in CSV. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block length (default: 32)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of coefficients to retain (default: 8)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Smoothing window for deltas (default: 3)",
    )
    parser.add_argument(
        "--series-length",
        type=int,
        default=512,
        help="Length of synthetic series when --csv not used (default: 512)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.3,
        help="Noise level for synthetic data (default: 0.3)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["zscore", "minmax", "none"],
        default="none",
        help="Preprocessing normalization (default: none)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: outputs/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pipeline traces (traces are on by default; see NO_COLOR / FORCE_COLOR)",
    )

    args = parser.parse_args()
    output_dir = args.output_dir or (_project_root / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(
        block_size=args.block_size,
        top_k=args.top_k,
        smooth_window=args.smooth_window,
        series_length=args.series_length,
        noise_level=args.noise,
        seed=42,
        output_dir=output_dir,
        csv_path=args.csv,
        value_column=args.value_column,
        normalize=args.normalize,
        trace_pipeline=not args.quiet,
    )

    print("\n" + "=" * 50)
    print("METRICS")
    print("=" * 50)
    print_metrics_table(results)
    print(f"\nPlots saved to: {output_dir}")
    print("  - forecast.png")
    print("  - topk_analysis.png")
    print("  - method_comparison.png")
