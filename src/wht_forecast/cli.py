"""
Command-line interface for WHT forecasting experiments.
"""

import argparse
from pathlib import Path

from wht_forecast.experiment import print_metrics_table, run_experiment


def main() -> None:
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="WHT Forecast: Walsh-Hadamard Transform for time series forecasting"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run-experiment", help="Run full forecasting experiment")
    run_parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to CSV file (timestamp,value or single column). If omitted, use synthetic data.",
    )
    run_parser.add_argument(
        "--value-column",
        type=str,
        default=None,
        metavar="COLUMN",
        help="Column name for values in CSV. Auto-detected if not specified.",
    )
    run_parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block length (default: 32)",
    )
    run_parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of coefficients to retain (default: 8)",
    )
    run_parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Smoothing window for deltas (default: 3)",
    )
    run_parser.add_argument(
        "--series-length",
        type=int,
        default=512,
        help="Length of synthetic series when --csv not used (default: 512)",
    )
    run_parser.add_argument(
        "--noise",
        type=float,
        default=0.3,
        help="Noise level for synthetic data (default: 0.3)",
    )
    run_parser.add_argument(
        "--normalize",
        type=str,
        choices=["zscore", "minmax", "none"],
        default="none",
        help="Preprocessing normalization: zscore, minmax, or none (default: none)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory for plots (default: ./outputs)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data (default: 42)",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress color-highlighted pipeline traces (on by default). "
            "NO_COLOR disables ANSI; FORCE_COLOR=1 forces colors when not a TTY."
        ),
    )
    run_parser.add_argument(
        "--no-pad-remainder",
        action="store_true",
        help=(
            "Do not pad the series to a multiple of block size; "
            "incomplete tail is ignored (legacy behavior)."
        ),
    )
    run_parser.add_argument(
        "--remainder-pad",
        type=str,
        choices=["repeat_last", "zeros"],
        default="repeat_last",
        metavar="MODE",
        help=(
            "When padding incomplete tail: repeat_last (default) or zeros "
            "(ignored with --no-pad-remainder)."
        ),
    )

    args = parser.parse_args()

    if args.command == "run-experiment":
        print("=" * 70)
        print("WHT TIME SERIES FORECASTING EXPERIMENT")
        print("=" * 70)
        print(f"\nParameters: block_size={args.block_size}, top_k={args.top_k}, "
              f"smooth_window={args.smooth_window}")
        if args.csv:
            print(f"Data source: CSV file: {args.csv}")
            if args.value_column:
                print(f"Value column: {args.value_column}")
        else:
            print(f"Data source: synthetic (length={args.series_length}, noise={args.noise})")
        if args.normalize != "none":
            print(f"Normalization: {args.normalize}")
        if not args.no_pad_remainder:
            print(f"Remainder pad: {args.remainder_pad}")
        print()

        results = run_experiment(
            block_size=args.block_size,
            top_k=args.top_k,
            smooth_window=args.smooth_window,
            series_length=args.series_length,
            noise_level=args.noise,
            seed=args.seed,
            output_dir=args.output_dir,
            csv_path=args.csv,
            value_column=args.value_column,
            normalize=args.normalize,
            trace_pipeline=not args.quiet,
            pad_remainder=not args.no_pad_remainder,
            remainder_pad_mode=args.remainder_pad,
        )

        print("\nMetrics:")
        print_metrics_table(results)
        print(f"\nPlots saved to: {args.output_dir}")
        print("  - forecast.png")
        print("  - topk_analysis.png")
        print("  - wht_vs_actual.png")
        print("\nDone.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
