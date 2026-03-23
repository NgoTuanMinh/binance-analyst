#!/usr/bin/env python3
"""
Build ML feature Parquet files via :class:`features.pipeline.FeaturePipeline`.

Example::

    python scripts/generate_features.py --symbols top_100 --intervals 15m 1h 4h \\
        --start 2020-01-01 --end 2025-12-31

    python scripts/generate_features.py --symbols BTCUSDT,ETHUSDT --intervals M15 H1 H4 \\
        --start 2024-01-01 --end 2024-06-30 --output-dir features_output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    from features.pipeline import FeaturePipeline

    parser = argparse.ArgumentParser(description="Generate feature Parquet (FeaturePipeline)")
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated symbols, top_100, top_300, or path to JSON list",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["15m", "1h", "4h"],
        help="Timeframes to load (must include 15m). Default: 15m 1h 4h",
    )
    parser.add_argument("--start", type=str, required=True, help="Start date (UTC), e.g. 2020-01-01")
    parser.add_argument("--end", type=str, required=True, help="End date (UTC) inclusive day if date-only")
    parser.add_argument(
        "--merged-dir",
        type=str,
        default=str(_REPO_ROOT / "crypto-data-pipeline" / "data" / "merged"),
        help="Merged Parquet root (default: crypto-data-pipeline/data/merged)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "features" / "data"),
        help="Output directory for {SYMBOL}_features.parquet (default: features/data)",
    )
    parser.add_argument("--no-targets", action="store_true", help="Skip target columns")
    parser.add_argument("--no-trade-targets", action="store_true", help="Skip swing-strategy trade targets")
    parser.add_argument("--compression", type=str, default="snappy")
    parser.add_argument(
        "--correlation-target",
        type=str,
        default=None,
        help="If set, print JSON top-20 |correlation| with this target column after build",
    )
    parser.add_argument("--correlation-method", choices=("pearson", "spearman"), default="pearson")
    args = parser.parse_args()

    pl = FeaturePipeline(
        args.symbols,
        args.intervals,
        args.start,
        args.end,
        merged_dir=Path(args.merged_dir),
        output_dir=Path(args.output_dir),
        include_targets=not args.no_targets,
        include_trade_targets=not args.no_trade_targets,
    )
    pl.load_data()
    pl.generate_all_features()
    paths = pl.save_features(compression=args.compression)
    for sym, p in sorted(paths.items()):
        print(p)

    if args.correlation_target:
        rank = pl.get_feature_importance_ranking(
            args.correlation_target,
            method=args.correlation_method,
        )
        top = rank.head(20).to_dict(orient="records")
        print(json.dumps(top, indent=2))


if __name__ == "__main__":
    main()
