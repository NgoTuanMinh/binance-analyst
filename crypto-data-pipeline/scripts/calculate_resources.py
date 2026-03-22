"""Calculate storage/resource estimates for full-scale collection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_file_size_str


def _avg_zip_size_for_symbol(raw_dir: Path, symbol: str = "BTCUSDT") -> int:
    samples = list((raw_dir / symbol).rglob("*.zip"))
    if not samples:
        return 300_000
    size = sum(p.stat().st_size for p in samples[:300]) / min(300, len(samples))
    return int(size)


def _estimate(
    avg_file_size: int, symbols: int, intervals: int, years: int, include_leap: bool = True
) -> dict[str, Any]:
    days = years * 365 + (1 if include_leap else 0)
    total_files = symbols * intervals * days
    raw_bytes = total_files * avg_file_size
    processed_bytes = int(raw_bytes * 1.2)
    merged_bytes = int(raw_bytes * 0.45)
    total_bytes = raw_bytes + processed_bytes + merged_bytes
    return {
        "days": days,
        "total_files": total_files,
        "raw_bytes": raw_bytes,
        "processed_bytes": processed_bytes,
        "merged_bytes": merged_bytes,
        "total_estimated_bytes": total_bytes,
    }


def _recommend_batch_size(total_bytes: int, ram_gb: int, disk_free_bytes: int) -> dict[str, Any]:
    # Conservative heuristic: each batch should target <= 10% free disk and <= 25% RAM pressure.
    disk_cap = max(1, int(disk_free_bytes * 0.10))
    ram_cap = max(1, int(ram_gb * (1024**3) * 0.25))
    effective_cap = min(disk_cap, ram_cap)
    ratio = effective_cap / max(1, total_bytes)
    # Baseline for 300 symbols; clamp 10..80
    suggested = int(max(10, min(80, math.floor(300 * ratio))))
    if suggested < 50:
        mode = "safe"
    elif suggested <= 60:
        mode = "balanced"
    else:
        mode = "aggressive"
    return {
        "suggested_batch_size": suggested,
        "strategy": mode,
        "note": "Use lower batch size when disk usage > 80% or frequent 429 responses.",
    }


def main() -> None:
    """CLI entrypoint for resource estimation."""
    parser = argparse.ArgumentParser(description="Estimate full-scale resource requirements")
    parser.add_argument("--symbols", type=int, default=300)
    parser.add_argument("--intervals", type=int, default=3)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--sample-symbol", type=str, default="BTCUSDT")
    parser.add_argument("--raw-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--ram-gb", type=int, default=16)
    parser.add_argument("--disk-free-gb", type=int, default=200)
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "logs" / "resource_estimate.json"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    avg_size = _avg_zip_size_for_symbol(raw_dir, args.sample_symbol)
    estimate = _estimate(avg_size, args.symbols, args.intervals, args.years)
    recommendation = _recommend_batch_size(
        estimate["total_estimated_bytes"],
        args.ram_gb,
        args.disk_free_gb * (1024**3),
    )

    report = {
        "inputs": vars(args),
        "sample_avg_zip_size": avg_size,
        "sample_avg_zip_size_human": get_file_size_str(avg_size),
        "estimate": {
            **estimate,
            "raw_human": get_file_size_str(estimate["raw_bytes"]),
            "processed_human": get_file_size_str(estimate["processed_bytes"]),
            "merged_human": get_file_size_str(estimate["merged_bytes"]),
            "total_human": get_file_size_str(estimate["total_estimated_bytes"]),
        },
        "recommendation": recommendation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
