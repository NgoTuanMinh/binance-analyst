"""Batch downloader for full-scale symbol collection with checkpoint/resume."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.downloader import BinanceDataDownloader
from src.extractor import DataExtractor
from src.merger import DataMerger
from src.utils import ensure_dir, get_file_size_str, load_config, save_json


def _load_symbols(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("symbols file must be a JSON list")
    return [str(item) for item in data]


def _chunked(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _estimate_resources(
    symbols: int, intervals: int, days: int, sample_bytes_per_file: int
) -> dict[str, Any]:
    total_files = symbols * intervals * days
    estimated = total_files * sample_bytes_per_file
    return {
        "estimated_files": total_files,
        "estimated_bytes": estimated,
        "estimated_size_human": get_file_size_str(estimated),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"completed_batches": [], "started_at": int(time.time())}


def _sample_file_size(download_dir: Path) -> int:
    sample = list(download_dir.rglob("*.zip"))
    if not sample:
        return 300_000
    return int(sum(p.stat().st_size for p in sample[:100]) / min(100, len(sample)))


def main() -> None:
    """CLI entrypoint for large-scale batch downloads."""
    parser = argparse.ArgumentParser(description="Batch download symbols with resume")
    parser.add_argument("--symbols-file", default=str(PROJECT_ROOT / settings.SYMBOLS_FILE))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--intervals", nargs="+", default=settings.INTERVALS)
    parser.add_argument(
        "--checkpoint-file",
        default=str(PROJECT_ROOT / settings.LOG_DIR / "batch_download_checkpoint.json"),
    )
    parser.add_argument("--max-workers", type=int, default=settings.MAX_WORKERS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    conf = load_config()
    start_date = args.start or conf["data_start_date"]
    end_date = args.end or conf["data_end_date"]

    symbols = _load_symbols(Path(args.symbols_file))
    batches = _chunked(symbols, args.batch_size)
    checkpoint_path = Path(args.checkpoint_file)
    ensure_dir(checkpoint_path.parent)
    checkpoint = _load_checkpoint(checkpoint_path)
    done = set(checkpoint.get("completed_batches", []))

    downloader = BinanceDataDownloader(
        {
            "base_url": settings.BASE_URL,
            "download_dir": str(PROJECT_ROOT / settings.DOWNLOAD_DIR),
            "max_retries": settings.MAX_RETRIES,
            "rate_limit_sleep": settings.RATE_LIMIT_SLEEP,
            "max_workers": args.max_workers,
            "start_date": start_date,
            "end_date": end_date,
            "quiet": args.quiet,
        }
    )
    extractor = DataExtractor(
        str(PROJECT_ROOT / settings.DOWNLOAD_DIR), str(PROJECT_ROOT / settings.EXTRACT_DIR)
    )
    merger = DataMerger(str(PROJECT_ROOT / settings.EXTRACT_DIR), str(PROJECT_ROOT / settings.MERGE_DIR))

    days = len(downloader.generate_date_range(start_date, end_date))
    sample_size = _sample_file_size(PROJECT_ROOT / settings.DOWNLOAD_DIR)
    estimate = _estimate_resources(len(symbols), len(args.intervals), days, sample_size)
    print("=== Resource estimation ===")
    print(json.dumps(estimate, indent=2))

    for batch_idx, batch_symbols in enumerate(batches):
        if batch_idx in done:
            print(f"Skip completed batch {batch_idx + 1}/{len(batches)}")
            continue

        print(f"Running batch {batch_idx + 1}/{len(batches)} with {len(batch_symbols)} symbols")
        started = time.time()
        download_stats = downloader.download_batch(batch_symbols, args.intervals, start_date, end_date)
        extract_stats = extractor.extract_all()
        merge_stats = merger.merge_all(batch_symbols, args.intervals)

        checkpoint.setdefault("batches", []).append(
            {
                "batch_index": batch_idx,
                "symbols": batch_symbols,
                "download": download_stats,
                "extract": extract_stats,
                "merge": merge_stats,
                "elapsed_seconds": round(time.time() - started, 2),
                "finished_at": int(time.time()),
            }
        )
        checkpoint["completed_batches"] = sorted(
            list(set(checkpoint.get("completed_batches", []) + [batch_idx]))
        )
        checkpoint["progress"] = {
            "total_batches": len(batches),
            "completed_batches": len(checkpoint["completed_batches"]),
            "percent": round(len(checkpoint["completed_batches"]) / len(batches) * 100, 2),
            "remaining_batches": len(batches) - len(checkpoint["completed_batches"]),
            "remaining_symbols_estimate": (
                len(symbols) - (len(checkpoint["completed_batches"]) * args.batch_size)
            ),
        }
        save_json(checkpoint_path, checkpoint)

        completed = len(checkpoint["completed_batches"])
        eta_batches = len(batches) - completed
        avg_seconds = sum(b["elapsed_seconds"] for b in checkpoint.get("batches", [])) / max(
            1, len(checkpoint.get("batches", []))
        )
        eta_seconds = math.ceil(eta_batches * avg_seconds)
        print(
            f"Batch done. Progress: {completed}/{len(batches)} | "
            f"ETA ~ {eta_seconds // 60}m {eta_seconds % 60}s"
        )

    print("All batches completed.")


if __name__ == "__main__":
    main()
