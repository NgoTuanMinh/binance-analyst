"""Validate merged results per batch and optionally auto-recover low-quality symbols."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.downloader import BinanceDataDownloader
from src.extractor import DataExtractor
from src.merger import DataMerger
from src.utils import load_config
from src.validator import DataValidator


def _load_symbols(symbols_file: Path) -> list[str]:
    data = json.loads(symbols_file.read_text())
    if not isinstance(data, list):
        raise ValueError("symbols file must be list")
    return [str(s) for s in data]


def _group_by_symbol(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["symbol"], []).append(rec)
    return grouped


def main() -> None:
    """CLI entrypoint for validating and repairing batch outputs."""
    parser = argparse.ArgumentParser(description="Validate batch and auto-redownload weak symbols")
    parser.add_argument("--symbols-file", default=str(PROJECT_ROOT / settings.SYMBOLS_FILE))
    parser.add_argument("--intervals", nargs="+", default=settings.INTERVALS)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--quality-threshold", type=int, default=90)
    parser.add_argument("--auto-redownload", action="store_true")
    parser.add_argument("--output", default=str(PROJECT_ROOT / settings.LOG_DIR / "batch_validation_report.json"))
    args = parser.parse_args()

    conf = load_config()
    start_date = args.start or conf["data_start_date"]
    end_date = args.end or conf["data_end_date"]
    symbols = _load_symbols(Path(args.symbols_file))

    validator = DataValidator(str(PROJECT_ROOT / settings.MERGE_DIR))
    validation_df = validator.validate_all(symbols, args.intervals)
    records = validation_df.to_dict(orient="records")

    low_quality = [
        r
        for r in records
        if r.get("exists") and int(r.get("quality_score", 0)) < int(args.quality_threshold)
    ]

    repaired: list[dict[str, Any]] = []
    if args.auto_redownload and low_quality:
        downloader = BinanceDataDownloader(
            {
                "base_url": settings.BASE_URL,
                "download_dir": str(PROJECT_ROOT / settings.DOWNLOAD_DIR),
                "max_retries": settings.MAX_RETRIES,
                "rate_limit_sleep": settings.RATE_LIMIT_SLEEP,
                "max_workers": settings.MAX_WORKERS,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        extractor = DataExtractor(
            str(PROJECT_ROOT / settings.DOWNLOAD_DIR),
            str(PROJECT_ROOT / settings.EXTRACT_DIR),
        )
        merger = DataMerger(str(PROJECT_ROOT / settings.EXTRACT_DIR), str(PROJECT_ROOT / settings.MERGE_DIR))

        grouped = _group_by_symbol(low_quality)
        for symbol, items in grouped.items():
            intervals = sorted(set(i["interval"] for i in items))
            dl_stats = downloader.download_batch([symbol], intervals, start_date, end_date)
            ex_stats = extractor.extract_all()
            mg_stats = merger.merge_all([symbol], intervals)
            for interval in intervals:
                new_result = validator.validate_symbol(symbol, interval)
                repaired.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "download": dl_stats,
                        "extract": ex_stats,
                        "merge": mg_stats,
                        "post_repair_validation": new_result,
                    }
                )

    report = {
        "summary": {
            "total_rows": len(records),
            "low_quality_rows": len(low_quality),
            "threshold": args.quality_threshold,
            "auto_redownload": args.auto_redownload,
        },
        "low_quality": low_quality,
        "repaired": repaired,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
