"""Main orchestration script for crypto data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from src.downloader import BinanceDataDownloader
from src.extractor import DataExtractor
from src.merger import DataMerger
from src.utils import ensure_dir, load_config, save_json, setup_logger, validate_config
from src.validator import DataValidator


def load_symbols(symbols_arg: str) -> list[str]:
    """Load symbols from a JSON file path or comma-separated list."""
    path = Path(symbols_arg)
    if path.exists():
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [str(x) for x in data]
    return [s.strip() for s in symbols_arg.split(",") if s.strip()]


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint JSON if exists."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    """Persist checkpoint data."""
    save_json(path, data)


def build_parser() -> argparse.ArgumentParser:
    """Build command line parser."""
    parser = argparse.ArgumentParser(description="Crypto data pipeline")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=settings.SYMBOLS_FILE)
    parser.add_argument("--intervals", nargs="+", default=settings.INTERVALS)
    parser.add_argument("--update-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--merge-then-report",
        action="store_true",
        help="Run Step 3 (merge) then Step 4 and Step 5 sequentially.",
    )
    parser.add_argument("--format", type=str, default="html")
    return parser


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute pipeline based on provided args."""
    if args.merge_then_report and (args.validate or args.report):
        raise ValueError("--merge-then-report cannot be combined with --validate/--report.")

    conf = load_config()
    if not validate_config(conf):
        raise ValueError("Invalid configuration from environment.")

    project_root = settings.PROJECT_ROOT
    setup_logger(
        "crypto-data-pipeline",
        str(project_root / settings.LOG_DIR / "pipeline.log"),
    )
    checkpoint_file = project_root / settings.CHECKPOINT_FILE
    ensure_dir(checkpoint_file.parent)
    checkpoint = load_checkpoint(checkpoint_file)

    symbols = load_symbols(str(project_root / args.symbols) if args.symbols.endswith(".json") else args.symbols)
    intervals = args.intervals
    start_date = args.start or conf["data_start_date"]
    end_date = args.end or conf["data_end_date"]

    downloader_config = {
        "base_url": settings.BASE_URL,
        "download_dir": str(project_root / settings.DOWNLOAD_DIR),
        "max_retries": settings.MAX_RETRIES,
        "rate_limit_sleep": settings.RATE_LIMIT_SLEEP,
        "max_workers": settings.MAX_WORKERS,
        "start_date": start_date,
        "end_date": end_date,
    }
    downloader = BinanceDataDownloader(downloader_config)
    extractor = DataExtractor(
        str(project_root / settings.DOWNLOAD_DIR),
        str(project_root / settings.EXTRACT_DIR),
    )
    merger = DataMerger(
        str(project_root / settings.EXTRACT_DIR),
        str(project_root / settings.MERGE_DIR),
    )
    validator = DataValidator(str(project_root / settings.MERGE_DIR))

    results: dict[str, Any] = {}
    try:
        if args.merge_then_report:
            logger.info("Step 3: Merge")
            results["merge"] = merger.merge_all(symbols, intervals)
            checkpoint["merge_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)

            logger.info("Step 4: Validate merged dataset")
            validation_df = validator.validate_all(symbols, intervals)
            results["validation_rows"] = int(len(validation_df))
            checkpoint["validate_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)

            logger.info("Step 5: Generate reports")
            reports = validator.export_reports(validation_df)
            results["reports"] = reports
            checkpoint["report_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)
        elif not args.validate and not args.report:
            logger.info("Step 1: Download data")
            results["download"] = downloader.download_batch(
                symbols, intervals, start_date, end_date
            )
            checkpoint["download_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)

            logger.info("Step 2: Extract and validate files")
            results["extract"] = extractor.extract_all()
            checkpoint["extract_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)

            logger.info("Step 3: Merge")
            results["merge"] = merger.merge_all(symbols, intervals)
            checkpoint["merge_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)
 
            logger.info("Step 4: Validate merged dataset")
            validation_df = validator.validate_all(symbols, intervals)
            results["validation_rows"] = int(len(validation_df))
            checkpoint["validate_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)

            logger.info("Step 5: Generate reports")
            reports = validator.export_reports(validation_df)
            results["reports"] = reports
            checkpoint["report_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)
        elif args.validate and not args.report:
            logger.info("Step 4: Validate merged dataset")
            validation_df = validator.validate_all(symbols, intervals)
            results["validation_rows"] = int(len(validation_df))
            checkpoint["validate_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)
        else:
            logger.info("Step 4: Validate merged dataset")
            validation_df = validator.validate_all(symbols, intervals)
            results["validation_rows"] = int(len(validation_df))
            checkpoint["validate_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)

            logger.info("Step 5: Generate reports")
            reports = validator.export_reports(validation_df)
            results["reports"] = reports
            checkpoint["report_done"] = True
            save_checkpoint(checkpoint_file, checkpoint)
    except Exception as exc:
        logger.exception("Pipeline failed: {}", exc)
        error_path = project_root / settings.LOG_DIR / "pipeline_errors.log"
        ensure_dir(error_path.parent)
        with open(error_path, "a") as f:
            f.write(f"{exc}\n")
        raise

    return results


def main() -> None:
    """Program entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    results = run_pipeline(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
