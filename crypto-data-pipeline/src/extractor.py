"""Extractor module for decompressing Binance ZIP files."""

from __future__ import annotations

import gzip
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.utils import ensure_dir

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


class DataExtractor:
    """Extract ZIP archives into processed CSV files."""

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = ensure_dir(processed_dir)

    def extract_zip(self, zip_path: Path) -> list[Path]:
        """Extract files from a ZIP and return extracted paths."""
        extracted: list[Path] = []
        rel_dir = zip_path.parent.relative_to(self.raw_dir)
        out_dir = ensure_dir(self.processed_dir / rel_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                target = out_dir / member
                zf.extract(member, out_dir)
                extracted.append(target)
        return extracted

    def normalize_file(self, path: Path) -> Path:
        """Convert .csv.gz to .csv if needed."""
        if path.suffixes[-2:] == [".csv", ".gz"]:
            out_path = path.with_suffix("").with_suffix(".csv")
            with gzip.open(path, "rb") as fin, open(out_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            path.unlink(missing_ok=True)
            return out_path
        return path

    def validate_csv(self, csv_path: Path) -> bool:
        """Basic CSV schema/quality checks."""
        try:
            df = pd.read_csv(csv_path, header=None)
            if df.empty or df.shape[1] < 6:
                return False
            if df.shape[1] == len(KLINE_COLUMNS):
                df.columns = KLINE_COLUMNS
                _ = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
            return True
        except Exception:
            return False

    def extract_all(self) -> dict[str, Any]:
        """Extract all ZIPs found under raw directory."""
        zip_files = list(self.raw_dir.rglob("*.zip"))
        success = 0
        failed = 0
        for z in zip_files:
            try:
                extracted = self.extract_zip(z)
                for f in extracted:
                    norm = self.normalize_file(f)
                    if norm.suffix == ".csv":
                        if self.validate_csv(norm):
                            success += 1
                        else:
                            failed += 1
                            logger.warning("Invalid CSV format: {}", norm)
            except Exception as exc:
                failed += 1
                logger.exception("Failed extracting {}: {}", z, exc)
        return {"zip_files": len(zip_files), "success": success, "failed": failed}
