"""Merge processed daily CSV files into per-symbol interval datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.utils import ensure_dir, get_interval_minutes, normalize_open_time_ms_dataframe, progress_bar

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


class DataMerger:
    """Merge symbol+interval data and produce parquet/csv outputs."""

    def __init__(self, processed_dir: str, merged_dir: str, volume_ma_window: int = 20):
        self.processed_dir = Path(processed_dir)
        self.merged_dir = ensure_dir(merged_dir)
        self.volume_ma_window = volume_ma_window

    def _load_symbol_interval_files(self, symbol: str, interval: str) -> list[Path]:
        target_dir = self.processed_dir / symbol / interval
        return sorted(target_dir.glob("*.csv"))

    def _read_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            header=None,
            names=KLINE_COLUMNS,
            dtype={
                "open_time": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            },
        )
        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["daily_return"] = out["close"].pct_change()
        out["price_range"] = (out["high"] - out["low"]) / out["open"]
        out["volume_ma"] = out["volume"].rolling(self.volume_ma_window, min_periods=1).mean()
        return out

    def _find_gaps(self, df: pd.DataFrame, interval: str) -> list[dict[str, str]]:
        if df.empty:
            return []
        minutes = get_interval_minutes(interval)
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True).sort_values()
        diffs = ts.diff().dropna()
        expected = pd.Timedelta(minutes=minutes)
        gaps = diffs[diffs > expected]
        return [
            {"at": str(ts.iloc[idx]), "gap_minutes": str(g / pd.Timedelta(minutes=1))}
            for idx, g in zip(gaps.index, gaps, strict=False)
        ]

    def merge_symbol_interval(self, symbol: str, interval: str) -> pd.DataFrame:
        """Merge all CSV files of one symbol/interval into one DataFrame."""
        files = self._load_symbol_interval_files(symbol, interval)
        if not files:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        chunks = [self._read_csv(f) for f in files]
        merged = pd.concat(chunks, ignore_index=True)
        merged = normalize_open_time_ms_dataframe(merged)
        merged = merged.sort_values("open_time")
        merged = merged.drop_duplicates(subset=["open_time"], keep="last")
        merged["date"] = pd.to_datetime(merged["open_time"], unit="ms", utc=True).dt.date
        merged = self._add_features(merged)
        return merged

    def update_existing_merge(
        self, symbol: str, interval: str, new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Append new data to existing merged file while removing duplicates."""
        out_dir = ensure_dir(self.merged_dir / symbol)
        parquet_path = out_dir / f"{symbol}-{interval}.parquet"
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data.copy()
        combined = combined.sort_values("open_time").drop_duplicates(
            subset=["open_time"], keep="last"
        )
        combined.to_parquet(parquet_path, index=False)
        return combined

    def _write_outputs(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        out_dir = ensure_dir(self.merged_dir / symbol)
        parquet_path = out_dir / f"{symbol}-{interval}.parquet"
        csv_path = out_dir / f"{symbol}-{interval}.csv"
        meta_path = out_dir / f"{symbol}-{interval}.metadata.json"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        metadata = {
            "symbol": symbol,
            "interval": interval,
            "row_count": int(len(df)),
            "start_time": int(df["open_time"].min()) if not df.empty else None,
            "end_time": int(df["open_time"].max()) if not df.empty else None,
            "columns": list(df.columns),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def generate_merge_report(self, symbol: str, interval: str) -> dict[str, Any]:
        """Generate merge quality report for one symbol/interval."""
        parquet_path = self.merged_dir / symbol / f"{symbol}-{interval}.parquet"
        if not parquet_path.exists():
            return {"symbol": symbol, "interval": interval, "exists": False}
        df = pd.read_parquet(parquet_path)
        df = normalize_open_time_ms_dataframe(df)
        gaps = self._find_gaps(df, interval)
        quality_score = max(0, 100 - (len(gaps) * 5))
        return {
            "symbol": symbol,
            "interval": interval,
            "exists": True,
            "rows": len(df),
            "first_time": int(df["open_time"].min()) if not df.empty else None,
            "last_time": int(df["open_time"].max()) if not df.empty else None,
            "gaps": gaps,
            "quality_score": quality_score,
        }

    def merge_all(self, symbols: list[str], intervals: list[str]) -> dict[str, Any]:
        """Merge all requested symbols/intervals and return summary."""
        success = 0
        failed = 0
        reports: list[dict[str, Any]] = []
        jobs = [(symbol, interval) for symbol in symbols for interval in intervals]
        for symbol, interval in progress_bar(jobs, desc="Merging datasets", total=len(jobs)):
            try:
                df = self.merge_symbol_interval(symbol, interval)
                self._write_outputs(symbol, interval, df)
                reports.append(self.generate_merge_report(symbol, interval))
                success += 1
            except Exception as exc:
                failed += 1
                logger.exception("Merge failed for {} {}: {}", symbol, interval, exc)
        return {"success": success, "failed": failed, "reports": reports}
