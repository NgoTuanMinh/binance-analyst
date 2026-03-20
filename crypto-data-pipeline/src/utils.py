"""Utility helpers shared across the crypto data pipeline."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Set up Python logger and also configure loguru sink."""
    py_logger = logging.getLogger(name)
    py_logger.setLevel(level)

    if not py_logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        py_logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        py_logger.addHandler(sh)

    logger.remove()
    logger.add(log_file, level="INFO", rotation="10 MB")
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    return py_logger


def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for measuring function execution time."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("Function {} executed in {:.3f}s", func.__name__, elapsed)
        return result

    return wrapper


def ensure_dir(path: str | Path) -> Path:
    """Create directory if not exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_file_write(path: str | Path, content: str | bytes, mode: str = "w") -> None:
    """Atomically write file content by writing temp file then replacing."""
    target = Path(path)
    ensure_dir(target.parent)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(target.parent))
    os.close(tmp_fd)
    write_mode = mode
    with open(tmp_path, write_mode) as f:
        f.write(content)  # type: ignore[arg-type]
    os.replace(tmp_path, target)


def get_file_size_str(size_bytes: int) -> str:
    """Convert bytes to human readable KB/MB/GB."""
    value = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def calculate_checksum(file_path: str | Path) -> str:
    """Calculate SHA256 checksum for a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_date(date_str: str) -> datetime:
    """Parse date from multiple accepted formats."""
    formats = ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S")
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {date_str}")


def get_interval_minutes(interval: str) -> int:
    """Convert interval string to minutes."""
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60
    raise ValueError(f"Unsupported interval: {interval}")


def generate_date_ranges(
    start: str, end: str, chunk_days: int = 30
) -> list[tuple[str, str]]:
    """Split [start, end] range into chunks of chunk_days."""
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    if end_dt < start_dt:
        raise ValueError("end must be >= start")
    ranges: list[tuple[str, str]] = []
    cursor = start_dt
    while cursor <= end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_dt)
        ranges.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + timedelta(days=1)
    return ranges


def resample_ohlcv(
    df: pd.DataFrame, from_interval: str, to_interval: str
) -> pd.DataFrame:
    """Resample OHLCV data from lower interval to higher interval."""
    _ = from_interval
    if "open_time" not in df.columns:
        raise ValueError("DataFrame missing open_time")
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out = out.set_index("open_time")
    rule = f"{get_interval_minutes(to_interval)}min"
    rs = out.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    rs = rs.dropna().reset_index()
    return rs


def add_technical_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Add simple technical indicators to a dataframe."""
    out = df.copy()
    if "close" in out.columns:
        for ind in indicators:
            if ind.startswith("sma_"):
                period = int(ind.split("_")[1])
                out[ind] = out["close"].rolling(period).mean()
            elif ind.startswith("ema_"):
                period = int(ind.split("_")[1])
                out[ind] = out["close"].ewm(span=period, adjust=False).mean()
    return out


def detect_outliers(df: pd.DataFrame, column: str, method: str = "iqr") -> pd.DataFrame:
    """Return subset of outlier rows based on selected method."""
    if column not in df.columns:
        raise ValueError(f"Missing column {column}")
    if method != "iqr":
        raise ValueError("Only iqr is supported currently")
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] < lower) | (df[column] > upper)]


async def run_parallel(tasks: list[Awaitable[Any]], max_workers: int) -> list[Any]:
    """Run async tasks in parallel with a semaphore worker limit."""
    semaphore = asyncio.Semaphore(max_workers)

    async def _wrap(coro: Awaitable[Any]) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[_wrap(t) for t in tasks], return_exceptions=True)


def async_retry(
    func: Callable[..., Awaitable[Any]], retries: int = 3, backoff: int = 2
) -> Callable[..., Awaitable[Any]]:
    """Decorator for async function retries with exponential backoff."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception:
                if attempt >= retries:
                    raise
                sleep_for = backoff ** (attempt + 1)
                await asyncio.sleep(sleep_for)
        raise RuntimeError("Unreachable retry state")

    return wrapper


def load_config() -> dict[str, Any]:
    """Load runtime config from env and defaults."""
    load_dotenv()
    config = {
        "project_name": os.getenv("PROJECT_NAME", "CryptoDataPipeline"),
        "data_start_date": os.getenv("DATA_START_DATE", "2020-01-01"),
        "data_end_date": os.getenv("DATA_END_DATE", "2025-12-31"),
        "enable_logging": os.getenv("ENABLE_LOGGING", "True").lower() == "true",
    }
    return config


def validate_config(config: dict[str, Any]) -> bool:
    """Validate minimal required config fields."""
    required = ["project_name", "data_start_date", "data_end_date", "enable_logging"]
    for key in required:
        if key not in config:
            return False
    try:
        parse_date(str(config["data_start_date"]))
        parse_date(str(config["data_end_date"]))
    except ValueError:
        return False
    return True


def memory_usage() -> float:
    """Return current process memory in MB."""
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


def progress_bar(iterable: Iterable[Any], desc: str, total: int | None = None) -> tqdm:
    """Wrapper around tqdm for consistent progress display."""
    return tqdm(iterable, desc=desc, total=total)


def save_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    """Save JSON atomically."""
    safe_file_write(Path(path), json.dumps(payload, indent=2), mode="w")
