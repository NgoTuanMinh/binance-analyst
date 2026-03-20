"""Downloader module for Binance.vision daily kline ZIP files."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import aiohttp
import requests
from loguru import logger
from tqdm import tqdm

from src.utils import calculate_checksum, ensure_dir, parse_date


@dataclass
class DownloadResult:
    """Result object for single file download."""

    url: str
    path: str
    success: bool
    status_code: int | None = None
    error: str | None = None


class BinanceDataDownloader:
    """Smart downloader with retry, resume, checksum, and batching."""

    def __init__(self, config: dict[str, Any]):
        """Initialize downloader with runtime config."""
        self.config = config
        self.base_url = config["base_url"].rstrip("/")
        self.download_dir = ensure_dir(config["download_dir"])
        self.max_retries = int(config.get("max_retries", 3))
        self.rate_limit_sleep = float(config.get("rate_limit_sleep", 2))
        self.max_workers = int(config.get("max_workers", 5))
        self.quiet = bool(config.get("quiet", False))

    def construct_url(self, symbol: str, interval: str, date: str) -> str:
        """Construct daily kline ZIP URL for spot market."""
        return (
            f"{self.base_url}/data/spot/daily/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
        )

    def construct_checksum_url(self, symbol: str, interval: str, date: str) -> str:
        """Construct checksum file URL for a kline ZIP file."""
        return f"{self.construct_url(symbol, interval, date)}.CHECKSUM"

    def generate_date_range(self, start_date: str, end_date: str) -> list[str]:
        """Generate inclusive date list (YYYY-MM-DD)."""
        start = parse_date(start_date)
        end = parse_date(end_date)
        if end < start:
            raise ValueError("end_date must be >= start_date")
        dates = []
        cursor = start
        while cursor <= end:
            dates.append(cursor.strftime("%Y-%m-%d"))
            cursor += timedelta(days=1)
        return dates

    def _is_existing_file_valid(self, path: Path) -> bool:
        """Treat non-empty existing file as resumable success candidate."""
        return path.exists() and path.stat().st_size > 0

    def download_single_file(self, url: str, dest_path: str) -> bool:
        """Download one file with retry and exponential backoff."""
        dest = Path(dest_path)
        ensure_dir(dest.parent)
        if self._is_existing_file_valid(dest):
            logger.info("Skip existing file: {}", dest)
            return True

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, timeout=60)
                status = response.status_code
                if status == 200:
                    dest.write_bytes(response.content)
                    if dest.stat().st_size == 0:
                        raise ValueError("Downloaded empty file")
                    return True
                if status == 404:
                    logger.warning("404 not found: {}", url)
                    return False
                if status == 429:
                    sleep_for = self.rate_limit_sleep * (2**attempt)
                    logger.warning("429 rate limited. Sleeping {:.1f}s", sleep_for)
                    time.sleep(sleep_for)
                    continue
                if status >= 500:
                    sleep_for = self.rate_limit_sleep * (2**attempt)
                    logger.warning("Server error {}. Retry in {:.1f}s", status, sleep_for)
                    time.sleep(sleep_for)
                    continue
                logger.error("Unhandled status {} for {}", status, url)
                return False
            except Exception as exc:
                if attempt >= self.max_retries:
                    logger.exception("Download failed after retries: {}", url)
                    return False
                sleep_for = self.rate_limit_sleep * (2**attempt)
                logger.warning("Retry {} for {} in {:.1f}s ({})", attempt + 1, url, sleep_for, exc)
                time.sleep(sleep_for)
        return False

    def check_checksum(self, file_path: str, checksum_path: str) -> bool:
        """Verify file SHA256 against .CHECKSUM file content."""
        fpath = Path(file_path)
        cpath = Path(checksum_path)
        if not fpath.exists():
            return False
        if not cpath.exists():
            logger.warning("Checksum file missing: {}", cpath)
            return True
        checksum_content = cpath.read_text().strip().split()
        if not checksum_content:
            return True
        expected = checksum_content[0].lower()
        actual = calculate_checksum(fpath).lower()
        return expected == actual

    def get_missing_dates(self, symbol: str, interval: str) -> list[str]:
        """Return dates already missing inside configured start/end window."""
        dates = self.generate_date_range(
            self.config["start_date"], self.config["end_date"]
        )
        missing: list[str] = []
        symbol_dir = self.download_dir / symbol / interval
        for d in dates:
            file_path = symbol_dir / f"{symbol}-{interval}-{d}.zip"
            if not self._is_existing_file_valid(file_path):
                missing.append(d)
        return missing

    async def _download_one_async(
        self, session: aiohttp.ClientSession, url: str, dest: Path
    ) -> DownloadResult:
        """Async download with retry support."""
        if self._is_existing_file_valid(dest):
            return DownloadResult(url, str(dest), True, status_code=200)
        ensure_dir(dest.parent)

        for attempt in range(self.max_retries + 1):
            try:
                async with session.get(url, timeout=90) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        dest.write_bytes(content)
                        return DownloadResult(url, str(dest), True, status_code=200)
                    if resp.status == 404:
                        return DownloadResult(url, str(dest), False, status_code=404)
                    if resp.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(self.rate_limit_sleep * (2**attempt))
                        continue
                    return DownloadResult(url, str(dest), False, status_code=resp.status)
            except Exception as exc:
                if attempt >= self.max_retries:
                    return DownloadResult(url, str(dest), False, error=str(exc))
                await asyncio.sleep(self.rate_limit_sleep * (2**attempt))
        return DownloadResult(url, str(dest), False, error="retry exhausted")

    async def _run_batch_async(self, jobs: list[tuple[str, Path]]) -> list[DownloadResult]:
        """Run bounded-concurrency async batch downloads."""
        connector = aiohttp.TCPConnector(limit=self.max_workers)
        semaphore = asyncio.Semaphore(self.max_workers)
        results: list[DownloadResult] = []

        async with aiohttp.ClientSession(connector=connector) as session:
            async def run_job(url: str, dest: Path) -> DownloadResult:
                async with semaphore:
                    return await self._download_one_async(session, url, dest)

            coros = [run_job(url, dest) for url, dest in jobs]
            iterable = asyncio.as_completed(coros)
            if self.quiet:
                for done in iterable:
                    results.append(await done)
            else:
                for done in tqdm(iterable, total=len(coros), desc="Downloading"):
                    results.append(await done)
        return results

    def download_batch(
        self, symbols: list[str], intervals: list[str], start_date: str, end_date: str
    ) -> dict[str, Any]:
        """Download multiple files in batch and return stats."""
        jobs: list[tuple[str, Path]] = []
        dates = self.generate_date_range(start_date, end_date)
        for symbol in symbols:
            for interval in intervals:
                for d in dates:
                    url = self.construct_url(symbol, interval, d)
                    dest = self.download_dir / symbol / interval / f"{symbol}-{interval}-{d}.zip"
                    jobs.append((url, dest))

        results = asyncio.run(self._run_batch_async(jobs))
        success = sum(1 for r in results if r.success)
        failed = len(results) - success
        return {"total": len(results), "success": success, "failed": failed}
