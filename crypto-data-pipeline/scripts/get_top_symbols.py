"""Fetch top USDT symbols by quote volume from Binance public endpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
TICKER_24H_URL = "https://api.binance.com/api/v3/ticker/24hr"
EXCLUDED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT")


def _load_cache(cache_file: Path, ttl_seconds: int) -> list[str] | None:
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text())
        if time.time() - payload["created_at"] > ttl_seconds:
            return None
        return payload["symbols"]
    except Exception:
        return None


def _save_cache(cache_file: Path, symbols: list[str]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": time.time(), "symbols": symbols}
    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_top_symbols(top_n: int = 300) -> list[str]:
    """Fetch and rank top spot USDT symbols from Binance."""
    exchange_info = requests.get(EXCHANGE_INFO_URL, timeout=20).json()
    all_symbols = exchange_info.get("symbols", [])
    usdt_spot = {
        s["symbol"]
        for s in all_symbols
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed", True)
        and not s.get("symbol", "").endswith(EXCLUDED_SUFFIXES)
    }

    ticker_24h = requests.get(TICKER_24H_URL, timeout=20).json()
    volumes = []
    for t in ticker_24h:
        symbol = t.get("symbol")
        if symbol in usdt_spot:
            try:
                quote_volume = float(t.get("quoteVolume", 0))
            except (TypeError, ValueError):
                quote_volume = 0.0
            volumes.append((symbol, quote_volume))

    ranked = sorted(volumes, key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in ranked[:top_n]]


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Fetch top Binance USDT symbols")
    parser.add_argument("--top", type=int, default=300, help="Top N symbols")
    parser.add_argument(
        "--output",
        type=str,
        default="config/top_300_symbols.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default="config/top_300_symbols.txt",
        help="Output TXT file path",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="data/cache/top_symbols_cache.json",
        help="Cache file path",
    )
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL seconds")
    args = parser.parse_args()

    cache_path = Path(args.cache_file)
    symbols = _load_cache(cache_path, args.cache_ttl)
    from_cache = symbols is not None
    if symbols is None:
        symbols = fetch_top_symbols(args.top)
        _save_cache(cache_path, symbols)

    output_json = Path(args.output)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(symbols, indent=2), encoding="utf-8")
    output_txt.write_text("\n".join(symbols), encoding="utf-8")

    exchange_info = requests.get(EXCHANGE_INFO_URL, timeout=20).json()
    total_usdt = len(
        [
            s
            for s in exchange_info.get("symbols", [])
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        ]
    )
    print(f"Total USDT pairs found: {total_usdt}")
    print(f"Pairs after filtering: {len(symbols)}")
    print(f"Data source: {'cache' if from_cache else 'api'}")
    print("Top 10 pairs:")
    for idx, sym in enumerate(symbols[:10], start=1):
        print(f"{idx:2d}. {sym}")


if __name__ == "__main__":
    main()
