"""Fetch Binance spot klines and persist OHLCV to disk (Parquet)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BINANCE_SPOT = "https://api.binance.com"


def klines_response_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
            ]
        )
    rows = []
    for k in raw:
        rows.append(
            {
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
            }
        )
    return pd.DataFrame(rows)


def fetch_klines(
    symbol: str,
    interval: str,
    *,
    limit: int = 1000,
    start_time: int | None = None,
    end_time: int | None = None,
    session: requests.Session | None = None,
    base_url: str = BINANCE_SPOT,
) -> pd.DataFrame:
    """GET /api/v3/klines; returns oldest-first rows as DataFrame."""
    params: dict[str, int | str] = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)
    sess = session or requests.Session()
    r = sess.get(f"{base_url.rstrip('/')}/api/v3/klines", params=params, timeout=45)
    r.raise_for_status()
    return klines_response_to_df(r.json())


def refresh_series(
    cache_path: Path,
    symbol: str,
    interval: str,
    *,
    min_rows: int,
    max_rows: int,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Merge disk cache with newer Binance data; backfill until ``min_rows`` or API exhaustion.
    Trims to ``max_rows`` most recent bars.
    """
    cache_path = Path(cache_path)
    df = pd.read_parquet(cache_path) if cache_path.is_file() else pd.DataFrame()

    start = None
    if not df.empty:
        start = int(df["open_time"].max()) + 1

    newest = fetch_klines(symbol, interval, limit=1000, start_time=start, session=session)
    if not newest.empty:
        df = pd.concat([df, newest], ignore_index=True)
        df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)

    while len(df) < min_rows:
        end_before = int(df["open_time"].min()) - 1 if not df.empty else None
        chunk = fetch_klines(symbol, interval, limit=1000, end_time=end_before, session=session)
        if chunk.empty:
            break
        df = pd.concat([chunk, df], ignore_index=True)
        df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)
        if len(chunk) < 1000:
            break

    if len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def load_mtf_cached(
    cache_dir: Path,
    symbol: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Refresh 15m / 1h / 4h series with enough history for EMA200 (4h), H1 swing (64), M15 patterns.

    Keys: ``15m``, ``1h``, ``4h`` (canonical, matches backtest alignment).
    """
    cache_dir = Path(cache_dir)
    specs = {
        "15m": (4000, 6000),
        "1h": (1500, 2500),
        "4h": (400, 600),
    }
    out: dict[str, pd.DataFrame] = {}
    for interval, (mn, mx) in specs.items():
        path = cache_dir / f"{symbol.upper()}_{interval}.parquet"
        out[interval] = refresh_series(path, symbol, interval, min_rows=mn, max_rows=mx, session=session)
    return out
