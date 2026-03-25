"""Load merged OHLCV Parquet data for multi-timeframe backtests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

# Common labels -> pipeline interval strings (Binance / merger naming).
TIMEFRAME_ALIASES: dict[str, str] = {
    "M15": "15m",
    "15M": "15m",
    "m15": "15m",
    "H1": "1h",
    "1H": "1h",
    "h1": "1h",
    "H4": "4h",
    "4H": "4h",
    "h4": "4h",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
}

_DEFAULT_MERGED = Path(__file__).resolve().parent.parent / "crypto-data-pipeline" / "data" / "merged"


def _interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60
    raise ValueError(f"Unsupported interval: {interval}")


def canonical_interval(label: str) -> str:
    """Map M15/H1/H4 (or variants) to merger interval keys: 15m, 1h, 4h."""
    key = label.strip()
    if key in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[key]
    lower = key.lower()
    if lower in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[lower]
    raise ValueError(
        f"Unknown timeframe label: {label!r}; known: {sorted(set(TIMEFRAME_ALIASES.keys()))}"
    )


def _normalize_open_time_ms(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    x = series.astype("int64").to_numpy()
    ms = np.select(
        [
            x >= 10**17,
            (x >= 10**14) & (x < 10**17),
            x < 10**11,
        ],
        [
            x // 1_000_000,
            x // 1_000,
            x * 1_000,
        ],
        default=x,
    )
    return pd.Series(ms, index=series.index, dtype="int64")


def _prepare_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "open_time" not in out.columns:
        raise ValueError("DataFrame must contain open_time")
    out["open_time"] = _normalize_open_time_ms(out["open_time"])
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    return out.reset_index(drop=True)


def _parquet_path(merged_dir: Path, symbol: str, interval: str) -> Path:
    return merged_dir / symbol / f"{symbol}-{interval}.parquet"


def _file_signature(path: Path) -> tuple[float, int]:
    st = path.stat()
    return (st.st_mtime, st.st_size)


def _signatures_for_symbol(
    merged_dir: Path, symbol: str, intervals: Sequence[str]
) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for iv in intervals:
        p = _parquet_path(merged_dir, symbol, iv)
        if not p.is_file():
            raise FileNotFoundError(f"Missing Parquet for {symbol} {iv}: {p}")
        out[iv] = _file_signature(p)
    return out


class BacktestDataLoader:
    """Read per-symbol Parquet series, optional alignment across timeframes, with caching."""

    def __init__(
        self,
        merged_dir: str | Path | None = None,
        *,
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.merged_dir = Path(merged_dir) if merged_dir is not None else _DEFAULT_MERGED
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._raw_cache: MutableMapping[str, pd.DataFrame] = {}
        self._aligned_cache: MutableMapping[str, pd.DataFrame] = {}

    def clear_cache(self) -> None:
        """Drop in-memory caches (does not delete on-disk cache files)."""
        self._raw_cache.clear()
        self._aligned_cache.clear()

    def _raw_cache_key(self, symbol: str, interval: str, sig: tuple[float, int]) -> str:
        return f"raw:{symbol}:{interval}:{sig[0]}:{sig[1]}"

    def _aligned_cache_key(
        self,
        symbol: str,
        base: str,
        intervals: tuple[str, ...],
        sigs: Mapping[str, tuple[float, int]],
    ) -> str:
        payload = "|".join(f"{iv}:{sigs[iv][0]}:{sigs[iv][1]}" for iv in intervals)
        h = hashlib.sha256(f"{symbol}|{base}|{payload}".encode()).hexdigest()[:24]
        return f"aligned:{symbol}:{base}:{h}"

    def _disk_cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        assert self.cache_dir is not None
        return (self.cache_dir / f"{cache_key}.parquet", self.cache_dir / f"{cache_key}.meta.json")

    def load_parquet(self, symbol: str, interval_label: str) -> pd.DataFrame:
        """Load one timeframe from `{merged_dir}/{symbol}/{symbol}-{interval}.parquet`."""
        interval = canonical_interval(interval_label)
        path = _parquet_path(self.merged_dir, symbol, interval)
        if not path.is_file():
            raise FileNotFoundError(f"No Parquet at {path}")

        sig = _file_signature(path)
        ck = self._raw_cache_key(symbol, interval, sig)
        if self.use_cache and ck in self._raw_cache:
            return self._raw_cache[ck].copy()

        df = pd.read_parquet(path)
        df = _prepare_ohlcv_df(df)

        if self.use_cache:
            self._raw_cache[ck] = df.copy()
        return df

    def load_multi_timeframe(
        self,
        symbol: str,
        timeframes: Sequence[str] | None = None,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load several intervals (e.g. M15, H1, H4). Returns dict keyed by canonical interval.
        Applies optional open_time bounds on each series.
        """
        labels = list(timeframes) if timeframes is not None else ["M15", "H1", "H4"]
        seen: set[str] = set()
        canonicals: list[str] = []
        for lab in labels:
            iv = canonical_interval(lab)
            if iv not in seen:
                seen.add(iv)
                canonicals.append(iv)

        out: dict[str, pd.DataFrame] = {}
        for iv in canonicals:
            df = self.load_parquet(symbol, iv)
            if start_ms is not None:
                df = df[df["open_time"] >= start_ms]
            if end_ms is not None:
                df = df[df["open_time"] <= end_ms]
            out[iv] = df.reset_index(drop=True)
        return out

    def align_timeframes(
        self,
        frames: Mapping[str, pd.DataFrame],
        base_interval: str | None = None,
    ) -> pd.DataFrame:
        """
        Align OHLCV frames on the base timeframe's `open_time` using backward as-of joins:
        each base row gets the latest higher-TF bar with open_time <= base open_time.
        Non-open_time columns are prefixed with `{interval}_` (e.g. `1h_close`).
        """
        if not frames:
            raise ValueError("frames must be non-empty")

        by_iv: dict[str, pd.DataFrame] = {}
        for k, v in frames.items():
            iv = canonical_interval(k)
            by_iv[iv] = v

        if base_interval is None:
            base_iv = min(by_iv.keys(), key=_interval_to_minutes)
        else:
            base_iv = canonical_interval(base_interval)
            if base_iv not in by_iv:
                raise KeyError(f"base_interval {base_iv!r} not in frames (have {sorted(by_iv)})")

        base_df = by_iv[base_iv].sort_values("open_time").copy()
        rename_base = {c: f"{base_iv}_{c}" for c in base_df.columns if c != "open_time"}
        left = base_df.rename(columns=rename_base)

        for iv in sorted(by_iv.keys(), key=_interval_to_minutes):
            if iv == base_iv:
                continue
            right = by_iv[iv].sort_values("open_time").copy()
            r_time = f"_asof_open_{iv}"
            rename_r = {
                c: (f"{iv}_{c}" if c != "open_time" else r_time) for c in right.columns
            }
            right = right.rename(columns=rename_r)
            
            # Ensure merge keys are numeric to prevent MergeError
            left['open_time'] = pd.to_numeric(left['open_time'], errors='coerce')
            right[r_time] = pd.to_numeric(right[r_time], errors='coerce')
            
            left = pd.merge_asof(
                left,
                right,
                left_on="open_time",
                right_on=r_time,
                direction="backward",
            )
            left = left.drop(columns=[r_time])

        return left.reset_index(drop=True)

    def load_aligned(
        self,
        symbol: str,
        timeframes: Sequence[str] | None = None,
        *,
        base: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """
        Load canonical timeframes, align on `base` (default: finest interval present), cache
        full aligned history, then apply optional open_time bounds.
        """
        labels = list(timeframes) if timeframes is not None else ["M15", "H1", "H4"]
        canonicals_list: list[str] = []
        seen: set[str] = set()
        for lab in labels:
            iv = canonical_interval(lab)
            if iv not in seen:
                seen.add(iv)
                canonicals_list.append(iv)
        canonicals = tuple(sorted(canonicals_list, key=_interval_to_minutes))

        base_iv = (
            canonical_interval(base)
            if base is not None
            else min(canonicals, key=_interval_to_minutes)
        )
        if base_iv not in canonicals:
            raise ValueError(f"base {base_iv!r} must be one of loaded timeframes {canonicals!r}")

        sigs = _signatures_for_symbol(self.merged_dir, symbol, canonicals)
        ck = self._aligned_cache_key(symbol, base_iv, canonicals, sigs)

        df: pd.DataFrame | None = None
        if self.use_cache and ck in self._aligned_cache:
            df = self._aligned_cache[ck].copy()
        elif self.cache_dir is not None and self.use_cache:
            pq_path, meta_path = self._disk_cache_paths(ck)
            if pq_path.is_file() and meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                stored = {k: tuple(v) for k, v in meta.get("signatures", {}).items()}
                if stored == sigs:
                    df = pd.read_parquet(pq_path)

        if df is None:
            frames = self.load_multi_timeframe(symbol, list(canonicals))
            df = self.align_timeframes(frames, base_interval=base_iv)
            if self.use_cache:
                self._aligned_cache[ck] = df.copy()
            if self.cache_dir is not None and self.use_cache:
                pq_path, meta_path = self._disk_cache_paths(ck)
                df.to_parquet(pq_path, index=False)
                meta_path.write_text(
                    json.dumps({"signatures": {k: [s[0], s[1]] for k, s in sigs.items()}}),
                    encoding="utf-8",
                )

        if start_ms is not None:
            df = df[df["open_time"] >= start_ms]
        if end_ms is not None:
            df = df[df["open_time"] <= end_ms]
        return df.reset_index(drop=True)
