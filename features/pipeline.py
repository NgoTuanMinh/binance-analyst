"""
End-to-end feature pipeline: load merged Parquet (M15/H1/H4), build features, write Parquet.

Uses ``crypto-data-pipeline/data/merged/{SYMBOL}/{SYMBOL}-{interval}.parquet`` via
:class:`backtest.data_loader.BacktestDataLoader`.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Repo root (parent of ``features/``)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MERGED = _REPO_ROOT / "crypto-data-pipeline" / "data" / "merged"
_DEFAULT_OUT = _REPO_ROOT / "features_output"
_DEFAULT_FEATURE_DATA = _REPO_ROOT / "features" / "data"

_DEFAULT_TIMEFRAMES = ("M15", "H1", "H4")


def _ensure_sys_path() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def normalize_interval_labels(intervals: Sequence[str]) -> list[str]:
    """Map labels (e.g. ``M15``, ``15m``) to canonical merge keys ``15m`` / ``1h`` / ``4h`` (deduped)."""
    _ensure_sys_path()
    from backtest.data_loader import canonical_interval

    out: list[str] = []
    seen: set[str] = set()
    for lab in intervals:
        c = canonical_interval(str(lab).strip())
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _date_strings_to_ms_range(start_date: str, end_date: str) -> tuple[int, int]:
    """UTC range inclusive. Date-only ``end_date`` is treated as end of that calendar day."""
    s = str(start_date).strip()
    e = str(end_date).strip()
    t0 = pd.Timestamp(s, tz="UTC")
    t1 = pd.Timestamp(e, tz="UTC")
    start_ms = int(t0.timestamp() * 1000)
    if "T" not in e.upper() and len(e) <= 10:
        t1 = t1 + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    end_ms = int(t1.timestamp() * 1000)
    return start_ms, end_ms


def resolve_symbols_argument(
    symbols: str | Sequence[str],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    """
    Resolve CLI-style symbol arguments:

    - ``top_100`` / ``TOP100`` — first 100 from ``crypto-data-pipeline/config/top_300_symbols.json``
    - ``top_300`` / ``TOP300`` — full list from that file
    - Path to JSON (list or ``{{\"symbols\": [...]}}``)
    - Comma / semicolon-separated list
    - Sequence of strings
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    top_json = root / "crypto-data-pipeline" / "config" / "top_300_symbols.json"

    if isinstance(symbols, (list, tuple)):
        out: list[str] = []
        seen: set[str] = set()
        for x in symbols:
            u = str(x).strip().upper()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        return out

    raw = str(symbols).strip()
    if not raw:
        return []

    key = raw.upper().replace("-", "_")
    if key in ("TOP_100", "TOP100"):
        if not top_json.is_file():
            raise FileNotFoundError(f"Expected {top_json} for {raw!r}")
        data = json.loads(top_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("top_300_symbols.json must be a JSON list")
        return [str(x).strip().upper() for x in data[:100] if str(x).strip()]

    if key in ("TOP_300", "TOP300"):
        if not top_json.is_file():
            raise FileNotFoundError(f"Expected {top_json} for {raw!r}")
        data = json.loads(top_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("top_300_symbols.json must be a JSON list")
        return [str(x).strip().upper() for x in data if str(x).strip()]

    p = Path(raw)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x).strip().upper() for x in data if str(x).strip()]
        if isinstance(data, dict):
            for k in ("symbols", "SYMBOLS"):
                if k in data and isinstance(data[k], list):
                    return [str(x).strip().upper() for x in data[k] if str(x).strip()]
        raise ValueError(f"Unrecognized symbols JSON structure in {p}")

    return [x.strip().upper() for x in raw.replace(";", ",").split(",") if x.strip()]


def load_mtf_frames(
    symbol: str,
    merged_dir: Path | str,
    *,
    timeframes: Sequence[str] = _DEFAULT_TIMEFRAMES,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV dict keyed by canonical interval (``15m``, ``1h``, ``4h``)."""
    _ensure_sys_path()
    from backtest.data_loader import BacktestDataLoader

    loader = BacktestDataLoader(merged_dir)
    return loader.load_multi_timeframe(symbol, list(timeframes), start_ms=start_ms, end_ms=end_ms)


def build_feature_matrix(
    symbol: str,
    frames: dict[str, pd.DataFrame],
    *,
    include_technical: bool = True,
    technical_vp_window: int = 96,
    technical_vp_bins: int = 24,
    include_price_action: bool = True,
    include_smart_money: bool = True,
    include_market_regime: bool = True,
    include_targets: bool = True,
    target_horizons: Sequence[int] = (1, 4, 16, 96),
    include_trade_targets: bool = True,
    target_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build a single wide DataFrame on **base timeframe** (finest in ``frames`` — expected ``15m``).

    Expects ``frames`` keys ``15m``, ``1h``, ``4h`` (or subset; higher TF optional).

    Targets: see :func:`features.target.make_targets`. Trade-related columns need ``1h``/``4h``
    in ``frames`` when ``include_trade_targets`` is True (passed as ``mtf_frames``).
    Extra :func:`make_targets` kwargs go through ``target_kwargs``.
    """
    if "15m" not in frames:
        raise KeyError("frames must include '15m' as base for this pipeline")

    higher: dict[str, pd.DataFrame] = {}
    if "1h" in frames:
        higher["1h"] = frames["1h"]
    if "4h" in frames:
        higher["4h"] = frames["4h"]

    from .price_action import candle_patterns, fvg_detection, support_resistance, swing_points
    from .technical import TechnicalFeatures

    if include_technical:
        base = TechnicalFeatures(vp_window=technical_vp_window, vp_bins=technical_vp_bins).transform(frames)
    else:
        base = frames["15m"].copy()

    if include_price_action:
        base = candle_patterns(base)
        base = support_resistance(base, timeframes=higher or None)
        base = swing_points(base)
        base = fvg_detection(base)

    if include_smart_money:
        from .smart_money import smart_money_features

        base = smart_money_features(base)

    if include_market_regime:
        from .market_regime import market_regime

        base = market_regime(base)

    base.insert(0, "symbol", symbol)

    if include_targets:
        from .target import make_targets

        tk: dict[str, Any] = {"legacy_horizons": target_horizons, "mtf_frames": frames}
        if target_kwargs:
            tk.update(target_kwargs)
        if not include_trade_targets:
            tk["include_trade_targets"] = False
            tk["mtf_frames"] = None
        base = make_targets(base, **tk)

    return base.reset_index(drop=True)


class FeaturePipeline:
    """
    High-level orchestration: load merged Parquet for many symbols / intervals, build the full
    feature matrix, save Parquet per symbol, and inspect feature names / correlation ranking.

    Requires **15m** in ``intervals`` (base bar series). Default merged root:
    ``crypto-data-pipeline/data/merged``.
    """

    def __init__(
        self,
        symbols: str | Sequence[str],
        intervals: Sequence[str],
        start_date: str,
        end_date: str,
        *,
        merged_dir: Path | str | None = None,
        output_dir: str | Path | None = None,
        **feature_matrix_kwargs: Any,
    ) -> None:
        self.symbols = resolve_symbols_argument(symbols)
        if not self.symbols:
            raise ValueError("No symbols resolved.")
        self.intervals = normalize_interval_labels(intervals)
        if "15m" not in self.intervals:
            raise ValueError("intervals must include 15m (base timeframe for the feature matrix).")
        self.date_range: tuple[str, str] = (str(start_date), str(end_date))
        self.start_ms, self.end_ms = _date_strings_to_ms_range(start_date, end_date)
        self.merged_dir = Path(merged_dir) if merged_dir is not None else _DEFAULT_MERGED
        self.output_dir = Path(output_dir) if output_dir is not None else _DEFAULT_FEATURE_DATA
        self._feature_matrix_kwargs = feature_matrix_kwargs
        self._frames_by_symbol: dict[str, dict[str, pd.DataFrame]] = {}
        self._feature_by_symbol: dict[str, pd.DataFrame] = {}

    def load_data(self) -> FeaturePipeline:
        """Load ``data/merged`` Parquet for all ``self.symbols`` and ``self.intervals``."""
        self._frames_by_symbol.clear()
        for sym in self.symbols:
            self._frames_by_symbol[sym] = load_mtf_frames(
                sym,
                self.merged_dir,
                timeframes=self.intervals,
                start_ms=self.start_ms,
                end_ms=self.end_ms,
            )
        return self

    def generate_all_features(self) -> FeaturePipeline:
        """Run technical, price_action, smart_money, market_regime, targets for every symbol."""
        if not self._frames_by_symbol:
            self.load_data()
        self._feature_by_symbol.clear()
        for sym, frames in self._frames_by_symbol.items():
            self._feature_by_symbol[sym] = build_feature_matrix(sym, frames, **self._feature_matrix_kwargs)
        return self

    def save_features(
        self,
        output_dir: str | Path | None = None,
        *,
        compression: str = "snappy",
    ) -> dict[str, Path]:
        """Write ``{symbol}_features.parquet`` under ``output_dir`` (default ``features/data``)."""
        out = Path(output_dir) if output_dir is not None else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        if not self._feature_by_symbol:
            raise RuntimeError("Call generate_all_features() before save_features().")
        paths: dict[str, Path] = {}
        for sym, df in self._feature_by_symbol.items():
            p = out / f"{sym}_features.parquet"
            df.to_parquet(p, index=False, compression=compression)
            paths[sym] = p
        return paths

    def get_feature_names(
        self,
        *,
        include_targets: bool = False,
        include_ids: bool = False,
    ) -> list[str]:
        """
        Column names from the first symbol's matrix (schemas are identical across symbols).

        By default drops ``open_time``, ``symbol``, and any column starting with ``target_``.
        """
        if not self._feature_by_symbol:
            raise RuntimeError("Call generate_all_features() before get_feature_names().")
        df = next(iter(self._feature_by_symbol.values()))
        cols = list(df.columns)
        drop: set[str] = set()
        if not include_ids:
            drop.update({"open_time", "symbol"})
        if not include_targets:
            drop.update(c for c in cols if str(c).startswith("target_"))
        return [c for c in cols if c not in drop]

    def combined_features(self) -> pd.DataFrame:
        """Vertically stack all per-symbol feature matrices."""
        if not self._feature_by_symbol:
            raise RuntimeError("Call generate_all_features() first.")
        return pd.concat(self._feature_by_symbol.values(), ignore_index=True)

    def get_feature_importance_ranking(
        self,
        target_col: str,
        *,
        method: str = "pearson",
        min_non_na: int = 50,
    ) -> pd.DataFrame:
        """
        Rank features by absolute linear (Pearson) or monotonic (Spearman) correlation with ``target_col``.

        Uses :meth:`combined_features` (all symbols). Non-numeric / constant columns skipped.
        """
        if method not in ("pearson", "spearman"):
            raise ValueError("method must be 'pearson' or 'spearman'")
        big = self.combined_features()
        if target_col not in big.columns:
            raise KeyError(f"target_col {target_col!r} not in matrix columns")

        feat_cols = self.get_feature_names(include_targets=False, include_ids=False)
        y = big[target_col]
        rows: list[tuple[str, float, float]] = []
        for col in feat_cols:
            if col not in big.columns:
                continue
            s = big[col]
            if not pd.api.types.is_numeric_dtype(s):
                continue
            pair = pd.concat([y, s], axis=1).dropna()
            if len(pair) < min_non_na:
                continue
            if pair[col].std() == 0 or pair[target_col].std() == 0:
                continue
            c = float(pair[target_col].corr(pair[col], method=method))
            if not np.isfinite(c):
                continue
            rows.append((col, c, abs(c)))

        out = pd.DataFrame(rows, columns=["feature", "correlation", "abs_correlation"])
        return out.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def build_and_write_symbol(
    symbol: str,
    merged_dir: Path | str,
    output_dir: Path | str,
    *,
    timeframes: Sequence[str] = _DEFAULT_TIMEFRAMES,
    start_ms: int | None = None,
    end_ms: int | None = None,
    compression: str = "snappy",
    **kwargs: Any,
) -> Path:
    """Load merged data, build features, write ``{output_dir}/{symbol}_features.parquet``."""
    merged_dir = Path(merged_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_mtf_frames(symbol, merged_dir, timeframes=timeframes, start_ms=start_ms, end_ms=end_ms)
    df = build_feature_matrix(symbol, frames, **kwargs)
    out_path = output_dir / f"{symbol}_features.parquet"
    df.to_parquet(out_path, index=False, compression=compression)
    return out_path


def _worker(payload: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """Picklable worker for process pool."""
    try:
        root = Path(payload["root"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        path = build_and_write_symbol(
            payload["symbol"],
            payload["merged_dir"],
            payload["output_dir"],
            timeframes=payload.get("timeframes", ("M15", "H1", "H4")),
            start_ms=payload.get("start_ms"),
            end_ms=payload.get("end_ms"),
            compression=payload.get("compression", "snappy"),
            include_technical=payload.get("include_technical", True),
            include_price_action=payload.get("include_price_action", True),
            include_smart_money=payload.get("include_smart_money", True),
            include_market_regime=payload.get("include_market_regime", True),
            include_targets=payload.get("include_targets", True),
            target_horizons=tuple(payload.get("target_horizons", (1, 4, 16, 96))),
            include_trade_targets=payload.get("include_trade_targets", True),
            target_kwargs=payload.get("target_kwargs"),
            technical_vp_window=int(payload.get("technical_vp_window", 96)),
            technical_vp_bins=int(payload.get("technical_vp_bins", 24)),
        )
        return payload["symbol"], str(path), None
    except Exception as exc:
        return payload.get("symbol", "?"), None, str(exc)


def run_parallel(
    symbols: Sequence[str],
    merged_dir: Path | str,
    output_dir: Path | str,
    *,
    max_workers: int = 4,
    **kwargs: Any,
) -> dict[str, str | None]:
    """
    Build Parquet feature files for many symbols in parallel (process pool).

    Returns mapping ``symbol -> output path``; on failure value is ``None`` (check stderr/logs).
    """
    merged_dir = Path(merged_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = str(_REPO_ROOT)
    payloads = [
        {
            "root": root,
            "symbol": s.strip().upper(),
            "merged_dir": str(merged_dir),
            "output_dir": str(output_dir),
            **kwargs,
        }
        for s in symbols
        if s.strip()
    ]
    results: dict[str, str | None] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, p): p["symbol"] for p in payloads}
        for fut in as_completed(futs):
            sym, path, err = fut.result()
            if err:
                print(f"[features] ERROR {sym}: {err}", file=sys.stderr)
                results[sym] = None
            else:
                results[sym] = path
    return results


def main() -> None:
    _ensure_sys_path()
    parser = argparse.ArgumentParser(description="Build ML feature Parquet from merged OHLCV")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--symbols-json", type=str, help="JSON list or {symbols: [...]}")
    parser.add_argument("--merged-dir", type=str, default=str(_DEFAULT_MERGED))
    parser.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=1, help="1 = sequential in this process; >1 uses process pool")
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--no-targets", action="store_true")
    parser.add_argument("--compression", type=str, default="snappy")
    args = parser.parse_args()

    symbols: list[str] = []
    if args.symbols_json:
        raw = json.loads(Path(args.symbols_json).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            symbols = [str(x).strip().upper() for x in raw if str(x).strip()]
        elif isinstance(raw, dict):
            for key in ("symbols", "SYMBOLS"):
                if key in raw and isinstance(raw[key], list):
                    symbols = [str(x).strip().upper() for x in raw[key] if str(x).strip()]
                    break
        else:
            raise SystemExit("Unsupported --symbols-json format")
    elif args.symbols:
        symbols = [x.strip().upper() for x in args.symbols.replace(";", ",").split(",") if x.strip()]
    else:
        raise SystemExit("Provide --symbols or --symbols-json")

    kw: dict[str, Any] = {
        "include_targets": not args.no_targets,
        "compression": args.compression,
    }
    if args.start_ms is not None:
        kw["start_ms"] = args.start_ms
    if args.end_ms is not None:
        kw["end_ms"] = args.end_ms

    if args.workers <= 1:
        for sym in symbols:
            try:
                p = build_and_write_symbol(sym, args.merged_dir, args.output_dir, **kw)
                print(p)
            except Exception as exc:
                print(f"[features] ERROR {sym}: {exc}", file=sys.stderr)
    else:
        out = run_parallel(symbols, args.merged_dir, args.output_dir, max_workers=args.workers, **kw)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
