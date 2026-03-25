"""
Target variables for supervised ML: horizon returns, binaries, multi-class, risk-adjusted, strategy labels.

Base timeframe is assumed **15m** unless ``bar_minutes`` is set. Calendar horizons:

- 1h → ``60 // bar_minutes`` bars
- 4h → ``240 // bar_minutes`` bars
- 1d → ``1440 // bar_minutes`` bars

Trade-related targets need **1h + 4h OHLCV** merged onto the base index; pass ``mtf_frames`` from the pipeline.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .technical import _merge_timeframes, _require_ohlcv, _sort_reset, _atr_wilder


def _bars_for_minutes(bar_minutes: int, *, hours: int | None = None, days: int | None = None) -> int:
    bm = int(bar_minutes)
    if hours is not None:
        return max(1, int(round(60 * hours / bm)))
    if days is not None:
        return max(1, int(round(1440 * days / bm)))
    raise ValueError("Specify hours= or days=")


def _attach_mtf_ohlcv(work: pd.DataFrame, mtf_frames: Mapping[str, pd.DataFrame] | None) -> pd.DataFrame:
    """Prefix M15 OHLCV as ``15m_*``; merge raw ``1h`` / ``4h`` OHLCV via as-of join."""
    out = work.copy()
    for c in ("open", "high", "low", "close", "volume"):
        pc = f"15m_{c}"
        if pc not in out.columns and c in out.columns:
            out[pc] = out[c].astype(float)
    if not mtf_frames:
        return out
    others: dict[str, pd.DataFrame] = {}
    for k in ("1h", "4h"):
        v = mtf_frames.get(k)
        if v is not None and not getattr(v, "empty", True):
            others[k] = _sort_reset(v)
    if others:
        out = _merge_timeframes(out, others)
    return out


def _has_strategy_ohlcv(df: pd.DataFrame) -> bool:
    need = [
        "15m_open",
        "15m_high",
        "15m_low",
        "15m_close",
        "1h_high",
        "1h_low",
        "1h_close",
        "4h_high",
        "4h_low",
        "4h_close",
    ]
    return all(c in df.columns for c in need)


def _forward_pct_return(close: pd.Series, bars: int) -> pd.Series:
    b = int(bars)
    fwd = close.astype(float).shift(-b)
    return (fwd / close.replace(0, np.nan) - 1.0) * 100.0


def _forward_simple_return(close: pd.Series, bars: int) -> pd.Series:
    b = int(bars)
    fwd = close.astype(float).shift(-b)
    return (fwd - close) / close.replace(0, np.nan)


def make_targets(
    df: pd.DataFrame,
    *,
    bar_minutes: int = 15,
    mtf_frames: Mapping[str, pd.DataFrame] | None = None,
    # Legacy columns (pipeline / older notebooks)
    include_legacy_fwd: bool = True,
    legacy_horizons: Sequence[int] = (1, 4, 16, 96),
    log_returns_legacy: bool = True,
    binary_threshold: float | None = None,
    # Spec: calendar horizons (percent return)
    include_horizon_returns: bool = True,
    include_binary_up: bool = True,
    include_target_class: bool = True,
    include_risk_adjusted: bool = True,
    include_trade_targets: bool = True,
    # Binary thresholds on **percent** return
    up_threshold_1h_pct: float = 0.5,
    up_threshold_4h_pct: float = 1.0,
    up_threshold_1d_pct: float = 2.0,
    # Multi-class (symmetric strong move vs sideways)
    target_class_horizon_bars: int | None = None,
    target_class_strong_pct: float = 1.0,
    target_class_sideways_pct: float = 0.35,
    # Risk-adjusted: forward_simple_return / ATR
    risk_adj_atr_period: int = 14,
    trade_strategy_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Append target columns aligned per bar (tail rows NaN where forward window missing).

    **Returns (% price change)**

    - ``target_1h_return``, ``target_4h_return``, ``target_1d_return``

    **Binary (1 if forward return exceeds threshold else 0)**

    - ``target_up_1h`` (> ``up_threshold_1h_pct``), ``target_up_4h``, ``target_up_1d``

    **Multi-class**

    - ``target_class``: ``-1`` if forward %% return < ``-target_class_strong_pct``, ``1`` if
      > ``+target_class_strong_pct``, else ``0`` (sideways or mild move), including when
      ``abs(ret) <= target_class_sideways_pct``.

    **Risk-adjusted**

    - ``target_risk_adjusted_return_1h``, ``_4h``, ``_1d``: forward simple return / ATR
      (ATR Wilder on base ``close`` series; uses ``15m_atr_*`` only if exact column absent).

    **Strategy-linked** (``SwingTradingStrategy``; NaN if no signal at bar)

    - ``would_trade_successful``: 1 = TP hit before SL within time stop, 0 otherwise.
    - ``optimal_sl`` / ``optimal_tp``: MAE (adverse extreme) and MFE (favorable extreme)
      **prices** over the simulation window (entry bar excluded).
    - ``optimal_sl_pct`` / ``optimal_tp_pct``: same vs entry as %% (always ≥ 0).

    **Legacy** (if ``include_legacy_fwd``): ``target_fwd_ret_{h}``, optional ``target_dir_{h}``.
    """
    _require_ohlcv(df)
    work = _sort_reset(df)
    close = work["close"].astype(float)
    high = work["high"].astype(float)
    low = work["low"].astype(float)

    b1h = _bars_for_minutes(bar_minutes, hours=1)
    b4h = _bars_for_minutes(bar_minutes, hours=4)
    b1d = _bars_for_minutes(bar_minutes, days=1)
    class_h = int(target_class_horizon_bars) if target_class_horizon_bars is not None else b1d

    if include_horizon_returns:
        work["target_1h_return"] = _forward_pct_return(close, b1h)
        work["target_4h_return"] = _forward_pct_return(close, b4h)
        work["target_1d_return"] = _forward_pct_return(close, b1d)

    if include_binary_up:
        r1 = _forward_pct_return(close, b1h)
        r4 = _forward_pct_return(close, b4h)
        rd = _forward_pct_return(close, b1d)
        work["target_up_1h"] = (r1 > up_threshold_1h_pct).astype(np.float64)
        work.loc[r1.isna(), "target_up_1h"] = np.nan
        work["target_up_4h"] = (r4 > up_threshold_4h_pct).astype(np.float64)
        work.loc[r4.isna(), "target_up_4h"] = np.nan
        work["target_up_1d"] = (rd > up_threshold_1d_pct).astype(np.float64)
        work.loc[rd.isna(), "target_up_1d"] = np.nan

    if include_target_class:
        r_cls = _forward_pct_return(close, class_h)
        tc = np.full(len(work), np.nan, dtype=np.float64)
        fin = r_cls.notna()
        absr = r_cls.abs()
        # Sideways: small move; strong: beyond ±target_class_strong_pct; else still sideways
        sideways = absr <= target_class_sideways_pct
        strong_up = fin & (r_cls > target_class_strong_pct)
        strong_dn = fin & (r_cls < -target_class_strong_pct)
        tc[fin & sideways.to_numpy()] = 0.0
        tc[strong_up.to_numpy()] = 1.0
        tc[strong_dn.to_numpy()] = -1.0
        mid = fin & ~sideways & ~strong_up & ~strong_dn
        tc[mid.to_numpy()] = 0.0
        work["target_class"] = tc

    if include_risk_adjusted:
        atr_col = f"15m_atr_{risk_adj_atr_period}"
        if atr_col in work.columns:
            atr = work[atr_col].astype(float).replace(0, np.nan)
        else:
            atr = _atr_wilder(high, low, close, int(risk_adj_atr_period)).replace(0, np.nan)
        for name, nb in (("1h", b1h), ("4h", b4h), ("1d", b1d)):
            fr = _forward_simple_return(close, nb)
            work[f"target_risk_adjusted_return_{name}"] = fr / atr

    if include_legacy_fwd:
        for h in legacy_horizons:
            h = int(h)
            if h <= 0:
                raise ValueError("legacy_horizons must be positive")
            fwd_close = close.shift(-h)
            if log_returns_legacy:
                ratio = fwd_close / close.replace(0, np.nan)
                work[f"target_fwd_ret_{h}"] = np.log(ratio)
            else:
                work[f"target_fwd_ret_{h}"] = (fwd_close - close) / close.replace(0, np.nan)

            diff = fwd_close - close
            if binary_threshold is None:
                work[f"target_dir_{h}"] = np.sign(diff).astype(float)
                work.loc[diff.isna(), f"target_dir_{h}"] = np.nan
            else:
                thr = float(binary_threshold) * close
                up = diff > thr
                down = diff < -thr
                td = np.full(len(work), np.nan, dtype=np.float64)
                td[up.to_numpy(dtype=bool)] = 1.0
                td[down.to_numpy(dtype=bool)] = -1.0
                neutral = ~(up | down) & diff.notna()
                td[neutral.to_numpy(dtype=bool)] = 0.0
                work[f"target_dir_{h}"] = td

    if include_trade_targets:
        _add_trade_targets(
            work,
            mtf_frames=mtf_frames,
            bar_minutes=bar_minutes,
            config=trade_strategy_config,
        )

    return work


def _add_trade_targets(
    work: pd.DataFrame,
    *,
    mtf_frames: Mapping[str, pd.DataFrame] | None,
    bar_minutes: int,
    config: Mapping[str, Any] | None,
) -> None:
    """In-place: append strategy columns; uses lazy import of ``SwingTradingStrategy``."""
    prep = _attach_mtf_ohlcv(work, mtf_frames)
    if not _has_strategy_ohlcv(prep):
        work["would_trade_successful"] = np.nan
        work["optimal_sl"] = np.nan
        work["optimal_tp"] = np.nan
        work["optimal_sl_pct"] = np.nan
        work["optimal_tp_pct"] = np.nan
        return

    try:
        from backtest.configs.default_config import DEFAULT_BACKTEST_CONFIG
        from backtest.models import SignalSide
        from backtest.strategies.swing_strategy import SwingTradingStrategy
    except ImportError:
        work["would_trade_successful"] = np.nan
        work["optimal_sl"] = np.nan
        work["optimal_tp"] = np.nan
        work["optimal_sl_pct"] = np.nan
        work["optimal_tp_pct"] = np.nan
        return

    cfg = {**DEFAULT_BACKTEST_CONFIG, **(dict(config) if config else {})}
    strat = SwingTradingStrategy(cfg)
    prepared = strat.prepare(prep)
    # Keep strategy prep narrow to avoid large DataFrame copies inside `strat.prepare`.
    # Only columns used by SwingTradingStrategy are required.
    # need_cols = [
    #     "open_time",
    #     "open",
    #     "high",
    #     "low",
    #     "close",
    #     "15m_open",
    #     "15m_high",
    #     "15m_low",
    #     "15m_close",
    #     "1h_high",
    #     "1h_low",
    #     "1h_close",
    #     "4h_high",
    #     "4h_low",
    #     "4h_close",
    # ]
    # prep_min = prep[[c for c in need_cols if c in prep.columns]]
    # prepared = strat.prepare(prep_min)


    max_bars = max(1, int(round(float(cfg["time_stop_hours"]) * 60.0 / float(bar_minutes))))

    m_low = prepared["low"].astype(float).to_numpy()
    m_high = prepared["high"].astype(float).to_numpy()
    n = len(prepared)

    ok = np.full(n, np.nan, dtype=np.float64)
    sl_px = np.full(n, np.nan, dtype=np.float64)
    tp_px = np.full(n, np.nan, dtype=np.float64)
    sl_pct = np.full(n, np.nan, dtype=np.float64)
    tp_pct = np.full(n, np.nan, dtype=np.float64)

    setup_mask = (
        prepared["strat_long_setup"].to_numpy(dtype=bool)
        | prepared["strat_short_setup"].to_numpy(dtype=bool)
    )
    for i in np.flatnonzero(setup_mask):
        sig = strat.signal_at(prepared, i)
        if sig is None:
            continue
        entry = float(sig.price)
        if entry <= 0:
            continue
        sl = float(sig.meta["stop_loss"])
        tp = float(sig.meta["take_profit"])
        is_long = sig.side == SignalSide.LONG

        end = min(n, i + 1 + max_bars)
        if i + 1 >= end:
            continue

        mae_price = entry
        mfe_price = entry

        if is_long:
            mae_price = float(np.nanmin(m_low[i + 1 : end]))
            mfe_price = float(np.nanmax(m_high[i + 1 : end]))
            for j in range(i + 1, end):
                lo, hi = m_low[j], m_high[j]
                if lo <= sl:
                    success = 0.0
                    break
                if hi >= tp:
                    success = 1.0
                    break
            else:
                success = 0.0
        else:
            mae_price = float(np.nanmax(m_high[i + 1 : end]))
            mfe_price = float(np.nanmin(m_low[i + 1 : end]))
            for j in range(i + 1, end):
                lo, hi = m_low[j], m_high[j]
                if hi >= sl:
                    success = 0.0
                    break
                if lo <= tp:
                    success = 1.0
                    break
            else:
                success = 0.0

        ok[i] = success
        sl_px[i] = mae_price
        tp_px[i] = mfe_price
        if is_long:
            sl_pct[i] = max(0.0, (entry - mae_price) / entry * 100.0)
            tp_pct[i] = max(0.0, (mfe_price - entry) / entry * 100.0)
        else:
            sl_pct[i] = max(0.0, (mae_price - entry) / entry * 100.0)
            tp_pct[i] = max(0.0, (entry - mfe_price) / entry * 100.0)

    work["would_trade_successful"] = ok
    work["optimal_sl"] = sl_px
    work["optimal_tp"] = tp_px
    work["optimal_sl_pct"] = sl_pct
    work["optimal_tp_pct"] = tp_pct
