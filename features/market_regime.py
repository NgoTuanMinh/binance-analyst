"""
Market regime classification: ADX trend/range, ATR volatility percentiles, multi-timeframe labels.

Expects merged technical columns ``15m_*``, ``1h_*``, ``4h_*`` when available
(e.g. after :class:`features.technical.TechnicalFeatures`). Falls back to computing
ADX/ATR on ``df`` OHLCV only for the base (15m) row set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import _adx_di, _atr_wilder, _require_ohlcv, _sort_reset

# Composite labels for ``regime_M15`` / ``regime_H1`` / ``regime_H4``
REGIME_QUIET = 0
REGIME_RANGING = 1
REGIME_TRENDING = 2
REGIME_VOLATILE = 3

# Backward-compatible aliases (legacy bull/bear collapsed into TRENDING)
REGIME_RANGE = REGIME_RANGING
REGIME_HIGH_VOL = REGIME_VOLATILE
REGIME_TREND_BULL = REGIME_TRENDING
REGIME_TREND_BEAR = REGIME_TRENDING


def _rolling_rank_pct(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    """Empirical percentile of the last value within each rolling window (0–1)."""
    w = int(window)
    mp = min_periods if min_periods is not None else max(15, w // 5)

    def _pct_last(x: pd.Series) -> float:
        x = x.astype(float)
        if x.notna().sum() < 2:
            return np.nan
        last = x.iloc[-1]
        if not np.isfinite(last):
            return np.nan
        v = x.dropna()
        return float((v <= last).sum() / len(v))

    return series.rolling(w, min_periods=mp).apply(_pct_last, raw=False)


def _resolve_adx_atr(
    df: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    prefix: str,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Read ``{prefix}_adx_14`` / ``{prefix}_atr_14`` or return (None, None)."""
    adx_c = f"{prefix}_adx_14"
    atr_c = f"{prefix}_atr_14"
    adx_s = df[adx_c].astype(float) if adx_c in df.columns else None
    atr_s = df[atr_c].astype(float) if atr_c in df.columns else None
    if adx_s is None and prefix == "15m":
        adx_s = _adx_di(high, low, close, 14)["adx"]
    if atr_s is None and prefix == "15m":
        atr_s = _atr_wilder(high, low, close, 14)
    return adx_s, atr_s


def _composite_regime(
    adx: np.ndarray,
    atr_pct: np.ndarray,
    *,
    adx_trend: float,
    adx_range: float,
    vol_hi: float,
    vol_lo: float,
) -> np.ndarray:
    """
    Priority: volatile → trending → ranging → (mid ADX) quiet → ranging.
    """
    n = len(adx)
    r = np.full(n, REGIME_RANGING, dtype=np.int8)
    finite_adx = np.isfinite(adx)
    finite_pct = np.isfinite(atr_pct)

    high_vol = finite_pct & (atr_pct >= vol_hi)
    r[high_vol] = REGIME_VOLATILE

    rest = ~high_vol
    r[rest & finite_adx & (adx > adx_trend)] = REGIME_TRENDING
    r[rest & finite_adx & (adx < adx_range)] = REGIME_RANGING

    mid = rest & finite_adx & (adx >= adx_range) & (adx <= adx_trend)
    r[mid & finite_pct & (atr_pct <= vol_lo)] = REGIME_QUIET

    return r


def market_regime(
    df: pd.DataFrame,
    *,
    adx_trend_threshold: float = 25.0,
    adx_range_threshold: float = 20.0,
    atr_percentile_window: int = 100,
    volatile_percentile: float = 0.75,
    quiet_percentile: float = 0.25,
) -> pd.DataFrame:
    """
    Add regime features on the **base** bar index (typically 15m rows).

    **Booleans (M15 / base ADX + M15 ATR percentile)**

    - ``regime_trending``: ADX\\ :sub:`14` > ``adx_trend_threshold`` (default 25).
    - ``regime_ranging``: ADX\\ :sub:`14` < ``adx_range_threshold`` (default 20).
    - ``regime_volatile``: ATR percentile ≥ ``volatile_percentile`` (default 0.75).

    **Scalars**

    - ``trend_strength``: ADX\\ :sub:`14` on M15 (same as ``15m_adx_14`` when present).
    - ``volatility_regime``: rolling percentile (0–1) of M15 ATR\\ :sub:`14` in
      ``atr_percentile_window``.

    **Per-timeframe composite** (int8: 0 quiet, 1 ranging, 2 trending, 3 volatile)

    - ``regime_M15``, ``regime_H1``, ``regime_H4`` — each uses that TF's ADX/ATR merged
      onto the base index; NaN if the corresponding technical columns are missing.
    """
    _require_ohlcv(df)
    work = _sort_reset(df)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    close = work["close"].astype(float)

    adx_m15, atr_m15 = _resolve_adx_atr(work, high, low, close, "15m")
    if adx_m15 is None or atr_m15 is None:
        raise ValueError("Could not resolve M15 ADX/ATR (need OHLCV or 15m_adx_14 / 15m_atr_14).")

    adx_np = adx_m15.to_numpy()
    vol_pct = _rolling_rank_pct(atr_m15, atr_percentile_window).to_numpy()

    work["trend_strength"] = adx_m15
    work["volatility_regime"] = vol_pct

    work["regime_trending"] = (adx_np > adx_trend_threshold).astype(np.int8)
    work["regime_ranging"] = (adx_np < adx_range_threshold).astype(np.int8)
    work["regime_volatile"] = (
        np.isfinite(vol_pct) & (vol_pct >= volatile_percentile)
    ).astype(np.int8)

    tf_map = (("M15", "15m"), ("H1", "1h"), ("H4", "4h"))
    for label, prefix in tf_map:
        adx_tf, atr_tf = _resolve_adx_atr(work, high, low, close, prefix)
        col = f"regime_{label}"
        if adx_tf is None or atr_tf is None:
            work[col] = np.full(len(work), np.nan, dtype=np.float64)
            continue
        ap_tf = _rolling_rank_pct(atr_tf, atr_percentile_window).to_numpy()
        comp = _composite_regime(
            adx_tf.to_numpy(),
            ap_tf,
            adx_trend=adx_trend_threshold,
            adx_range=adx_range_threshold,
            vol_hi=volatile_percentile,
            vol_lo=quiet_percentile,
        )
        comp_f = comp.astype(np.float64)
        bad = ~(np.isfinite(adx_tf.to_numpy()) & np.isfinite(ap_tf))
        comp_f[bad] = np.nan
        work[col] = comp_f

    return work
