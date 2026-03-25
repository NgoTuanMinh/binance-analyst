"""
Technical indicators for ML feature engineering.

- :class:`TechnicalFeatures` implements the full **M15 / H1 / H4** bundle (prefixed columns).

All public functions accept optional ``timeframes``: a mapping of timeframe label
(e.g. ``\"1h\"``, ``\"4h\"``) to OHLCV DataFrames. Indicators are computed on the
primary ``df`` and on each extra frame, then higher-TF columns are merged onto
``df`` with :func:`pandas.merge_asof` (backward) on ``open_time``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REQUIRED_OHLCV = ("open_time", "open", "high", "low", "close", "volume")


def _require_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}; required: {REQUIRED_OHLCV}")


def _sort_reset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    return out.reset_index(drop=True)


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder / RMA-style smoothing (EMA with alpha = 1/period)."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return _wilder_smooth(tr, period)


def _adx_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    tr = _true_range(high, low, close)
    atr = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr.replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = _wilder_smooth(dx, period)
    return pd.DataFrame(
        {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend line and direction (+1 bullish, -1 bearish)."""
    atr = _atr_wilder(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = upper_basic.copy().to_numpy(dtype=float)
    lower = lower_basic.copy().to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)
    final_upper = np.copy(upper)
    final_lower = np.copy(lower)
    for i in range(1, n):
        if not np.isnan(upper[i]):
            if upper[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
                final_upper[i] = upper[i]
            else:
                final_upper[i] = final_upper[i - 1]
        if not np.isnan(lower[i]):
            if lower[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
                final_lower[i] = lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

    supertrend = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(final_upper[i]) or np.isnan(final_lower[i]):
            continue
        if i == 0:
            supertrend[i] = final_upper[i]
            direction[i] = -1
            continue
        if supertrend[i - 1] == final_upper[i - 1] and c[i] <= final_upper[i]:
            supertrend[i] = final_upper[i]
            direction[i] = -1
        elif supertrend[i - 1] == final_upper[i - 1] and c[i] > final_upper[i]:
            supertrend[i] = final_lower[i]
            direction[i] = 1
        elif supertrend[i - 1] == final_lower[i - 1] and c[i] >= final_lower[i]:
            supertrend[i] = final_lower[i]
            direction[i] = 1
        elif supertrend[i - 1] == final_lower[i - 1] and c[i] < final_lower[i]:
            supertrend[i] = final_upper[i]
            direction[i] = -1
        else:
            supertrend[i] = final_lower[i]
            direction[i] = 1

    idx = close.index
    return pd.DataFrame(
        {
            "supertrend": pd.Series(supertrend, index=idx),
            "supertrend_direction": pd.Series(direction, index=idx),
        }
    )


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
        }
    )


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    stoch_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d})


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    return -100.0 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct_b": pct_b,
            "bb_width": bandwidth,
        }
    )


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period), min_periods=int(period)).mean()


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    ma = tp.rolling(period, min_periods=period).mean()

    def _mean_dev(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x - np.mean(x))))

    md = tp.rolling(period, min_periods=period).apply(_mean_dev, raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _stochastic_14_3_3(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
) -> pd.DataFrame:
    """Slow stochastic: raw %K over ``k_period``, then SMA ``smooth1``, then SMA ``smooth2`` (%D)."""
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100.0 * (close - lowest_low) / denom
    k = raw_k.rolling(smooth1, min_periods=smooth1).mean()
    d = k.rolling(smooth2, min_periods=smooth2).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def _keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    mid = close.ewm(span=ema_period, adjust=False).mean()
    atr = _atr_wilder(high, low, close, atr_period)
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr
    return pd.DataFrame({"keltner_mid": mid, "keltner_upper": upper, "keltner_lower": lower})


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).fillna(0.0).cumsum()


def _rolling_volume_profile_poc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 96,
    bins: int = 24,
) -> pd.Series:
    """Point of control: bin center with max volume over rolling window (typical price)."""
    tp = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    vol = volume.astype(float)
    n = len(tp)
    out = np.full(n, np.nan, dtype=float)
    arr_tp = tp.to_numpy()
    arr_v = vol.to_numpy()
    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        tps = arr_tp[sl]
        vols = arr_v[sl]
        lo, hi = float(np.nanmin(tps)), float(np.nanmax(tps))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[i] = float(np.nanmean(tps))
            continue
        edges = np.linspace(lo, hi, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(tps, edges[1:-1], right=False), 0, bins - 1)
        agg = np.zeros(bins, dtype=float)
        for k in range(len(tps)):
            agg[int(idx[k])] += vols[k]
        out[i] = centers[int(np.argmax(agg))]
    return pd.Series(out, index=tp.index, name="vp_poc")


def _merge_timeframes(
    base: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge feature columns from higher TFs onto base using backward as-of join."""
    # Avoid deep-copying large, wide frames here (can trigger big temporary allocations
    # during block consolidation). ``sort_values`` already materializes a new frame.
    left = base.sort_values("open_time")
    for tf_label, other in frames.items():
        if other is None or other.empty:
            continue
        right = _sort_reset(other)
        # Avoid re-merging the same higher-TF column names across indicator steps
        # (e.g. ``1h_open`` after ``trend_indicators`` already exists).
        feat_cols = [
            c
            for c in right.columns
            if c != "open_time" and f"{tf_label}_{c}" not in left.columns
        ]
        if not feat_cols:
            continue
        rename = {c: f"{tf_label}_{c}" for c in feat_cols}
        r = right[["open_time", *feat_cols]].rename(columns=rename)
        r_time = f"_asof_{tf_label}"
        r = r.rename(columns={"open_time": r_time})
        left = pd.merge_asof(
            left,
            r.sort_values(r_time),
            left_on="open_time",
            right_on=r_time,
            direction="backward",
        )
        left = left.drop(columns=[r_time])
    return left.reset_index(drop=True)


def trend_indicators(
    df: pd.DataFrame,
    *,
    timeframes: Mapping[str, pd.DataFrame] | None = None,
    ema_periods: Sequence[int] = (12, 26, 50, 200),
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    adx_period: int = 14,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
    supertrend_periods: Sequence[tuple[int, float]] | None = None,
) -> pd.DataFrame:
    """
    Trend features: EMA (multiple periods), MACD, ADX/DMI, Supertrend (multiple period/mult pairs).

    If ``supertrend_periods`` is provided, each ``(period, multiplier)`` adds columns
    ``supertrend_p{period}_m{mult}`` and ``supertrend_direction_p{period}_m{mult}``.
    """
    _require_ohlcv(df)
    work = _sort_reset(df)

    def _block(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        for p in ema_periods:
            out[f"ema_{p}"] = close.ewm(span=int(p), adjust=False).mean()
        macd_df = _macd(close, macd_fast, macd_slow, macd_signal)
        for c in macd_df.columns:
            out[c] = macd_df[c]
        adx_df = _adx_di(high, low, close, adx_period)
        for c in adx_df.columns:
            out[c] = adx_df[c]
        st_specs: list[tuple[int, float]]
        if supertrend_periods:
            st_specs = [(int(a), float(b)) for a, b in supertrend_periods]
        else:
            st_specs = [(supertrend_period, supertrend_multiplier)]
        for per, mult in st_specs:
            st = _supertrend(high, low, close, period=per, multiplier=mult)
            suf = f"_p{per}_m{str(mult).replace('.', '_')}"
            out[f"supertrend{suf}"] = st["supertrend"]
            out[f"supertrend_direction{suf}"] = st["supertrend_direction"]
        return out

    out = _block(work)
    if timeframes:
        others: dict[str, pd.DataFrame] = {}
        for label, dfo in timeframes.items():
            _require_ohlcv(dfo)
            others[str(label)] = _block(_sort_reset(dfo))
        out = _merge_timeframes(out, others)
    return out


def momentum_indicators(
    df: pd.DataFrame,
    *,
    timeframes: Mapping[str, pd.DataFrame] | None = None,
    rsi_periods: Sequence[int] = (7, 14, 21),
    stoch_k: int = 14,
    stoch_d: int = 3,
    williams_periods: Sequence[int] = (14, 28),
) -> pd.DataFrame:
    """Momentum features: RSI (multiple periods), Stochastic, Williams %R."""
    _require_ohlcv(df)
    work = _sort_reset(df)

    def _block(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        for p in rsi_periods:
            out[f"rsi_{p}"] = _rsi(close, int(p))
        st = _stochastic(high, low, close, stoch_k, stoch_d)
        out["stoch_k"] = st["stoch_k"]
        out["stoch_d"] = st["stoch_d"]
        for p in williams_periods:
            out[f"williams_r_{p}"] = _williams_r(high, low, close, int(p))
        return out

    out = _block(work)
    if timeframes:
        others = {}
        for k, v in timeframes.items():
            _require_ohlcv(v)
            others[str(k)] = _block(_sort_reset(v))
        out = _merge_timeframes(out, others)
    return out


def volatility_indicators(
    df: pd.DataFrame,
    *,
    timeframes: Mapping[str, pd.DataFrame] | None = None,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_periods: Sequence[int] = (14, 21),
    keltner_ema: int = 20,
    keltner_atr: int = 10,
    keltner_mult: float = 2.0,
) -> pd.DataFrame:
    """Volatility features: Bollinger Bands, ATR (multiple), Keltner."""
    _require_ohlcv(df)
    work = _sort_reset(df)

    def _block(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        bb = _bollinger(close, bb_period, bb_std)
        for c in bb.columns:
            out[c] = bb[c]
        for p in atr_periods:
            out[f"atr_{p}"] = _atr_wilder(high, low, close, int(p))
        kel = _keltner(high, low, close, keltner_ema, keltner_atr, keltner_mult)
        for c in kel.columns:
            out[c] = kel[c]
        return out

    out = _block(work)
    if timeframes:
        others = {}
        for k, v in timeframes.items():
            _require_ohlcv(v)
            others[str(k)] = _block(_sort_reset(v))
        out = _merge_timeframes(out, others)
    return out


def volume_indicators(
    df: pd.DataFrame,
    *,
    timeframes: Mapping[str, pd.DataFrame] | None = None,
    vwap_window: int | None = None,
    vp_window: int = 96,
    vp_bins: int = 24,
) -> pd.DataFrame:
    """
    Volume features: OBV, rolling volume profile POC, VWAP.

    - **VWAP**: if ``vwap_window`` is set, uses rolling VWAP over that many bars;
      otherwise cumulative VWAP from the start of the series (typical for a single session
      slice; for multi-day data prefer passing ``vwap_window`` or pre-segmenting by day).
    """
    _require_ohlcv(df)
    work = _sort_reset(df)

    def _block(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        volume = out["volume"].astype(float)
        out["obv"] = _obv(close, volume)
        typical = (high + low + close) / 3.0
        pv = typical * volume
        if vwap_window is None:
            cum_vol = volume.cumsum()
            out["vwap"] = pv.cumsum() / cum_vol.replace(0, np.nan)
        else:
            w = int(vwap_window)
            out["vwap"] = pv.rolling(w, min_periods=1).sum() / volume.rolling(w, min_periods=1).sum().replace(
                0, np.nan
            )
        out["vp_poc"] = _rolling_volume_profile_poc(high, low, close, volume, window=vp_window, bins=vp_bins)
        return out

    out = _block(work)
    if timeframes:
        others = {}
        for k, v in timeframes.items():
            _require_ohlcv(v)
            others[str(k)] = _block(_sort_reset(v))
        out = _merge_timeframes(out, others)
    return out


class TechnicalFeatures:
    """
    Spec-complete technical bundle: all indicators for **M15, H1, H4** (prefix ``15m_``, ``1h_``, ``4h_``).

    Trend: EMA 20/50/100/200, SMA 20/50/200, MACD(12,26,9), ADX(14)+DI+/DI-,
    Supertrend (3,10) and (5,20). Momentum: RSI 7/14/21, Stochastic (14,3,3),
    Williams %R 14, CCI 20. Volatility: Bollinger (20,2) mid/upper/lower/%B/bandwidth,
    ATR 7/14/21, Keltner (20, 1.5) with ATR period 20. Volume: OBV, volume SMA 20/50, VP POC.

    Use :meth:`transform` with ``frames`` keys ``15m``, ``1h``, ``4h`` (higher TFs optional).
    Output rows follow **15m** ``open_time``; higher-TF features are ``merge_asof`` backward.
    """

    def __init__(self, *, vp_window: int = 96, vp_bins: int = 24) -> None:
        self.vp_window = int(vp_window)
        self.vp_bins = int(vp_bins)

    def _single(self, d: pd.DataFrame) -> pd.DataFrame:
        """One timeframe: ``open_time`` + indicator columns (no OHLCV)."""
        _require_ohlcv(d)
        w = _sort_reset(d)
        out = w[["open_time"]].copy()
        high = w["high"].astype(float)
        low = w["low"].astype(float)
        close = w["close"].astype(float)
        volume = w["volume"].astype(float)

        for p in (20, 50, 100, 200):
            out[f"ema_{p}"] = close.ewm(span=int(p), adjust=False).mean()
        for p in (20, 50, 200):
            out[f"sma_{p}"] = _sma(close, int(p))

        macd_df = _macd(close, 12, 26, 9)
        out["macd"] = macd_df["macd"]
        out["macd_signal"] = macd_df["macd_signal"]
        out["macd_hist"] = macd_df["macd_hist"]

        adx_df = _adx_di(high, low, close, 14)
        out["adx_14"] = adx_df["adx"]
        out["di_plus_14"] = adx_df["plus_di"]
        out["di_minus_14"] = adx_df["minus_di"]

        for per, mult in ((3, 10.0), (5, 20.0)):
            st = _supertrend(high, low, close, period=int(per), multiplier=float(mult))
            suf = f"_p{per}_m{str(mult).replace('.', '_')}"
            out[f"supertrend{suf}"] = st["supertrend"]
            out[f"supertrend_direction{suf}"] = st["supertrend_direction"]

        for p in (7, 14, 21):
            out[f"rsi_{p}"] = _rsi(close, int(p))

        stoch = _stochastic_14_3_3(high, low, close, 14, 3, 3)
        out["stoch_k_14_3"] = stoch["stoch_k"]
        out["stoch_d_14_3_3"] = stoch["stoch_d"]

        out["williams_r_14"] = _williams_r(high, low, close, 14)
        out["cci_20"] = _cci(high, low, close, 20)

        bb = _bollinger(close, 20, 2.0)
        out["bb_mid"] = bb["bb_mid"]
        out["bb_upper"] = bb["bb_upper"]
        out["bb_lower"] = bb["bb_lower"]
        out["bb_pct_b"] = bb["bb_pct_b"]
        out["bb_bandwidth"] = bb["bb_bandwidth"]

        for p in (7, 14, 21):
            out[f"atr_{p}"] = _atr_wilder(high, low, close, int(p))

        kel = _keltner(high, low, close, ema_period=20, atr_period=20, multiplier=1.5)
        out["keltner_mid"] = kel["keltner_mid"]
        out["keltner_upper"] = kel["keltner_upper"]
        out["keltner_lower"] = kel["keltner_lower"]

        out["obv"] = _obv(close, volume)
        out["volume_sma_20"] = _sma(volume, 20)
        out["volume_sma_50"] = _sma(volume, 50)
        out["vp_poc"] = _rolling_volume_profile_poc(
            high, low, close, volume, window=self.vp_window, bins=self.vp_bins
        )

        return out

    def transform(self, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Wide frame on **15m** timeline with ``15m_*``, ``1h_*``, ``4h_*`` indicator columns.

        ``frames`` must include ``15m``; ``1h`` / ``4h`` optional. Raw OHLCV = 15m only.
        """
        if "15m" not in frames:
            raise ValueError("frames must include '15m'.")

        m15 = _sort_reset(frames["15m"])
        left = m15.copy()
        f15 = self._single(m15)
        for col in f15.columns:
            if col != "open_time":
                left[f"15m_{col}"] = f15[col].to_numpy()

        others: dict[str, pd.DataFrame] = {}
        for tf in ("1h", "4h"):
            if tf in frames and frames[tf] is not None and not getattr(frames[tf], "empty", True):
                others[tf] = self._single(_sort_reset(frames[tf]))

        if others:
            left = _merge_timeframes(left, others)
        return left.reset_index(drop=True)
