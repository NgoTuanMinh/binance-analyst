"""
Price-action features for ML: candle patterns, S/R, swings, fair value gaps.

Optional ``timeframes`` for :func:`support_resistance` merges higher-TF S/R columns via
``merge_asof`` on ``open_time`` (same as :mod:`features.technical`).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .technical import _atr_wilder, _merge_timeframes, _require_ohlcv, _sort_reset

# --- Fractal swing detection (used by support_resistance & swing_points) ---


def _fractal_swing_mask_simple(high: pd.Series, low: pd.Series, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractal pivot: high[i] / low[i] strictly higher/lower than all ``left`` and ``right`` adjacent bars."""
    h = high.astype(float).to_numpy()
    l = low.astype(float).to_numpy()
    n = len(h)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if all(h[i] > h[j] for j in range(i - left, i)) and all(
            h[i] > h[j] for j in range(i + 1, i + right + 1)
        ):
            sh[i] = True
        if all(l[i] < l[j] for j in range(i - left, i)) and all(
            l[i] < l[j] for j in range(i + 1, i + right + 1)
        ):
            sl[i] = True
    return sh, sl


def candle_patterns(
    df: pd.DataFrame,
    *,
    body_ratio_doji: float = 0.1,
    long_body_ratio: float = 0.55,
    small_body_ratio: float = 0.35,
    pin_wick_ratio: float = 0.6,
) -> pd.DataFrame:
    """
    One-hot (0/1) candlestick patterns:

    Doji; Hammer; Shooting Star; Bull/Bear Engulfing; Morning/Evening Star;
    Three White Soldiers / Three Black Crows; Pin Bar (long wick, small body).
    """
    _require_ohlcv(df)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    body_pct = body / rng
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull = c > o
    bear = c < o

    pat_doji = ((body_pct <= body_ratio_doji) & (rng > 0)).astype(np.int8)

    pat_hammer = (
        (lower >= 2 * body)
        & (upper <= body * 1.05)
        & (rng > 0)
        & (lower >= 0.5 * rng)
    ).astype(np.int8)

    pat_shooting_star = (
        (upper >= 2 * body)
        & (lower <= body * 1.05)
        & (rng > 0)
        & bear
        & (upper >= 0.6 * rng)
    ).astype(np.int8)

    prev_bull = bull.shift(1).fillna(False)
    prev_bear = bear.shift(1).fillna(False)
    o1, c1 = o.shift(1), c.shift(1)
    bull_engulf = (
        prev_bear
        & bull
        & (o <= c1)
        & (c >= o1)
        & (c - o) > (o1 - c1)
    ).astype(np.int8)
    bear_engulf = (
        prev_bull
        & bear
        & (o >= c1)
        & (c <= o1)
        & (o - c) > (c1 - o1)
    ).astype(np.int8)
    pat_bull_engulfing = bull_engulf
    pat_bear_engulfing = bear_engulf

    o2, h2, l2, c2 = o.shift(2), h.shift(2), l.shift(2), c.shift(2)
    rng2 = (h2 - l2).replace(0, np.nan)
    body2 = (c2 - o2).abs()
    mid1 = (o2 + c2) / 2
    middle_small = body.shift(1) / (h.shift(1) - l.shift(1)).replace(0, np.nan) <= small_body_ratio
    pat_morning_star = (
        (c2 < o2)
        & (body2 / rng2 >= long_body_ratio)
        & middle_small.fillna(False)
        & bull
        & (c > mid1)
    ).astype(np.int8)

    pat_evening_star = (
        (c2 > o2)
        & (body2 / rng2 >= long_body_ratio)
        & middle_small.fillna(False)
        & bear
        & (c < mid1)
    ).astype(np.int8)

    # Three White Soldiers: three consecutive strong bullish candles, higher closes
    bull3 = bull & bull.shift(1).fillna(False) & bull.shift(2).fillna(False)
    body3 = body / rng.replace(0, np.nan)
    higher_closes = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    pat_three_white_soldiers = (
        bull3 & higher_closes & (body3 >= long_body_ratio * 0.85) & (body3.shift(1) >= long_body_ratio * 0.85) & (body3.shift(2) >= long_body_ratio * 0.85)
    ).astype(np.int8)

    bear3 = bear & bear.shift(1).fillna(False) & bear.shift(2).fillna(False)
    lower_closes = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    pat_three_black_crows = (
        bear3
        & lower_closes
        & (body3 >= long_body_ratio * 0.85)
        & (body3.shift(1) >= long_body_ratio * 0.85)
        & (body3.shift(2) >= long_body_ratio * 0.85)
    ).astype(np.int8)

    # Pin bar: small body, dominant wick (bullish pin = long lower wick; bearish = long upper)
    long_lower_pin = (lower >= pin_wick_ratio * rng) & (body_pct <= small_body_ratio) & (rng > 0)
    long_upper_pin = (upper >= pin_wick_ratio * rng) & (body_pct <= small_body_ratio) & (rng > 0)
    pat_pin_bar = (long_lower_pin | long_upper_pin).astype(np.int8)

    out = _sort_reset(df)
    out["pat_doji"] = pat_doji
    out["pat_hammer"] = pat_hammer
    out["pat_shooting_star"] = pat_shooting_star
    out["pat_bull_engulfing"] = pat_bull_engulfing
    out["pat_bear_engulfing"] = pat_bear_engulfing
    out["pat_morning_star"] = pat_morning_star
    out["pat_evening_star"] = pat_evening_star
    out["pat_three_white_soldiers"] = pat_three_white_soldiers
    out["pat_three_black_crows"] = pat_three_black_crows
    out["pat_pin_bar"] = pat_pin_bar
    return out


def _count_touches_near_level(level: float, prices: list[float], rel_tol: float = 0.001) -> int:
    """Count pivots in ``prices`` within relative tolerance of ``level``."""
    if not np.isfinite(level) or not prices:
        return 0
    tol = max(abs(level) * rel_tol, 1e-12)
    return sum(1 for p in prices if abs(p - level) <= tol)


def support_resistance(
    df: pd.DataFrame,
    *,
    swing_left: int = 2,
    swing_right: int = 2,
    max_pivot_levels: int = 30,
    atr_period: int = 14,
    at_level_atr: float = 0.25,
    touch_rel_tol: float = 0.001,
    timeframes: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Swing-based S/R with ATR-normalized distances, proximity flags, and touch counts.

    Adds (among others): ``distance_to_nearest_support``, ``distance_to_nearest_resistance`` (÷ ATR),
    ``is_at_support``, ``is_at_resistance``, ``nearest_support_touches``, ``nearest_resistance_touches``,
    plus legacy ``sr_*`` columns.
    """
    _require_ohlcv(df)
    work = _sort_reset(df)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    close = work["close"].astype(float)
    atr = _atr_wilder(high, low, close, atr_period).replace(0, np.nan)
    sh_mask, sl_mask = _fractal_swing_mask_simple(high, low, swing_left, swing_right)

    n = len(work)
    pivot_highs: list[float] = []
    pivot_lows: list[float] = []
    ns = np.full(n, np.nan)
    nr = np.full(n, np.nan)
    sup_touches = np.zeros(n, dtype=np.int32)
    res_touches = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if sh_mask[i]:
            pivot_highs.append(float(high.iloc[i]))
            if len(pivot_highs) > max_pivot_levels:
                pivot_highs.pop(0)
        if sl_mask[i]:
            pivot_lows.append(float(low.iloc[i]))
            if len(pivot_lows) > max_pivot_levels:
                pivot_lows.pop(0)
        c = float(close.iloc[i])
        below = [p for p in pivot_lows if p < c]
        above = [p for p in pivot_highs if p > c]
        if below:
            lvl = max(below)
            ns[i] = lvl
            sup_touches[i] = _count_touches_near_level(lvl, pivot_lows, touch_rel_tol)
        if above:
            lvl = min(above)
            nr[i] = lvl
            res_touches[i] = _count_touches_near_level(lvl, pivot_highs, touch_rel_tol)

    dist_sup = close.to_numpy() - ns
    dist_res = nr - close.to_numpy()
    atr_a = atr.to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        d_sup_atr = dist_sup / atr_a
        d_res_atr = dist_res / atr_a

    work["sr_nearest_support"] = ns
    work["sr_nearest_resistance"] = nr
    work["sr_dist_support"] = dist_sup
    work["sr_dist_resistance"] = dist_res
    c_np = close.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        work["sr_dist_support_pct"] = dist_sup / np.where(c_np == 0, np.nan, c_np)
        work["sr_dist_resistance_pct"] = dist_res / np.where(c_np == 0, np.nan, c_np)

    work["distance_to_nearest_support"] = d_sup_atr
    work["distance_to_nearest_resistance"] = d_res_atr

    near_sup = np.abs(dist_sup) <= (at_level_atr * atr_a)
    near_res = np.abs(dist_res) <= (at_level_atr * atr_a)
    work["is_at_support"] = (near_sup & np.isfinite(ns)).astype(np.int8)
    work["is_at_resistance"] = (near_res & np.isfinite(nr)).astype(np.int8)

    work["nearest_support_touches"] = sup_touches
    work["nearest_resistance_touches"] = res_touches
    work["strength_of_level"] = np.maximum(sup_touches, res_touches)

    if timeframes:
        others: dict[str, pd.DataFrame] = {}
        for label, dfo in timeframes.items():
            _require_ohlcv(dfo)
            others[str(label)] = support_resistance(
                dfo,
                swing_left=swing_left,
                swing_right=swing_right,
                max_pivot_levels=max_pivot_levels,
                atr_period=atr_period,
                at_level_atr=at_level_atr,
                touch_rel_tol=touch_rel_tol,
                timeframes=None,
            )
        work = _merge_timeframes(work, others)
    return work


def swing_points(
    df: pd.DataFrame,
    *,
    swing_left: int = 2,
    swing_right: int = 2,
) -> pd.DataFrame:
    """
    Fractal swings with spec column names:

    - ``is_swing_high``, ``is_swing_low`` (0/1)
    - ``bars_since_swing_high``, ``bars_since_swing_low``
    - ``swing_high_distance_pct``: percent distance from ``close`` to last swing high above price
      (positive if swing high is above close).
    """
    _require_ohlcv(df)
    work = _sort_reset(df)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    close = work["close"].astype(float)
    sh_mask, sl_mask = _fractal_swing_mask_simple(high, low, swing_left, swing_right)

    work["is_swing_high"] = sh_mask.astype(np.int8)
    work["is_swing_low"] = sl_mask.astype(np.int8)
    work["swing_is_high"] = work["is_swing_high"]
    work["swing_is_low"] = work["is_swing_low"]

    n = len(work)
    last_h = np.full(n, np.nan)
    last_l = np.full(n, np.nan)
    last_sh_price = np.full(n, np.nan)
    bh, bl = -1, -1
    cur_sh = np.nan
    for i in range(n):
        if sh_mask[i]:
            bh = i
            cur_sh = float(high.iloc[i])
        if sl_mask[i]:
            bl = i
        last_h[i] = i - bh if bh >= 0 else np.nan
        last_l[i] = i - bl if bl >= 0 else np.nan
        last_sh_price[i] = cur_sh

    work["bars_since_swing_high"] = last_h
    work["bars_since_swing_low"] = last_l
    work["swing_bars_since_high"] = last_h
    work["swing_bars_since_low"] = last_l

    with np.errstate(divide="ignore", invalid="ignore"):
        # % distance from close up to last swing high (when swing high is above close)
        shp = last_sh_price
        pct = np.where(np.isfinite(shp) & (shp > close.to_numpy()), (shp - close.to_numpy()) / close.to_numpy() * 100.0, np.nan)
    work["swing_high_distance_pct"] = pct

    return work


def fvg_detection(df: pd.DataFrame, *, atr_period: int = 14) -> pd.DataFrame:
    """
    Fair Value Gaps + ML-oriented columns:

    - ``fvg_bullish`` / ``fvg_bearish`` (formation bar)
    - ``fvg_size`` (absolute gap), ``fvg_size_normalized`` (÷ ATR)
    - ``bars_since_fvg`` since last formation (any side)
    - ``is_price_in_fvg`` if ``close`` lies in the **active** most-recent unfilled gap zone
    """
    _require_ohlcv(df)
    work = _sort_reset(df)
    h = work["high"].astype(float)
    l = work["low"].astype(float)
    c = work["close"].astype(float)
    atr = _atr_wilder(h, l, c, atr_period)
    h2 = h.shift(2)
    l2 = l.shift(2)

    bull = (l > h2).fillna(False).to_numpy()
    bear = (h < l2).fillna(False).to_numpy()

    n = len(work)
    hn = h.to_numpy()
    ln = l.to_numpy()
    cn = c.to_numpy()
    atrn = atr.to_numpy()

    work["fvg_bullish"] = bull.astype(np.int8)
    work["fvg_bearish"] = bear.astype(np.int8)

    bull_low = np.where(bull, h2.to_numpy(), np.nan)
    bull_high = np.where(bull, l.to_numpy(), np.nan)
    work["fvg_bull_low"] = bull_low
    work["fvg_bull_high"] = bull_high
    abs_bull = np.where(bull, ln - h2.to_numpy(), np.nan)
    work["fvg_bull_size"] = abs_bull

    bear_high = np.where(bear, l2.to_numpy(), np.nan)
    bear_low = np.where(bear, h.to_numpy(), np.nan)
    work["fvg_bear_high"] = bear_high
    work["fvg_bear_low"] = bear_low
    abs_bear = np.where(bear, l2.to_numpy() - hn, np.nan)
    work["fvg_bear_size"] = abs_bear

    fvg_size = np.full(n, np.nan)
    fvg_size_norm = np.full(n, np.nan)
    for i in range(n):
        if bull[i]:
            s = abs_bull[i]
            fvg_size[i] = s
            fvg_size_norm[i] = s / atrn[i] if atrn[i] and atrn[i] > 0 else np.nan
        elif bear[i]:
            s = abs_bear[i]
            fvg_size[i] = s
            fvg_size_norm[i] = s / atrn[i] if atrn[i] and atrn[i] > 0 else np.nan

    work["fvg_size"] = fvg_size
    work["fvg_size_normalized"] = fvg_size_norm

    bars_since = np.full(n, np.nan)
    last_fvg = -1
    b_lo, b_hi = np.nan, np.nan
    r_lo, r_hi = np.nan, np.nan

    in_fvg = np.zeros(n, dtype=np.int8)
    for i in range(2, n):
        # Invalidate filled zones (simplified mitigation)
        if np.isfinite(b_lo) and ln[i] <= b_lo:
            b_lo, b_hi = np.nan, np.nan
        if np.isfinite(r_lo) and np.isfinite(r_hi) and hn[i] >= r_hi:
            r_lo, r_hi = np.nan, np.nan

        if bull[i]:
            b_lo, b_hi = float(hn[i - 2]), float(ln[i])
            last_fvg = i
        elif bear[i]:
            r_lo, r_hi = float(hn[i]), float(ln[i - 2])
            last_fvg = i

        if last_fvg >= 0:
            bars_since[i] = float(i - last_fvg)

        inside_b = np.isfinite(b_lo) and b_lo <= cn[i] <= b_hi
        inside_r = np.isfinite(r_lo) and r_lo <= cn[i] <= r_hi
        in_fvg[i] = 1 if (inside_b or inside_r) else 0

    work["bars_since_fvg"] = bars_since
    work["is_price_in_fvg"] = in_fvg

    return work
