"""
Smart Money Concepts (SMC) features — OHLCV + optional ``15m_atr_14`` from :class:`TechnicalFeatures`.

Liquidity (daily/weekly), order-block proxies, market structure / BOS / CHoCH, Fibonacci / OTE.
Legacy ``smc_*`` columns are kept for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .price_action import _fractal_swing_mask_simple
from .technical import _atr_wilder, _require_ohlcv, _sort_reset


def _open_time_to_datetime(ot: pd.Series) -> pd.Series:
    """Parse ``open_time`` (ms int or datetime) to UTC-aware datetime."""
    s = pd.to_datetime(ot, unit="ms", utc=True, errors="coerce")
    if s.isna().all():
        s = pd.to_datetime(ot, utc=True, errors="coerce")
    return s


def _resolve_atr(df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    if "15m_atr_14" in df.columns:
        return df["15m_atr_14"].astype(float)
    return _atr_wilder(high.astype(float), low.astype(float), close.astype(float), 14)


def smart_money_features(
    df: pd.DataFrame,
    *,
    liquidity_lookback: int = 20,
    swing_left: int = 2,
    swing_right: int = 2,
    displacement_body_ratio: float = 0.65,
    ob_impulse_body_ratio: float = 0.55,
    premium_lookback: int = 50,
    ob_lookback: int = 40,
) -> pd.DataFrame:
    """
    Add SMC-inspired columns on the **base** timeframe bars of ``df``.

    **Liquidity**

    - ``daily_high`` / ``daily_low``: running session extrema (UTC calendar day).
    - ``weekly_high`` / ``weekly_low``: running extrema (ISO week Mon–Sun).
    - ``distance_to_daily_high`` / ``distance_to_daily_low``: ``(extremum - close)`` / ATR
      (high: positive when close below daily high; low: positive when close above daily low).
    - ``is_liquidity_sweep``: wick beyond **previous** day or **previous** week high/low, close back inside.

    **Order blocks** (heuristic: last opposing candle before displacement + break of prior bar extreme)

    - ``last_bullish_ob_distance`` / ``last_bearish_ob_distance``: distance to last OB zone / ATR (0 if inside zone).
    - ``ob_strength``: volume of OB candle / rolling volume SMA(20).

    **Structure**

    - ``market_structure``: ``1`` uptrend (HH+HL), ``-1`` downtrend (LH+LL), ``0`` ranging.
    - ``break_of_structure_bull`` / ``break_of_structure_bear``: BOS (close vs last swing high / low).
    - ``choch_bullish`` / ``choch_bearish``: CHoCH — after downtrend, close breaks last swing **high**;
      after uptrend, close breaks last swing **low**.

    **Fibonacci** (retracement from last swing high toward swing low, leg = H - L)

    - ``fib_0_382_distance`` … ``fib_0_786_distance``: (close - level) / ATR.
    - ``is_in_ote_zone``: ICT-style OTE: (H - close) / (H - L) ∈ [0.618, 0.786].

    **Legacy** (unchanged semantics): ``smc_liq_grab_*``, ``smc_displacement``, ``smc_premium_discount``,
    ``smc_last_swing_*``, ``smc_bos_up`` / ``smc_bos_down``.
    """
    _require_ohlcv(df)
    work = _sort_reset(df)
    h = work["high"].astype(float)
    l_ = work["low"].astype(float)
    c = work["close"].astype(float)
    o = work["open"].astype(float)
    v = work["volume"].astype(float)
    n = len(work)
    rng = (h - l_).replace(0, np.nan)

    atr = _resolve_atr(work, h, l_, c)
    atr_np = atr.to_numpy()
    safe_atr = np.where(np.isfinite(atr_np) & (atr_np > 0), atr_np, np.nan)

    # --- Calendar day / week keys (UTC) ---
    ts = _open_time_to_datetime(work["open_time"])
    work["_cal_day"] = ts.dt.normalize()
    # ISO week (Mon-based), no Period conversion (avoids tz drop warning)
    work["_cal_week"] = ts.dt.strftime("%G-W%V")

    work["daily_high"] = work.groupby("_cal_day", sort=False)["high"].cummax().to_numpy()
    work["daily_low"] = work.groupby("_cal_day", sort=False)["low"].cummin().to_numpy()
    work["weekly_high"] = work.groupby("_cal_week", sort=False)["high"].cummax().to_numpy()
    work["weekly_low"] = work.groupby("_cal_week", sort=False)["low"].cummin().to_numpy()

    day_agg = work.groupby("_cal_day", as_index=False).agg(_day_h=("high", "max"), _day_l=("low", "min"))
    day_agg["prev_day_high"] = day_agg["_day_h"].shift(1)
    day_agg["prev_day_low"] = day_agg["_day_l"].shift(1)
    work = work.merge(
        day_agg[["_cal_day", "prev_day_high", "prev_day_low"]],
        on="_cal_day",
        how="left",
    )

    week_agg = work.groupby("_cal_week", as_index=False).agg(_wk_h=("high", "max"), _wk_l=("low", "min"))
    week_agg["prev_week_high"] = week_agg["_wk_h"].shift(1)
    week_agg["prev_week_low"] = week_agg["_wk_l"].shift(1)
    work = work.merge(
        week_agg[["_cal_week", "prev_week_high", "prev_week_low"]],
        on="_cal_week",
        how="left",
    )
    work.drop(columns=["_cal_day", "_cal_week"], inplace=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        work["distance_to_daily_high"] = (work["daily_high"].to_numpy() - c.to_numpy()) / safe_atr
        work["distance_to_daily_low"] = (c.to_numpy() - work["daily_low"].to_numpy()) / safe_atr

    ph = work["prev_day_high"].to_numpy()
    pl = work["prev_day_low"].to_numpy()
    pwh = work["prev_week_high"].to_numpy()
    pwl = work["prev_week_low"].to_numpy()
    hn = h.to_numpy()
    ln = l_.to_numpy()
    cn = c.to_numpy()

    sweep_day = (np.isfinite(ph) & (hn > ph) & (cn < ph)) | (np.isfinite(pl) & (ln < pl) & (cn > pl))
    sweep_wk = (np.isfinite(pwh) & (hn > pwh) & (cn < pwh)) | (np.isfinite(pwl) & (ln < pwl) & (cn > pwl))
    work["is_liquidity_sweep"] = (sweep_day | sweep_wk).astype(np.int8)

    work.drop(
        columns=[
            c
            for c in (
                "prev_day_high",
                "prev_day_low",
                "prev_week_high",
                "prev_week_low",
            )
            if c in work.columns
        ],
        inplace=True,
    )

    # --- Order blocks (stateful) ---
    vol_sma = v.rolling(20, min_periods=5).mean().to_numpy()
    ho, lo, oo, co, vo = hn, ln, o.to_numpy(), cn, v.to_numpy()

    bull_lo = np.full(n, np.nan)
    bull_hi = np.full(n, np.nan)
    bear_lo = np.full(n, np.nan)
    bear_hi = np.full(n, np.nan)
    ob_strength_arr = np.full(n, np.nan)

    cur_bl, cur_bh = np.nan, np.nan
    cur_brl, cur_brh = np.nan, np.nan
    cur_ob_str = np.nan
    lb = int(ob_lookback)

    for i in range(1, n):
        ri = ho[i] - lo[i]
        body = abs(co[i] - oo[i])
        brat = body / ri if ri > 0 else 0.0
        bull_imp = (co[i] > oo[i]) and (brat >= ob_impulse_body_ratio) and (ho[i] >= ho[i - 1])
        bear_imp = (co[i] < oo[i]) and (brat >= ob_impulse_body_ratio) and (lo[i] <= lo[i - 1])

        if bull_imp:
            j = i - 1
            jmin = max(0, i - lb)
            found = False
            while j >= jmin:
                if co[j] < oo[j]:
                    cur_bl, cur_bh = float(lo[j]), float(ho[j])
                    vs = vol_sma[j]
                    cur_ob_str = float(vo[j] / vs) if vs and np.isfinite(vs) and vs > 0 else 1.0
                    found = True
                    break
                j -= 1
            if not found:
                pass

        if bear_imp:
            j = i - 1
            jmin = max(0, i - lb)
            while j >= jmin:
                if co[j] > oo[j]:
                    cur_brl, cur_brh = float(lo[j]), float(ho[j])
                    vs = vol_sma[j]
                    cur_ob_str = float(vo[j] / vs) if vs and np.isfinite(vs) and vs > 0 else 1.0
                    break
                j -= 1

        bull_lo[i], bull_hi[i] = cur_bl, cur_bh
        bear_lo[i], bear_hi[i] = cur_brl, cur_brh
        ob_strength_arr[i] = cur_ob_str

    def _zone_dist(price: float, zlo: float, zhi: float) -> float:
        if not (np.isfinite(price) and np.isfinite(zlo) and np.isfinite(zhi) and zhi >= zlo):
            return np.nan
        if zlo <= price <= zhi:
            return 0.0
        return float(min(abs(price - zlo), abs(price - zhi)))

    dist_bull = np.full(n, np.nan)
    dist_bear = np.full(n, np.nan)
    for i in range(n):
        dist_bull[i] = _zone_dist(co[i], bull_lo[i], bull_hi[i])
        dist_bear[i] = _zone_dist(co[i], bear_lo[i], bear_hi[i])

    with np.errstate(divide="ignore", invalid="ignore"):
        work["last_bullish_ob_distance"] = dist_bull / safe_atr
        work["last_bearish_ob_distance"] = dist_bear / safe_atr
    work["ob_strength"] = ob_strength_arr

    # --- Swings: structure, BOS, CHoCH, Fib ---
    sh_mask, sl_mask = _fractal_swing_mask_simple(h, l_, swing_left, swing_right)
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    cur_h, cur_l = np.nan, np.nan
    # Track last two swing highs / lows for HH-HL / LH-LL
    sh_hist: list[float] = []
    sl_hist: list[float] = []
    ms = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if sh_mask[i]:
            cur_h = float(hn[i])
            sh_hist.append(cur_h)
            if len(sh_hist) > 2:
                sh_hist.pop(0)
        if sl_mask[i]:
            cur_l = float(ln[i])
            sl_hist.append(cur_l)
            if len(sl_hist) > 2:
                sl_hist.pop(0)
        last_sh[i] = cur_h
        last_sl[i] = cur_l

        trend = 0
        if len(sh_hist) >= 2 and len(sl_hist) >= 2:
            hh = sh_hist[-1] > sh_hist[-2]
            hl = sl_hist[-1] > sl_hist[-2]
            lh = sh_hist[-1] < sh_hist[-2]
            ll_ = sl_hist[-1] < sl_hist[-2]
            if hh and hl:
                trend = 1
            elif lh and ll_:
                trend = -1
        ms[i] = trend

    work["market_structure"] = ms

    last_sh_s = pd.Series(last_sh, index=work.index)
    last_sl_s = pd.Series(last_sl, index=work.index)
    prev_sh = last_sh_s.shift(1)
    prev_sl = last_sl_s.shift(1)

    bos_bull = (c > prev_sh) & prev_sh.notna()
    bos_bear = (c < prev_sl) & prev_sl.notna()
    work["break_of_structure_bull"] = bos_bull.fillna(False).astype(np.int8)
    work["break_of_structure_bear"] = bos_bear.fillna(False).astype(np.int8)

    ms_s = pd.Series(ms, index=work.index)
    prev_ms = ms_s.shift(1).fillna(0).astype(int)
    choch_bull = (prev_ms == -1) & bos_bull
    choch_bear = (prev_ms == 1) & bos_bear
    work["choch_bullish"] = choch_bull.fillna(False).astype(np.int8)
    work["choch_bearish"] = choch_bear.fillna(False).astype(np.int8)

    # Fib levels from last swing low / high (same running refs)
    fib382 = np.full(n, np.nan)
    fib5 = np.full(n, np.nan)
    fib618 = np.full(n, np.nan)
    fib786 = np.full(n, np.nan)
    ote = np.zeros(n, dtype=np.int8)

    for i in range(n):
        lo_p, hi_p = last_sl[i], last_sh[i]
        if not (np.isfinite(lo_p) and np.isfinite(hi_p) and hi_p > lo_p):
            continue
        leg = hi_p - lo_p
        f382 = hi_p - 0.382 * leg
        f5 = hi_p - 0.5 * leg
        f618 = hi_p - 0.618 * leg
        f786 = hi_p - 0.786 * leg
        fib382[i] = f382
        fib5[i] = f5
        fib618[i] = f618
        fib786[i] = f786
        retr = (hi_p - co[i]) / leg
        if 0.618 <= retr <= 0.786:
            ote[i] = 1

    with np.errstate(divide="ignore", invalid="ignore"):
        work["fib_0_382_distance"] = (cn - fib382) / safe_atr
        work["fib_0_5_distance"] = (cn - fib5) / safe_atr
        work["fib_0_618_distance"] = (cn - fib618) / safe_atr
        work["fib_0_786_distance"] = (cn - fib786) / safe_atr
    work["is_in_ote_zone"] = ote

    # --- Legacy smc_* (rolling liquidity grab, displacement, premium, BOS) ---
    lb_ = int(liquidity_lookback)
    roll_high = h.shift(1).rolling(lb_, min_periods=3).max()
    roll_low = l_.shift(1).rolling(lb_, min_periods=3).min()
    work["smc_liq_grab_high"] = ((h > roll_high) & (c < roll_high)).fillna(False).astype(np.int8)
    work["smc_liq_grab_low"] = ((l_ < roll_low) & (c > roll_low)).fillna(False).astype(np.int8)

    body_ratio = (c - o).abs() / rng
    work["smc_displacement"] = (body_ratio >= displacement_body_ratio).astype(np.int8)

    hh = h.rolling(premium_lookback, min_periods=5).max()
    ll_r = l_.rolling(premium_lookback, min_periods=5).min()
    span = (hh - ll_r).replace(0, np.nan)
    work["smc_premium_discount"] = ((c - ll_r) / span).clip(0.0, 1.0)

    work["smc_last_swing_high"] = last_sh
    work["smc_last_swing_low"] = last_sl
    work["smc_bos_up"] = work["break_of_structure_bull"]
    work["smc_bos_down"] = work["break_of_structure_bear"]

    return work
