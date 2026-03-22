"""
Swing trading strategy: H4 regime + H1 Fibonacci OTE zone + M15 candlestick trigger.

Expects an aligned multi-timeframe DataFrame from ``BacktestDataLoader.load_aligned``:
``open_time``, ``15m_*``, ``1h_*``, ``4h_*`` OHLCV columns.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from backtest.configs.default_config import DEFAULT_BACKTEST_CONFIG
from backtest.models import Signal, SignalSide


def _col(df: pd.DataFrame, prefix: str, name: str) -> pd.Series:
    c = f"{prefix}_{name}"
    if c not in df.columns:
        raise ValueError(f"Missing column {c!r}; pass aligned MTF data from BacktestDataLoader.")
    return df[c]


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_c).abs(),
            (low - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    multiplier: float,
) -> tuple[pd.Series, pd.Series]:
    """
    ATR-based Supertrend. Returns (supertrend_line, direction) with direction +1 bull, -1 bear.
    """
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    n = len(close)
    final_upper = np.zeros(n, dtype=np.float64)
    final_lower = np.zeros(n, dtype=np.float64)
    st = np.zeros(n, dtype=np.float64)
    direction = np.ones(n, dtype=np.int8)

    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    c = close.to_numpy(dtype=np.float64)
    ub = upper_basic.to_numpy(dtype=np.float64)
    lb = lower_basic.to_numpy(dtype=np.float64)

    for i in range(n):
        if i == 0 or np.isnan(atr.iloc[i]):
            final_upper[i] = ub[i] if not np.isnan(ub[i]) else c[i]
            final_lower[i] = lb[i] if not np.isnan(lb[i]) else c[i]
            st[i] = final_lower[i]
            direction[i] = 1
            continue

        if np.isnan(ub[i]):
            final_upper[i] = final_upper[i - 1]
            final_lower[i] = final_lower[i - 1]
            st[i] = st[i - 1]
            direction[i] = direction[i - 1]
            continue

        if ub[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = ub[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if lb[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = lb[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if direction[i - 1] == -1:
            if c[i] <= final_upper[i]:
                st[i] = final_upper[i]
                direction[i] = -1
            else:
                st[i] = final_lower[i]
                direction[i] = 1
        else:
            if c[i] >= final_lower[i]:
                st[i] = final_lower[i]
                direction[i] = 1
            else:
                st[i] = final_upper[i]
                direction[i] = -1

    idx = close.index
    return pd.Series(st, index=idx), pd.Series(direction, index=idx)


def bullish_engulfing(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_bear = c.shift(1) < o.shift(1)
    curr_bull = c > o
    body_engulfs = (c > o.shift(1)) & (o < c.shift(1))
    return (prev_bear & curr_bull & body_engulfs).fillna(False)


def bearish_engulfing(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_bull = c.shift(1) > o.shift(1)
    curr_bear = c < o
    body_engulfs = (c < o.shift(1)) & (o > c.shift(1))
    return (prev_bull & curr_bear & body_engulfs).fillna(False)


def hammer(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    small_body = body <= 0.35 * rng
    long_lower = lower_wick >= 2.0 * body.replace(0, np.nan)
    short_upper = upper_wick <= body
    return (small_body & long_lower & short_upper & rng.notna()).fillna(False)


def shooting_star(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    small_body = body <= 0.35 * rng
    long_upper = upper_wick >= 2.0 * body.replace(0, np.nan)
    short_lower = lower_wick <= body
    return (small_body & long_upper & short_lower & rng.notna()).fillna(False)


def bullish_pin_bar(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    return ((lower_wick >= 2.0 * body) & (upper_wick <= 0.6 * body) & (rng > 0)).fillna(False)


def bearish_pin_bar(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    return ((upper_wick >= 2.0 * body) & (lower_wick <= 0.6 * body) & (rng > 0)).fillna(False)


def fib_ote_zones(
    swing_high: pd.Series,
    swing_low: pd.Series,
    close: pd.Series,
    fib_lo: float = 0.618,
    fib_hi: float = 0.786,
) -> tuple[pd.Series, pd.Series]:
    """
    Fibonacci OTE (optimal trade entry) bands on a swing range.

    Long OTE: retrace down from ``swing_high`` into [78.6%, 61.8%] of the range (deeper .. shallower).
    Short OTE: retrace up from ``swing_low`` into [61.8%, 78.6%] of the range.
    """
    rng = (swing_high - swing_low).replace(0, np.nan)
    # Pullback buy zone (bullish bias): between high - 0.786*R and high - 0.618*R
    long_bound_deep = swing_high - fib_hi * rng
    long_bound_shallow = swing_high - fib_lo * rng
    in_long_ote = (close <= long_bound_shallow) & (close >= long_bound_deep)

    short_bound_shallow = swing_low + fib_lo * rng
    short_bound_deep = swing_low + fib_hi * rng
    in_short_ote = (close >= short_bound_shallow) & (close <= short_bound_deep)

    return in_long_ote.fillna(False), in_short_ote.fillna(False)


class SwingTradingStrategy:
    """
    H4: EMA200 + Supertrend regime.
    H1: Fibonacci OTE zone vs swing range.
    M15: Engulfing / Pin bar / Hammer (plus shooting star for shorts).
    Risk: SL 2%, TP 6%, optional time stop 48h (ms) for execution layer.
    """

    TF_EXEC = "15m"
    TF_ZONE = "1h"
    TF_TREND = "4h"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = {**DEFAULT_BACKTEST_CONFIG, **(dict(config) if config else {})}
        self.stop_loss_pct: float = float(cfg["stop_loss_pct"])
        self.take_profit_pct: float = float(cfg["take_profit_pct"])
        self.time_stop_hours: int = int(cfg["time_stop_hours"])
        self.ema_period: int = int(cfg["ema_period"])
        self.rsi_period: int = int(cfg["rsi_period"])
        self.supertrend_period: int = int(cfg["supertrend_period"])
        self.supertrend_multiplier: float = float(cfg["supertrend_multiplier"])
        self.h1_swing_bars: int = int(cfg["h1_swing_bars"])
        self.m15_rsi_long_max: float = float(cfg["m15_rsi_long_max"])
        self.m15_rsi_short_min: float = float(cfg["m15_rsi_short_min"])

    def time_stop_ms(self) -> int:
        return self.time_stop_hours * 60 * 60 * 1000

    def risk_bracket(self, side: SignalSide, entry_price: float) -> dict[str, float]:
        """Absolute SL/TP prices from entry (symmetric %% move)."""
        if side == SignalSide.LONG:
            return {
                "stop_loss": entry_price * (1.0 - self.stop_loss_pct),
                "take_profit": entry_price * (1.0 + self.take_profit_pct),
            }
        if side == SignalSide.SHORT:
            return {
                "stop_loss": entry_price * (1.0 + self.stop_loss_pct),
                "take_profit": entry_price * (1.0 - self.take_profit_pct),
            }
        return {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator / regime / pattern columns (in-place safe: returns copy)."""
        out = df.copy()

        p4 = self.TF_TREND
        p1 = self.TF_ZONE
        pm = self.TF_EXEC

        h4_c = _col(out, p4, "close")
        h4_h = _col(out, p4, "high")
        h4_l = _col(out, p4, "low")
        h1_c = _col(out, p1, "close")
        h1_h = _col(out, p1, "high")
        h1_l = _col(out, p1, "low")
        m_o = _col(out, pm, "open")
        m_h = _col(out, pm, "high")
        m_l = _col(out, pm, "low")
        m_c = _col(out, pm, "close")

        out[f"{p4}_ema{self.ema_period}"] = ema(h4_c, self.ema_period)
        st_line, st_dir = supertrend(
            h4_h, h4_l, h4_c, self.supertrend_period, self.supertrend_multiplier
        )
        out[f"{p4}_supertrend"] = st_line
        out[f"{p4}_supertrend_dir"] = st_dir

        swing_high = h1_h.rolling(self.h1_swing_bars, min_periods=self.h1_swing_bars // 2).max()
        swing_low = h1_l.rolling(self.h1_swing_bars, min_periods=self.h1_swing_bars // 2).min()
        in_long_ote, in_short_ote = fib_ote_zones(swing_high, swing_low, h1_c)
        out[f"{p1}_in_long_ote"] = in_long_ote
        out[f"{p1}_in_short_ote"] = in_short_ote

        out[f"{pm}_rsi"] = rsi(m_c, self.rsi_period)

        out[f"{pm}_bull_engulfing"] = bullish_engulfing(m_o, m_h, m_l, m_c)
        out[f"{pm}_bear_engulfing"] = bearish_engulfing(m_o, m_h, m_l, m_c)
        out[f"{pm}_hammer"] = hammer(m_o, m_h, m_l, m_c)
        out[f"{pm}_shooting_star"] = shooting_star(m_o, m_h, m_l, m_c)
        out[f"{pm}_bull_pin"] = bullish_pin_bar(m_o, m_h, m_l, m_c)
        out[f"{pm}_bear_pin"] = bearish_pin_bar(m_o, m_h, m_l, m_c)

        ema4 = out[f"{p4}_ema{self.ema_period}"]
        h4_bull = (h4_c > ema4) & (st_dir == 1)
        h4_bear = (h4_c < ema4) & (st_dir == -1)
        out[f"{p4}_regime_long"] = h4_bull
        out[f"{p4}_regime_short"] = h4_bear

        m_long_pat = (
            out[f"{pm}_bull_engulfing"] | out[f"{pm}_hammer"] | out[f"{pm}_bull_pin"]
        )
        m_short_pat = (
            out[f"{pm}_bear_engulfing"] | out[f"{pm}_shooting_star"] | out[f"{pm}_bear_pin"]
        )

        out["strat_long_setup"] = (
            h4_bull & in_long_ote & m_long_pat & (out[f"{pm}_rsi"] <= self.m15_rsi_long_max)
        )
        out["strat_short_setup"] = (
            h4_bear & in_short_ote & m_short_pat & (out[f"{pm}_rsi"] >= self.m15_rsi_short_min)
        )

        return out

    def signal_at(self, prepared: pd.DataFrame, index: int) -> Signal | None:
        """Return entry signal at row ``index`` or None."""
        row = prepared.iloc[index]
        t = int(row["open_time"])
        price = float(row[f"{self.TF_EXEC}_close"])

        if bool(row.get("strat_long_setup", False)):
            rb = self.risk_bracket(SignalSide.LONG, price)
            return Signal(
                time=t,
                side=SignalSide.LONG,
                price=price,
                meta={
                    "stop_loss": rb["stop_loss"],
                    "take_profit": rb["take_profit"],
                    "time_stop_ms": self.time_stop_ms(),
                    "stop_loss_pct": self.stop_loss_pct,
                    "take_profit_pct": self.take_profit_pct,
                },
            )
        if bool(row.get("strat_short_setup", False)):
            rb = self.risk_bracket(SignalSide.SHORT, price)
            return Signal(
                time=t,
                side=SignalSide.SHORT,
                price=price,
                meta={
                    "stop_loss": rb["stop_loss"],
                    "take_profit": rb["take_profit"],
                    "time_stop_ms": self.time_stop_ms(),
                    "stop_loss_pct": self.stop_loss_pct,
                    "take_profit_pct": self.take_profit_pct,
                },
            )
        return None

    def signals_dataframe(self, prepared: pd.DataFrame) -> pd.DataFrame:
        """Attach a ``signal_side`` column (string or NaN) for inspection / research."""
        out = prepared.copy()
        sides: list[str | float] = []
        for i in range(len(out)):
            sig = self.signal_at(out, i)
            sides.append(sig.side.value if sig else np.nan)
        out["signal_side"] = sides
        return out
