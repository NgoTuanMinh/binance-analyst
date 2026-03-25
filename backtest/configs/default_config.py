"""Default parameters for the backtest engine and swing-style strategies."""

from __future__ import annotations

from typing import Any

# Timeframes used by multi-timeframe logic (labels match pipeline intervals).
TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h")

# Risk / reward (%% of price). Tuned on BTCUSDT 2024 merged data: win_rate ~54.5%,
# TP in 5–8%% band (take_profit_pct=6.5%%); widen SL + stricter RSI + longer zone swing.
STOP_LOSS_PCT: float = 0.035
TAKE_PROFIT_PCT: float = 0.065
TIME_STOP_HOURS: int = 96

# Execution assumptions
INITIAL_CAPITAL: float = 10_000.0
COMMISSION_RATE: float = 0.0004
SLIPPAGE_BPS: float = 1.0

# Portfolio backtest (``--portfolio``): tối đa N symbol có vị thế; mỗi lệnh ~pct equity.
PORTFOLIO_MAX_OPEN_SYMBOLS: int = 5
PORTFOLIO_POSITION_SIZE_PCT: float = 0.2

# Multi-symbol CLI (scripts/run_backtest.py): process pool size. ``0`` = all logical CPUs.
# Ignored for single-symbol runs. Override per invocation with ``--workers`` or env
# ``BACKTEST_WORKERS`` (only when ``--workers`` is omitted).
PARALLEL_WORKERS: int = 1

# Indicator defaults (swing strategy)
EMA_PERIOD: int = 200
RSI_PERIOD: int = 14
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
H1_SWING_BARS: int = 64
M15_RSI_LONG_MAX: float = 42.0
M15_RSI_SHORT_MIN: float = 60.0

DEFAULT_BACKTEST_CONFIG: dict[str, Any] = {
    "timeframes": TIMEFRAMES,
    "parallel_workers": PARALLEL_WORKERS,
    "stop_loss_pct": STOP_LOSS_PCT,
    "take_profit_pct": TAKE_PROFIT_PCT,
    "time_stop_hours": TIME_STOP_HOURS,
    "initial_capital": INITIAL_CAPITAL,
    "commission_rate": COMMISSION_RATE,
    "slippage_bps": SLIPPAGE_BPS,
    "portfolio_max_open_symbols": PORTFOLIO_MAX_OPEN_SYMBOLS,
    "portfolio_position_size_pct": PORTFOLIO_POSITION_SIZE_PCT,
    "ema_period": EMA_PERIOD,
    "rsi_period": RSI_PERIOD,
    "supertrend_period": SUPERTREND_PERIOD,
    "supertrend_multiplier": SUPERTREND_MULTIPLIER,
    "h1_swing_bars": H1_SWING_BARS,
    "m15_rsi_long_max": M15_RSI_LONG_MAX,
    "m15_rsi_short_min": M15_RSI_SHORT_MIN,
}
