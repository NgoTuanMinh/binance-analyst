"""Core dataclasses for candle data, signals, trades, and backtest output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalSide(str, Enum):
    """Direction of a strategy signal."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Candle:
    """Single OHLCV bar (aligned with merged pipeline columns)."""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int | None = None
    symbol: str | None = None
    interval: str | None = None


@dataclass
class Signal:
    """Strategy output at a point in time."""

    time: int
    side: SignalSide
    price: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """One round-trip or open position record."""

    symbol: str
    side: SignalSide
    entry_time: int
    entry_price: float
    size: float
    exit_time: int | None = None
    exit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Aggregated outcome of a backtest run."""

    symbol: str
    interval: str | None = None
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
