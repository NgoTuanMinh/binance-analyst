"""Rule-based backtest engine (data models and configuration)."""

from backtest.data_loader import BacktestDataLoader
from backtest.engine import (
    BacktestEngine,
    adjust_parameters,
    analyze_trades,
    auto_tune,
    compare_runs,
    merge_parameter_adjustments,
    read_trades_csv,
    save_run_summary,
    suggest_parameter_adjustments,
    trades_to_dataframe,
)
from backtest.models import BacktestResult, Candle, Signal, SignalSide, Trade
from backtest.strategies.swing_strategy import SwingTradingStrategy

__all__ = [
    "BacktestDataLoader",
    "BacktestEngine",
    "BacktestResult",
    "Candle",
    "Signal",
    "SignalSide",
    "SwingTradingStrategy",
    "Trade",
    "adjust_parameters",
    "analyze_trades",
    "auto_tune",
    "compare_runs",
    "merge_parameter_adjustments",
    "read_trades_csv",
    "save_run_summary",
    "suggest_parameter_adjustments",
    "trades_to_dataframe",
]
