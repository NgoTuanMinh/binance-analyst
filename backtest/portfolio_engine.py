"""Multi-symbol portfolio backtest: shared cash, cap concurrent symbols, size each leg by equity %%."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pandas as pd
from tqdm import tqdm

from backtest.engine import BacktestEngine, _OpenPosition
from backtest.models import BacktestResult, Signal, SignalSide, Trade
from backtest.strategies.swing_strategy import SwingTradingStrategy


def _ms_utc(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def iter_year_chunk_bounds(
    global_start_ms: int,
    global_end_ms: int,
    warmup_days: int,
) -> list[dict[str, int]]:
    """
    Split [global_start_ms, global_end_ms] into calendar-year chunks.

    Each chunk has:
      - load_*: Parquet range (includes warmup before year entry for indicators)
      - entry_*: only bars in this range may open new positions (warmup excluded)
    """
    warmup_ms = int(warmup_days * 24 * 3600 * 1000)
    start_dt = datetime.fromtimestamp(global_start_ms / 1000.0, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(global_end_ms / 1000.0, tz=timezone.utc)
    y0 = start_dt.year
    y1 = end_dt.year
    chunks: list[dict[str, int]] = []
    for y in range(y0, y1 + 1):
        year_start = datetime(y, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        year_end = datetime(y, 12, 31, 23, 59, 59, 999_000, tzinfo=timezone.utc)
        entry_start_ms = max(global_start_ms, _ms_utc(year_start))
        entry_end_ms = min(global_end_ms, _ms_utc(year_end))
        if entry_start_ms > entry_end_ms:
            continue
        load_start_ms = max(global_start_ms, entry_start_ms - warmup_ms)
        load_end_ms = min(global_end_ms, entry_end_ms)
        chunks.append(
            {
                "year": y,
                "load_start_ms": load_start_ms,
                "load_end_ms": load_end_ms,
                "entry_start_ms": entry_start_ms,
                "entry_end_ms": entry_end_ms,
            }
        )
    return chunks


@dataclass
class PortfolioCarryover:
    """Cash and open positions passed to the next year chunk."""

    cash: float
    positions: dict[str, _OpenPosition]


def _asof_close_row(df: pd.DataFrame, t_ms: int) -> pd.Series | None:
    sub = df[df["open_time"] <= t_ms]
    if sub.empty:
        return None
    return sub.iloc[-1]


def _mark_map(
    prepared_by_symbol: Mapping[str, pd.DataFrame],
    positions: Mapping[str, _OpenPosition],
    t_ms: int,
    prefix: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym in positions:
        r = _asof_close_row(prepared_by_symbol[sym], t_ms)
        if r is not None:
            out[sym] = float(r[f"{prefix}_close"])
    return out


def _portfolio_equity(cash: float, positions: Mapping[str, _OpenPosition], marks: Mapping[str, float]) -> float:
    e = cash
    for sym, pos in positions.items():
        px = marks.get(sym)
        if px is None:
            continue
        if pos.side == SignalSide.LONG:
            e += pos.size * px
        else:
            e -= pos.size * px
    return e


def _open_long_budget(
    eng: BacktestEngine,
    cash: float,
    budget: float,
    sig: Signal,
    close: float,
) -> tuple[_OpenPosition, float] | None:
    if sig.price is None or cash <= 0:
        return None
    fill = eng._slip_buy(close)
    if fill <= 0:
        return None
    spend = min(max(0.0, cash), max(0.0, budget))
    cost_per_unit = fill * (1.0 + eng.commission_rate)
    size = spend / cost_per_unit
    if size <= 0:
        return None
    cash_out = cash - size * cost_per_unit
    meta = sig.meta
    sl = float(meta.get("stop_loss", fill * (1.0 - 0.02)))
    tp = float(meta.get("take_profit", fill * (1.0 + 0.06)))
    ts_ms = meta.get("time_stop_ms")
    time_stop_at = int(sig.time + int(ts_ms)) if ts_ms is not None else None
    pos = _OpenPosition(
        side=SignalSide.LONG,
        entry_time=sig.time,
        entry_price=fill,
        size=size,
        stop_loss=sl,
        take_profit=tp,
        time_stop_at=time_stop_at,
    )
    return pos, cash_out


def _open_short_budget(
    eng: BacktestEngine,
    cash: float,
    budget: float,
    sig: Signal,
    close: float,
) -> tuple[_OpenPosition, float] | None:
    if sig.price is None or cash <= 0:
        return None
    fill = eng._slip_sell(close)
    if fill <= 0:
        return None
    cap = min(max(0.0, cash), max(0.0, budget))
    denom = fill * (1.0 + eng.commission_rate)
    size = cap / denom
    if size <= 0:
        return None
    proceeds = size * fill * (1.0 - eng.commission_rate)
    cash_out = cash + proceeds
    meta = sig.meta
    sl = float(meta.get("stop_loss", fill * (1.0 + 0.02)))
    tp = float(meta.get("take_profit", fill * (1.0 - 0.06)))
    ts_ms = meta.get("time_stop_ms")
    time_stop_at = int(sig.time + int(ts_ms)) if ts_ms is not None else None
    pos = _OpenPosition(
        side=SignalSide.SHORT,
        entry_time=sig.time,
        entry_price=fill,
        size=size,
        stop_loss=sl,
        take_profit=tp,
        time_stop_at=time_stop_at,
    )
    return pos, cash_out


def _finalize_metrics(
    eng: BacktestEngine,
    trades: list[Trade],
    equity_curve: list[tuple[int, float]],
    *,
    initial_capital_override: float | None = None,
) -> dict[str, Any]:
    ic = float(initial_capital_override if initial_capital_override is not None else eng.initial_capital)
    final_eq = equity_curve[-1][1] if equity_curve else ic
    wins = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl is not None and t.pnl <= 0)
    total_pnl = sum(t.pnl or 0.0 for t in trades)
    n = len(trades)
    win_rate = wins / n if n else 0.0
    win_pnls = [t.pnl for t in trades if t.pnl is not None and t.pnl > 0]
    loss_pnls = [t.pnl for t in trades if t.pnl is not None and t.pnl <= 0]
    gross_profit = sum(win_pnls) if win_pnls else 0.0
    gross_loss_abs = abs(sum(loss_pnls)) if loss_pnls else 0.0
    if gross_loss_abs > 1e-12:
        profit_factor = gross_profit / gross_loss_abs
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    avg_trade_pnl = total_pnl / n if n else 0.0
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    win_pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None and t.pnl and t.pnl > 0]
    loss_pnl_pcts = [
        t.pnl_pct for t in trades if t.pnl_pct is not None and t.pnl is not None and t.pnl <= 0
    ]
    avg_return_win_pct = sum(win_pnl_pcts) / len(win_pnl_pcts) if win_pnl_pcts else 0.0
    avg_return_loss_pct = sum(loss_pnl_pcts) / len(loss_pnl_pcts) if loss_pnl_pcts else 0.0
    return {
        "initial_capital": ic,
        "final_equity": final_eq,
        "total_return_pct": (final_eq / ic - 1.0) * 100.0 if ic else 0.0,
        "n_trades": n,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "avg_trade_pnl": avg_trade_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_return_win_pct": avg_return_win_pct,
        "avg_return_loss_pct": avg_return_loss_pct,
        "gross_profit": gross_profit,
        "gross_loss_abs": gross_loss_abs,
        "commission_rate": eng.commission_rate,
        "slippage_bps": eng.slippage_bps,
    }


def run_portfolio_simulation(
    symbols: list[str],
    prepared_by_symbol: Mapping[str, pd.DataFrame],
    strategy_cfg: Mapping[str, Any] | None,
    engine: BacktestEngine,
    *,
    max_open_symbols: int = 5,
    position_size_pct: float = 0.2,
    initial_cash: float | None = None,
    initial_positions: Mapping[str, _OpenPosition] | None = None,
    entry_signal_start_ms: int | None = None,
    entry_signal_end_ms: int | None = None,
    sim_time_after_ms: int | None = None,
    close_all_at_end: bool = True,
    metrics_initial_capital: float | None = None,
) -> tuple[BacktestResult, PortfolioCarryover]:
    """
    Core portfolio walk. Use ``sim_time_after_ms`` to avoid re-processing bars already
    simulated in a previous chunk (yearly mode). New entries only inside
    [entry_signal_start_ms, entry_signal_end_ms] when those are set.
    """
    symbols = [s for s in symbols if s in prepared_by_symbol]
    empty = BacktestResult(
        symbol="PORTFOLIO",
        interval="15m",
        trades=[],
        equity_curve=[],
        metrics={"error": "no symbols with prepared data"},
    )
    if not symbols:
        return empty, PortfolioCarryover(cash=float(engine.initial_capital), positions={})

    prefix = ""
    strategies: dict[str, SwingTradingStrategy] = {}
    for sym in symbols:
        st = SwingTradingStrategy(config=dict(strategy_cfg) if strategy_cfg else None)
        strategies[sym] = st
        prefix = st.TF_EXEC

    time_index: dict[str, dict[int, int]] = {}
    all_times: set[int] = set()
    for sym in symbols:
        df = prepared_by_symbol[sym]
        if df.empty:
            continue
        ot = df["open_time"].astype("int64")
        idx_map = {int(ot.iloc[i]): i for i in range(len(df))}
        time_index[sym] = idx_map
        all_times.update(idx_map.keys())

    if not all_times:
        cash0 = float(initial_cash) if initial_cash is not None else float(engine.initial_capital)
        pos0 = dict(initial_positions) if initial_positions else {}
        return (
            BacktestResult(
                symbol="PORTFOLIO",
                interval=prefix or "15m",
                trades=[],
                equity_curve=[],
                metrics={"error": "empty prepared frames"},
            ),
            PortfolioCarryover(cash=cash0, positions=pos0),
        )

    times_sorted = sorted(all_times)
    if sim_time_after_ms is not None:
        times_sorted = [t for t in times_sorted if t > int(sim_time_after_ms)]

    cash = float(initial_cash) if initial_cash is not None else float(engine.initial_capital)
    positions: dict[str, _OpenPosition] = dict(initial_positions) if initial_positions else {}
    trades: list[Trade] = []
    equity_curve: list[tuple[int, float]] = []

    max_open = max(1, int(max_open_symbols))
    pos_pct = float(position_size_pct)
    if pos_pct <= 0:
        pos_pct = 0.2

    mic = metrics_initial_capital if metrics_initial_capital is not None else engine.initial_capital

    bar_iter = tqdm(times_sorted, desc="Backtest PORTFOLIO", unit="bar") if engine.show_progress else times_sorted
    for t in bar_iter:
        for sym in list(positions.keys()):
            if sym not in time_index or t not in time_index[sym]:
                continue
            i = time_index[sym][t]
            row = prepared_by_symbol[sym].iloc[i]
            ts = int(row["open_time"])
            h = float(row[f"{prefix}_high"])
            l = float(row[f"{prefix}_low"])
            c = float(row[f"{prefix}_close"])
            pos = positions[sym]
            cash, pos_out = engine._try_exit(sym, pos, ts, h, l, c, trades, cash)
            if pos_out is None:
                del positions[sym]
            else:
                positions[sym] = pos_out

        for sym in sorted(symbols):
            if sym in positions or len(positions) >= max_open:
                continue
            if sym not in time_index or t not in time_index[sym]:
                continue
            if entry_signal_start_ms is not None and t < int(entry_signal_start_ms):
                continue
            if entry_signal_end_ms is not None and t > int(entry_signal_end_ms):
                continue
            i = time_index[sym][t]
            prepared = prepared_by_symbol[sym]
            row = prepared.iloc[i]
            c = float(row[f"{prefix}_close"])
            marks = _mark_map(prepared_by_symbol, positions, t, prefix)
            equity = _portfolio_equity(cash, positions, marks)
            budget = equity * pos_pct
            strat = strategies[sym]
            sig = strat.signal_at(prepared, i)
            if sig is None:
                continue
            opened: tuple[_OpenPosition, float] | None = None
            if sig.side == SignalSide.LONG:
                opened = _open_long_budget(engine, cash, budget, sig, c)
            elif sig.side == SignalSide.SHORT:
                opened = _open_short_budget(engine, cash, budget, sig, c)
            if opened:
                pos_new, cash = opened
                positions[sym] = pos_new

        marks_end = _mark_map(prepared_by_symbol, positions, t, prefix)
        eq = _portfolio_equity(cash, positions, marks_end)
        equity_curve.append((t, eq))

    if close_all_at_end:
        for sym in list(positions.keys()):
            df = prepared_by_symbol[sym]
            last = df.iloc[-1]
            ts = int(last["open_time"])
            c = float(last[f"{prefix}_close"])
            pos = positions[sym]
            if pos.side == SignalSide.LONG:
                cash = engine._close_long(sym, pos, ts, c, "eod", trades, cash)
            else:
                cash = engine._close_short(sym, pos, ts, c, "eod", trades, cash)
            del positions[sym]
        if equity_curve:
            last_ts = equity_curve[-1][0]
            equity_curve[-1] = (last_ts, cash)
        carry = PortfolioCarryover(cash=cash, positions={})
    else:
        carry = PortfolioCarryover(cash=cash, positions=dict(positions))

    metrics = _finalize_metrics(engine, trades, equity_curve, initial_capital_override=mic)
    metrics["portfolio_max_open_symbols"] = max_open
    metrics["portfolio_position_size_pct"] = pos_pct
    metrics["portfolio_symbols"] = list(symbols)

    result = BacktestResult(
        symbol="PORTFOLIO",
        interval=prefix,
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
    )
    return result, carry


def run_portfolio_backtest(
    symbols: list[str],
    prepared_by_symbol: Mapping[str, pd.DataFrame],
    strategy_cfg: Mapping[str, Any] | None,
    engine: BacktestEngine,
    *,
    max_open_symbols: int = 5,
    position_size_pct: float = 0.2,
) -> BacktestResult:
    """
    Single pass over in-memory prepared frames (full date range).
    For large multi-year + many symbols, use yearly job with ``run_portfolio_simulation`` per chunk.
    """
    r, _ = run_portfolio_simulation(
        symbols,
        prepared_by_symbol,
        strategy_cfg,
        engine,
        max_open_symbols=max_open_symbols,
        position_size_pct=position_size_pct,
        close_all_at_end=True,
    )
    return r
