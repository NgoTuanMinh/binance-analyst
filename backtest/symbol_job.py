"""
Single-symbol backtest job (importable for multiprocessing workers).

``mp_run_symbol_pack`` / ``mp_auto_tune_pack`` must live in a real package module
so ``ProcessPoolExecutor`` can pickle them on Windows/macOS spawn.
"""

from __future__ import annotations

import gc
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from backtest.data_loader import BacktestDataLoader
from backtest.engine import BacktestEngine, _OpenPosition, auto_tune, save_run_summary, trades_to_dataframe
from backtest.models import BacktestResult
from backtest.portfolio_engine import (
    _finalize_metrics,
    iter_year_chunk_bounds,
    run_portfolio_backtest,
    run_portfolio_simulation,
)
from backtest.strategies.swing_strategy import SwingTradingStrategy


def run_one_symbol_job(
    symbol: str,
    args: Namespace,
    strategy_cfg: dict[str, Any],
    engine_cfg: dict[str, Any],
) -> tuple[int, dict]:
    base_row: dict = {"symbol": symbol}
    loader = BacktestDataLoader(merged_dir=args.merged_dir)
    try:
        aligned = loader.load_aligned(
            symbol,
            ["M15", "H1", "H4"],
            start_ms=args.start_ms,
            end_ms=args.end_ms,
        )
    except FileNotFoundError as e:
        print(f"[{symbol}] Error: {e}", file=sys.stderr)
        return 1, {**base_row, "status": "missing_parquet", "error": str(e)}

    if aligned.empty:
        print(f"[{symbol}] No rows after load/filter.", file=sys.stderr)
        return 1, {**base_row, "status": "empty_range", "error": "no rows after filter"}

    strategy = SwingTradingStrategy(config=strategy_cfg)
    prepared = strategy.prepare(aligned)

    engine = BacktestEngine(
        initial_capital=engine_cfg.get("initial_capital"),
        commission_rate=engine_cfg.get("commission_rate"),
        slippage_bps=engine_cfg.get("slippage_bps"),
        show_progress=not args.no_progress,
    )
    result = engine.run(symbol, prepared, strategy)
    m = result.metrics

    print(f"\n=== {symbol} ===", flush=True)
    print(f"Bars: {len(prepared)}  Trades: {m.get('n_trades', 0)}", flush=True)
    print(
        f"Win rate: {m.get('win_rate', 0) * 100:.2f}%  "
        f"Profit factor: {m.get('profit_factor')}  "
        f"Avg return/win %: {m.get('avg_return_win_pct', 0):.3f}  "
        f"Avg return/loss %: {m.get('avg_return_loss_pct', 0):.3f}",
        flush=True,
    )
    print(
        f"Final equity: {m.get('final_equity', 0):.2f}  "
        f"Return: {m.get('total_return_pct', 0):.2f}%  "
        f"Total PnL: {m.get('total_pnl', 0):.2f}",
        flush=True,
    )
    print(f"Wins / Losses: {m.get('winning_trades', 0)} / {m.get('losing_trades', 0)}", flush=True)

    if args.export_trades_csv:
        p = Path(args.export_trades_csv)
        if getattr(args, "_multi_symbol_run", False):
            p = p.parent / f"{p.stem}_{symbol}{p.suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        trades_to_dataframe(result.trades).to_csv(p, index=False)
        print(f"Wrote trades CSV: {p}", flush=True)

    if args.export_summary:
        p = Path(args.export_summary)
        if getattr(args, "_multi_symbol_run", False):
            p = p.parent / f"{p.stem}_{symbol}{p.suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        save_run_summary(p, m, strategy_cfg, symbol=symbol)
        print(f"Wrote run summary JSON: {p}", flush=True)

    ok_row = {
        **base_row,
        "status": "ok",
        "bars": len(prepared),
        "n_trades": m.get("n_trades"),
        "win_rate": m.get("win_rate"),
        "profit_factor": m.get("profit_factor"),
        "total_return_pct": m.get("total_return_pct"),
        "total_pnl": m.get("total_pnl"),
        "final_equity": m.get("final_equity"),
        "winning_trades": m.get("winning_trades"),
        "losing_trades": m.get("losing_trades"),
        "avg_return_win_pct": m.get("avg_return_win_pct"),
        "avg_return_loss_pct": m.get("avg_return_loss_pct"),
        "error": "",
    }
    return 0, ok_row


def run_portfolio_job(
    symbols: list[str],
    args: Namespace,
    strategy_cfg: dict[str, Any],
    engine_cfg: dict[str, Any],
    max_open_symbols: int,
    position_size_pct: float,
    *,
    portfolio_yearly: bool = True,
    portfolio_warmup_days: int = 120,
) -> tuple[int, dict[str, Any]]:
    """Shared-cash backtest. By default splits by calendar year to cap RAM (requires --start/--end)."""
    if portfolio_yearly and args.start_ms is not None and args.end_ms is not None:
        return run_portfolio_job_yearly(
            symbols,
            args,
            strategy_cfg,
            engine_cfg,
            max_open_symbols,
            position_size_pct,
            portfolio_warmup_days=portfolio_warmup_days,
        )
    return run_portfolio_job_single(
        symbols,
        args,
        strategy_cfg,
        engine_cfg,
        max_open_symbols,
        position_size_pct,
    )


def run_portfolio_job_single(
    symbols: list[str],
    args: Namespace,
    strategy_cfg: dict[str, Any],
    engine_cfg: dict[str, Any],
    max_open_symbols: int,
    position_size_pct: float,
) -> tuple[int, dict[str, Any]]:
    """One pass: load full [start_ms, end_ms] for all symbols (high RAM on long ranges)."""
    base_row: dict[str, Any] = {"symbol": "PORTFOLIO"}
    loader = BacktestDataLoader(merged_dir=args.merged_dir)
    prepared_by_symbol: dict[str, Any] = {}
    ok_symbols: list[str] = []
    for sym in symbols:
        try:
            aligned = loader.load_aligned(
                sym,
                ["M15", "H1", "H4"],
                start_ms=args.start_ms,
                end_ms=args.end_ms,
            )
        except FileNotFoundError as e:
            print(f"[{sym}] skip: {e}", file=sys.stderr)
            continue
        if aligned.empty:
            print(f"[{sym}] skip: empty range", file=sys.stderr)
            continue
        strat = SwingTradingStrategy(config=strategy_cfg)
        prepared_by_symbol[sym] = strat.prepare(aligned)
        ok_symbols.append(sym)

    if not ok_symbols:
        return 1, {**base_row, "status": "no_data", "error": "no symbols with merged Parquet in range"}

    engine = BacktestEngine(
        initial_capital=engine_cfg.get("initial_capital"),
        commission_rate=engine_cfg.get("commission_rate"),
        slippage_bps=engine_cfg.get("slippage_bps"),
        show_progress=not args.no_progress,
    )
    result = run_portfolio_backtest(
        ok_symbols,
        prepared_by_symbol,
        strategy_cfg,
        engine,
        max_open_symbols=max_open_symbols,
        position_size_pct=position_size_pct,
    )
    return _portfolio_job_finish(ok_symbols, result, args, strategy_cfg, base_row, max_open_symbols, position_size_pct)


def run_portfolio_job_yearly(
    symbols: list[str],
    args: Namespace,
    strategy_cfg: dict[str, Any],
    engine_cfg: dict[str, Any],
    max_open_symbols: int,
    position_size_pct: float,
    *,
    portfolio_warmup_days: int,
) -> tuple[int, dict[str, Any]]:
    """Year-by-year: load → prepare → simulate → carry cash/positions; drop prepared between years."""
    base_row: dict[str, Any] = {"symbol": "PORTFOLIO"}
    g0, g1 = int(args.start_ms), int(args.end_ms)
    chunks = iter_year_chunk_bounds(g0, g1, portfolio_warmup_days)
    if not chunks:
        return 1, {**base_row, "status": "no_data", "error": "empty year range"}

    loader = BacktestDataLoader(merged_dir=args.merged_dir)
    first = chunks[0]
    ok_symbols: list[str] = []
    for sym in symbols:
        try:
            aligned = loader.load_aligned(
                sym,
                ["M15", "H1", "H4"],
                start_ms=first["load_start_ms"],
                end_ms=first["load_end_ms"],
            )
        except FileNotFoundError as e:
            print(f"[{sym}] skip: {e}", file=sys.stderr)
            continue
        if aligned.empty:
            print(f"[{sym}] skip: empty range (first year chunk)", file=sys.stderr)
            continue
        ok_symbols.append(sym)

    if not ok_symbols:
        return 1, {**base_row, "status": "no_data", "error": "no symbols with merged Parquet in first chunk"}

    global_ic = float(engine_cfg.get("initial_capital", 10_000.0))
    comm = engine_cfg.get("commission_rate")
    slip = engine_cfg.get("slippage_bps")

    all_trades: list = []
    all_equity: list[tuple[int, float]] = []
    carry_cash = global_ic
    carry_pos: dict[str, _OpenPosition] | None = None
    last_sim_ts: int | None = None
    last_interval = "15m"

    for ci, chunk in enumerate(chunks):
        y = chunk["year"]
        print(f"\n=== PORTFOLIO year {y} ({ci + 1}/{len(chunks)}) load [{chunk['load_start_ms']} .. {chunk['load_end_ms']}] ===", flush=True)
        prepared_by_symbol: dict[str, Any] = {}
        for sym in ok_symbols:
            try:
                aligned = loader.load_aligned(
                    sym,
                    ["M15", "H1", "H4"],
                    start_ms=chunk["load_start_ms"],
                    end_ms=chunk["load_end_ms"],
                )
            except FileNotFoundError as e:
                print(f"[{y}][{sym}] skip: {e}", file=sys.stderr)
                continue
            if aligned.empty:
                continue
            strat = SwingTradingStrategy(config=strategy_cfg)
            prepared_by_symbol[sym] = strat.prepare(aligned)
            del aligned
        if not prepared_by_symbol:
            print(f"[{y}] No prepared data for any symbol; carrying state forward.", file=sys.stderr)
            last_sim_ts = chunk["load_end_ms"]
            continue

        if carry_pos:
            for sym in carry_pos:
                if sym not in prepared_by_symbol:
                    print(
                        f"[{y}] ERROR: open position on {sym} but no merged data in this chunk; "
                        "check Parquet / range.",
                        file=sys.stderr,
                    )
                    return 1, {
                        **base_row,
                        "status": "carry_missing_symbol",
                        "error": f"year {y}: carried position on {sym} without data in chunk",
                    }

        use_syms = [s for s in ok_symbols if s in prepared_by_symbol]
        engine = BacktestEngine(
            initial_capital=carry_cash,
            commission_rate=comm,
            slippage_bps=slip,
            show_progress=not args.no_progress,
        )
        is_last = ci == len(chunks) - 1
        r, carry = run_portfolio_simulation(
            use_syms,
            prepared_by_symbol,
            strategy_cfg,
            engine,
            max_open_symbols=max_open_symbols,
            position_size_pct=position_size_pct,
            initial_cash=carry_cash,
            initial_positions=carry_pos,
            entry_signal_start_ms=chunk["entry_start_ms"],
            entry_signal_end_ms=chunk["entry_end_ms"],
            sim_time_after_ms=last_sim_ts,
            close_all_at_end=is_last,
            metrics_initial_capital=global_ic,
        )
        last_interval = r.interval
        all_trades.extend(r.trades)
        all_equity.extend(r.equity_curve)
        carry_cash = carry.cash
        carry_pos = carry.positions if carry.positions else None

        if r.equity_curve:
            last_sim_ts = r.equity_curve[-1][0]
        else:
            mx = 0
            for df in prepared_by_symbol.values():
                if not df.empty and "open_time" in df.columns:
                    mx = max(mx, int(df["open_time"].max()))
            last_sim_ts = mx if mx else chunk["load_end_ms"]

        del prepared_by_symbol
        gc.collect()

    engine_report = BacktestEngine(
        initial_capital=global_ic,
        commission_rate=comm,
        slippage_bps=slip,
        show_progress=False,
    )
    metrics = _finalize_metrics(engine_report, all_trades, all_equity, initial_capital_override=global_ic)
    metrics["portfolio_max_open_symbols"] = max_open_symbols
    metrics["portfolio_position_size_pct"] = position_size_pct
    metrics["portfolio_symbols"] = list(ok_symbols)
    metrics["portfolio_yearly"] = True
    metrics["portfolio_yearly_chunks"] = len(chunks)
    metrics["portfolio_yearly_warmup_days"] = portfolio_warmup_days

    result = BacktestResult(
        symbol="PORTFOLIO",
        interval=last_interval,
        trades=all_trades,
        equity_curve=all_equity,
        metrics=metrics,
    )
    return _portfolio_job_finish(ok_symbols, result, args, strategy_cfg, base_row, max_open_symbols, position_size_pct)


def _portfolio_job_finish(
    ok_symbols: list[str],
    result: BacktestResult,
    args: Namespace,
    strategy_cfg: dict[str, Any],
    base_row: dict[str, Any],
    max_open_symbols: int,
    position_size_pct: float,
) -> tuple[int, dict[str, Any]]:
    m = result.metrics
    sym_list = ",".join(ok_symbols)
    print(f"\n=== PORTFOLIO ({sym_list}) ===", flush=True)
    print(f"Bars (equity points): {len(result.equity_curve)}  Trades: {m.get('n_trades', 0)}", flush=True)
    print(
        f"Win rate: {m.get('win_rate', 0) * 100:.2f}%  "
        f"Profit factor: {m.get('profit_factor')}  "
        f"Max open symbols: {max_open_symbols}  Pos size % equity: {position_size_pct * 100:.1f}%",
        flush=True,
    )
    print(
        f"Final equity: {m.get('final_equity', 0):.2f}  "
        f"Return: {m.get('total_return_pct', 0):.2f}%  "
        f"Total PnL: {m.get('total_pnl', 0):.2f}",
        flush=True,
    )
    print(f"Wins / Losses: {m.get('winning_trades', 0)} / {m.get('losing_trades', 0)}", flush=True)

    if args.export_trades_csv:
        p = Path(args.export_trades_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        trades_to_dataframe(result.trades).to_csv(p, index=False)
        print(f"Wrote trades CSV: {p}", flush=True)

    if args.export_summary:
        p = Path(args.export_summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        extra: dict[str, Any] = {"portfolio_symbols": ok_symbols}
        if m.get("portfolio_yearly"):
            extra["portfolio_yearly_chunks"] = m.get("portfolio_yearly_chunks")
            extra["portfolio_yearly_warmup_days"] = m.get("portfolio_yearly_warmup_days")
        save_run_summary(p, m, strategy_cfg, symbol="PORTFOLIO", extra=extra)
        print(f"Wrote run summary JSON: {p}", flush=True)

    ok_row = {
        **base_row,
        "status": "ok",
        "portfolio_symbols": sym_list,
        "bars_timeline": len(result.equity_curve),
        "n_trades": m.get("n_trades"),
        "win_rate": m.get("win_rate"),
        "profit_factor": m.get("profit_factor"),
        "total_return_pct": m.get("total_return_pct"),
        "total_pnl": m.get("total_pnl"),
        "final_equity": m.get("final_equity"),
        "winning_trades": m.get("winning_trades"),
        "losing_trades": m.get("losing_trades"),
        "avg_return_win_pct": m.get("avg_return_win_pct"),
        "avg_return_loss_pct": m.get("avg_return_loss_pct"),
        "error": "",
    }
    return 0, ok_row


def mp_run_symbol_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Process pool worker for batch backtest."""
    rr = Path(pack["repo_root"])
    rs = str(rr.resolve())
    if rs not in sys.path:
        sys.path.insert(0, rs)
    ns = Namespace(**pack["ns"])
    code, row = run_one_symbol_job(pack["symbol"], ns, pack["strategy_cfg"], pack["engine_cfg"])
    return {"symbol": pack["symbol"], "code": code, "row": row}


def mp_auto_tune_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Process pool worker for auto_tune(symbol, ...)."""
    rr = Path(pack["repo_root"])
    rs = str(rr.resolve())
    if rs not in sys.path:
        sys.path.insert(0, rs)

    sym = pack["symbol"]
    sym_out = Path(pack["sym_out"]) if pack.get("sym_out") else None
    merged = Path(pack["merged_dir"]) if pack.get("merged_dir") is not None else None

    try:
        report = auto_tune(
            sym,
            merged_dir=merged,
            start_ms=pack.get("start_ms"),
            end_ms=pack.get("end_ms"),
            initial_config=pack["initial_config"],
            target_win_rate=float(pack["target_win_rate"]),
            max_iterations=int(pack["max_iterations"]),
            show_progress=False,
            output_dir=sym_out,
        )
    except FileNotFoundError as e:
        report = {"error": "missing_parquet", "message": str(e)}
        print(f"[{sym}] auto_tune: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        report = {"error": "exception", "message": str(e)}
        print(f"[{sym}] auto_tune failed: {e}", file=sys.stderr, flush=True)

    if "symbol" not in report:
        report = {**report, "symbol": sym}
    fr = report.get("final_win_rate", "n/a")
    sr = report.get("stopped_reason", "n/a")
    print(f"\n>>> auto_tune {sym}  final_win_rate={fr}  stopped={sr}", flush=True)
    return report
