"""
Single-symbol backtest job (importable for multiprocessing workers).

``mp_run_symbol_pack`` / ``mp_auto_tune_pack`` must live in a real package module
so ``ProcessPoolExecutor`` can pickle them on Windows/macOS spawn.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from backtest.data_loader import BacktestDataLoader
from backtest.engine import BacktestEngine, auto_tune, save_run_summary, trades_to_dataframe
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
