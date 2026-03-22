#!/usr/bin/env python3
"""
Swing backtest CLI (repo root on sys.path).

Examples::

    python scripts/run_backtest.py --symbols BTCUSDT --start 2023-01-01 --end 2023-06-01 \\
        --export-trades-csv out/trades.csv --export-summary out/baseline.json

    python scripts/run_backtest.py --auto-tune --symbols BTCUSDT --start 2023-01-01 --end 2023-06-01 \\
        --tune-output-dir out/tune --no-progress

    python scripts/run_backtest.py --auto-tune --symbols-json crypto-data-pipeline/config/top_300_symbols.json \\
        --continue-on-error --no-progress --max-tune-iterations 3 \\
        --tune-output-dir out/tune_mtf --tune-batch-json out/tune_all.json

    python scripts/run_backtest.py --symbols-json crypto-data-pipeline/config/top_300_symbols.json \\
        --continue-on-error --no-progress --batch-summary-csv out/batch_summary.csv

    python scripts/run_backtest.py --symbols BTCUSDT,ETHUSDT --workers 4 --no-progress \\
        --batch-summary-csv out/batch_summary.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.configs.default_config import DEFAULT_BACKTEST_CONFIG
from backtest.engine import (
    analyze_trades,
    auto_tune,
    compare_runs,
    read_trades_csv,
)
from backtest.symbol_job import mp_auto_tune_pack, mp_run_symbol_pack, run_one_symbol_job


def _day_start_utc_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _day_end_utc_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = dt.replace(hour=23, minute=59, second=59, microsecond=999_000)
    return int(end.timestamp() * 1000)


def _parse_symbols(s: str) -> list[str]:
    return [x.strip().upper() for x in s.replace(";", ",").split(",") if x.strip()]


def _resolve_config_path(p: Path) -> Path:
    """Path relative to CWD, else relative to repo root."""
    if p.is_file():
        return p.resolve()
    cand = _ROOT / p
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"File not found: {p} (tried cwd and repo root {_ROOT})")


def _load_symbols_json(path: Path) -> list[str]:
    """
    Load symbol list from JSON: either ``[\"BTCUSDT\", ...]`` or ``{\"symbols\": [...]}``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        out = [str(x).strip().upper() for x in raw if str(x).strip()]
    elif isinstance(raw, dict):
        out = []
        for key in ("symbols", "SYMBOLS", "top_symbols"):
            if key in raw and isinstance(raw[key], list):
                out = [str(x).strip().upper() for x in raw[key] if str(x).strip()]
                break
        if not out:
            raise ValueError(f"No symbol list found in {path}; expected array or object with 'symbols' key.")
    else:
        raise ValueError(f"Unsupported JSON root type in {path}: {type(raw)}")
    # de-dupe, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _sanitize_summary_row(row: dict) -> dict:
    out: dict = {}
    for k, v in row.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def _finite_float(x: object) -> bool:
    try:
        f = float(x)  # type: ignore[arg-type]
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _compute_batch_aggregates(summary_rows: list[dict]) -> dict:
    """Counts + mean/median of metrics over rows with ``status == \"ok\"``."""
    total = len(summary_rows)
    by_status: dict[str, int] = {}
    for r in summary_rows:
        st = str(r.get("status", "unknown"))
        by_status[st] = by_status.get(st, 0) + 1

    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    n_ok = len(ok_rows)

    def col(name: str) -> list[float]:
        out: list[float] = []
        for r in ok_rows:
            v = r.get(name)
            if _finite_float(v):
                out.append(float(v))
        return out

    wr = col("win_rate")
    retp = col("total_return_pct")
    pnl = col("total_pnl")
    pf = col("profit_factor")
    nt = col("n_trades")
    aw = col("avg_return_win_pct")
    al = col("avg_return_loss_pct")

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    def median(xs: list[float]) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        m = len(s) // 2
        return (s[m] + s[m - 1]) / 2 if len(s) % 2 == 0 else s[m]

    return {
        "n_symbols_in_batch": total,
        "n_ok": n_ok,
        "n_missing_parquet": by_status.get("missing_parquet", 0),
        "n_empty_range": by_status.get("empty_range", 0),
        "n_exception": by_status.get("exception", 0),
        "status_counts": by_status,
        "mean_win_rate": mean(wr),
        "median_win_rate": median(wr),
        "mean_total_return_pct": mean(retp),
        "median_total_return_pct": median(retp),
        "mean_total_pnl": mean(pnl),
        "sum_total_pnl": sum(pnl) if pnl else None,
        "mean_profit_factor": mean(pf),
        "median_profit_factor": median(pf),
        "mean_n_trades": mean(nt),
        "sum_n_trades": sum(int(x) for x in nt) if nt else None,
        "mean_avg_return_win_pct": mean(aw),
        "mean_avg_return_loss_pct": mean(al),
    }


def _print_batch_aggregates(agg: dict) -> None:
    print(f"\n--- Trung bình / gộp (chỉ symbol status=ok, n_ok={agg['n_ok']}/{agg['n_symbols_in_batch']}) ---")
    print(f"Mean win_rate:        {agg.get('mean_win_rate')}")
    print(f"Median win_rate:      {agg.get('median_win_rate')}")
    print(f"Mean total_return %:  {agg.get('mean_total_return_pct')}")
    print(f"Median total_return %: {agg.get('median_total_return_pct')}")
    print(f"Mean profit_factor:   {agg.get('mean_profit_factor')}")
    print(f"Median profit_factor: {agg.get('median_profit_factor')}")
    print(f"Mean n_trades:        {agg.get('mean_n_trades')}  |  Sum trades: {agg.get('sum_n_trades')}")
    print(f"Mean total_pnl:       {agg.get('mean_total_pnl')}  |  Sum total_pnl: {agg.get('sum_total_pnl')}")
    print(f"Mean avg_return/win %:  {agg.get('mean_avg_return_win_pct')}")
    print(f"Mean avg_return/loss %: {agg.get('mean_avg_return_loss_pct')}")
    print(f"Thiếu Parquet: {agg.get('n_missing_parquet')}  |  empty_range: {agg.get('n_empty_range')}  |  exception: {agg.get('n_exception')}")


def _save_batch_aggregates_json(agg: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
    print(f"Wrote aggregates JSON: {json_path}")


def _resolve_workers(cli_value: int | None) -> int:
    """
    Effective worker count when CLI omits ``--workers``: env ``BACKTEST_WORKERS``,
    else ``parallel_workers`` in ``DEFAULT_BACKTEST_CONFIG``.
    """
    if cli_value is not None:
        return int(cli_value)
    env = os.environ.get("BACKTEST_WORKERS", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            print(f"Warning: invalid BACKTEST_WORKERS={env!r}; using config default.", file=sys.stderr)
    return int(DEFAULT_BACKTEST_CONFIG.get("parallel_workers", 1))


def _effective_workers(requested: int, n_tasks: int) -> int:
    """``requested`` <= 0 → ``os.cpu_count()``; cap by ``n_tasks``."""
    if n_tasks <= 0:
        return 1
    if requested <= 0:
        w = os.cpu_count() or 1
    else:
        w = requested
    return max(1, min(w, n_tasks))


def _worker_ns_dict(args: argparse.Namespace) -> dict:
    """Namespace fields passed to child processes (no tqdm in workers)."""
    return {
        "merged_dir": args.merged_dir,
        "start_ms": args.start_ms,
        "end_ms": args.end_ms,
        "no_progress": True,
        "export_trades_csv": args.export_trades_csv,
        "export_summary": args.export_summary,
        "_multi_symbol_run": getattr(args, "_multi_symbol_run", False),
    }


def _print_compare(df: pd.DataFrame) -> None:
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))


def main() -> int:
    default_merged = _ROOT / "crypto-data-pipeline" / "data" / "merged"

    p = argparse.ArgumentParser(description="Run swing backtest (merged Parquet).")
    p.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT",
        help="Comma-separated symbols (default: BTCUSDT); ignored if --symbols-json is set",
    )
    p.add_argument("--symbol", type=str, default=None, help="Alias for single symbol")
    p.add_argument(
        "--symbols-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON file: [\"SYMBOL\", ...] or {\"symbols\": [...]} (e.g. crypto-data-pipeline/config/top_300_symbols.json)",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="With multiple symbols: log failures and keep going (exit 1 if any failed)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel processes for multi-symbol backtest or auto_tune. "
        "Omit to use BACKTEST_WORKERS env, else parallel_workers in backtest/configs/default_config.py. "
        "0 = all logical CPUs (os.cpu_count). Each worker loads full symbol data — lower N if RAM is tight.",
    )
    p.add_argument(
        "--batch-summary-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write one row per symbol (metrics + status) after a multi-symbol run",
    )
    p.add_argument(
        "--batch-aggregate-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write mean/median aggregates JSON; default: next to --batch-summary-csv if set",
    )
    p.add_argument(
        "--merged-dir",
        type=Path,
        default=default_merged,
        help=f"Merged Parquet root (default: {default_merged})",
    )
    p.add_argument("--start", type=str, default=None, help="Start UTC YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End UTC YYYY-MM-DD")
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--export-trades-csv", type=Path, default=None)
    p.add_argument("--export-summary", type=Path, default=None)
    p.add_argument(
        "--analyze-trades-csv",
        type=Path,
        default=None,
        help="Load CSV and print analyze_trades() JSON (no backtest)",
    )
    p.add_argument("--compare-run1", type=Path, default=None, metavar="PATH")
    p.add_argument("--compare-run2", type=Path, default=None, metavar="PATH")
    p.add_argument(
        "--auto-tune",
        action="store_true",
        help="Per-symbol tuning loop; multi-symbol runs in parallel if --workers > 1",
    )
    p.add_argument("--target-win-rate", type=float, default=0.5)
    p.add_argument("--max-tune-iterations", type=int, default=10)
    p.add_argument(
        "--tune-output-dir",
        type=Path,
        default=None,
        help="Save tune_iter_XX.json here; multi-symbol → subfolder per symbol (e.g. OUT/BTCUSDT/)",
    )
    p.add_argument(
        "--tune-batch-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write tune report to JSON (one object if single symbol, else {\"results\": [...]})",
    )
    args = p.parse_args()
    args.workers = _resolve_workers(args.workers)

    if args.compare_run1 and args.compare_run2:
        df = compare_runs(args.compare_run1, args.compare_run2)
        _print_compare(df)
        return 0

    if args.analyze_trades_csv:
        df = read_trades_csv(args.analyze_trades_csv)
        a = analyze_trades(df)
        print(json.dumps(a, indent=2, default=str))
        return 0

    if args.symbols_json is not None:
        sj = _resolve_config_path(args.symbols_json)
        symbols = _load_symbols_json(sj)
    else:
        sym_arg = args.symbol if args.symbol else args.symbols
        symbols = _parse_symbols(sym_arg)

    start_ms = _day_start_utc_ms(args.start) if args.start else None
    end_ms = _day_end_utc_ms(args.end) if args.end else None
    args.start_ms = start_ms
    args.end_ms = end_ms

    base = DEFAULT_BACKTEST_CONFIG.copy()
    if args.capital is not None:
        base["initial_capital"] = args.capital

    strategy_keys = (
        "stop_loss_pct",
        "take_profit_pct",
        "time_stop_hours",
        "ema_period",
        "rsi_period",
        "supertrend_period",
        "supertrend_multiplier",
        "h1_swing_bars",
        "m15_rsi_long_max",
        "m15_rsi_short_min",
    )
    strategy_cfg = {k: base[k] for k in strategy_keys}
    engine_cfg = {
        "initial_capital": base["initial_capital"],
        "commission_rate": base["commission_rate"],
        "slippage_bps": base["slippage_bps"],
    }

    args._multi_symbol_run = len(symbols) > 1

    if args.auto_tune:
        n_sym = len(symbols)
        w_tune = _effective_workers(args.workers, n_sym) if n_sym > 1 else 1
        parallel_tune = n_sym > 1 and w_tune > 1
        if parallel_tune and not args.continue_on_error:
            print(
                "Note: --workers>1 runs auto_tune for all symbols in parallel; "
                "use --workers 1 to stop early on first failure.",
                file=sys.stderr,
            )

        reports: list[dict] = []
        rc_tune = 0
        repo_root = str(_ROOT.resolve())

        if parallel_tune:
            packs: list[dict] = []
            for sym in symbols:
                if args.tune_output_dir is not None:
                    out_dir = Path(args.tune_output_dir)
                    sym_out = out_dir / sym if n_sym > 1 else out_dir
                else:
                    sym_out = None
                packs.append(
                    {
                        "repo_root": repo_root,
                        "symbol": sym,
                        "merged_dir": args.merged_dir,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "initial_config": base,
                        "target_win_rate": args.target_win_rate,
                        "max_iterations": args.max_tune_iterations,
                        "sym_out": str(sym_out.resolve()) if sym_out is not None else None,
                    }
                )
            print(f"auto_tune: {n_sym} symbols, {w_tune} workers (ProcessPoolExecutor)", flush=True)
            with ProcessPoolExecutor(max_workers=w_tune) as pool:
                reports = list(pool.map(mp_auto_tune_pack, packs))
            for r in reports:
                if r.get("error"):
                    rc_tune = 1
        else:
            for sym in symbols:
                if args.tune_output_dir is not None:
                    out_dir = Path(args.tune_output_dir)
                    sym_out = out_dir / sym if n_sym > 1 else out_dir
                else:
                    sym_out = None
                try:
                    report = auto_tune(
                        sym,
                        merged_dir=args.merged_dir,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        initial_config=base,
                        target_win_rate=args.target_win_rate,
                        max_iterations=args.max_tune_iterations,
                        show_progress=not args.no_progress,
                        output_dir=sym_out,
                    )
                except FileNotFoundError as e:
                    report = {"error": "missing_parquet", "message": str(e)}
                    rc_tune = 1
                    print(f"[{sym}] auto_tune: {e}", file=sys.stderr)
                except Exception as e:
                    report = {"error": "exception", "message": str(e)}
                    rc_tune = 1
                    print(f"[{sym}] auto_tune failed: {e}", file=sys.stderr)

                if "symbol" not in report:
                    report = {**report, "symbol": sym}
                reports.append(report)

                if report.get("error"):
                    rc_tune = 1

                if n_sym > 1:
                    fr = report.get("final_win_rate", "n/a")
                    sr = report.get("stopped_reason", "n/a")
                    print(f"\n>>> auto_tune {sym}  final_win_rate={fr}  stopped={sr}")

                if rc_tune and n_sym > 1 and not args.continue_on_error:
                    break

        if len(symbols) == 1:
            print(json.dumps(reports[0], indent=2, default=str))
        else:
            print(json.dumps({"results": reports}, indent=2, default=str))

        if args.tune_batch_json is not None:
            tp = Path(args.tune_batch_json)
            tp.parent.mkdir(parents=True, exist_ok=True)
            payload = reports[0] if len(symbols) == 1 else {"results": reports}
            tp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            print(f"Wrote tune report JSON: {tp}")

        return rc_tune

    summary_rows: list[dict] = []
    rc = 0
    n_sym = len(symbols)
    repo_root = str(_ROOT.resolve())
    w_back = _effective_workers(args.workers, n_sym) if n_sym > 1 else 1
    parallel_back = n_sym > 1 and w_back > 1

    if parallel_back and not args.continue_on_error:
        print(
            "Note: --workers>1 runs all symbols in parallel; use --workers 1 for stop-on-first-error.",
            file=sys.stderr,
        )

    if parallel_back:
        packs = [
            {
                "repo_root": repo_root,
                "symbol": sym,
                "strategy_cfg": strategy_cfg,
                "engine_cfg": engine_cfg,
                "ns": _worker_ns_dict(args),
            }
            for sym in symbols
        ]
        print(f"Backtest: {n_sym} symbols, {w_back} workers (ProcessPoolExecutor)", flush=True)
        with ProcessPoolExecutor(max_workers=w_back) as pool:
            results = list(pool.map(mp_run_symbol_pack, packs))
        for r in results:
            row = _sanitize_summary_row(r["row"])
            summary_rows.append(row)
            if r["code"] != 0:
                rc = 1
    else:
        for sym in symbols:
            try:
                code, row = run_one_symbol_job(sym, args, strategy_cfg, engine_cfg)
            except Exception as e:
                code, row = 1, {"symbol": sym, "status": "exception", "error": str(e)}
                print(f"[{sym}] Unhandled error: {e}", file=sys.stderr)
            if n_sym > 1:
                summary_rows.append(_sanitize_summary_row(row))
            if code != 0:
                rc = 1
                if n_sym > 1 and not args.continue_on_error:
                    break

    if args.batch_summary_csv and summary_rows:
        out_p = Path(args.batch_summary_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(out_p, index=False)
        print(f"\nWrote batch summary: {out_p} ({len(summary_rows)} rows)")

    if len(symbols) > 1 and summary_rows:
        agg = _compute_batch_aggregates(summary_rows)
        _print_batch_aggregates(agg)
        if args.batch_aggregate_json is not None:
            agg_path: Path | None = Path(args.batch_aggregate_json)
        elif args.batch_summary_csv is not None:
            pcsv = Path(args.batch_summary_csv)
            agg_path = pcsv.with_name(f"{pcsv.stem}_aggregates.json")
        else:
            agg_path = None
        if agg_path is not None:
            _save_batch_aggregates_json(agg, agg_path)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
