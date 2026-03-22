"""Bar-by-bar backtest engine: execution, risk exits, equity curve, tuning helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from tqdm import tqdm

from backtest.configs.default_config import DEFAULT_BACKTEST_CONFIG
from backtest.models import BacktestResult, Signal, SignalSide, Trade


@runtime_checkable
class StrategySignalSource(Protocol):
    """Strategy that emits at most one actionable signal per bar."""

    TF_EXEC: str

    def signal_at(self, prepared: pd.DataFrame, index: int) -> Signal | None: ...


@dataclass
class _OpenPosition:
    side: SignalSide
    entry_time: int
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    time_stop_at: int | None


class BacktestEngine:
    """
    Walk forward bar-by-bar on a prepared MTF DataFrame.

    - Exits: intrabar SL/TP on execution timeframe OHLC; time stop at bar close.
    - Entries: signal on bar ``i`` fills at that bar's close (with slippage + commission).
    - One position at a time; SL checked before TP on the same bar (conservative for longs).
    """

    def __init__(
        self,
        *,
        initial_capital: float | None = None,
        commission_rate: float | None = None,
        slippage_bps: float | None = None,
        show_progress: bool = True,
    ) -> None:
        cfg = DEFAULT_BACKTEST_CONFIG
        self.initial_capital = float(initial_capital if initial_capital is not None else cfg["initial_capital"])
        self.commission_rate = float(commission_rate if commission_rate is not None else cfg["commission_rate"])
        self.slippage_bps = float(slippage_bps if slippage_bps is not None else cfg["slippage_bps"])
        self.show_progress = show_progress

    def _slip_buy(self, price: float) -> float:
        f = self.slippage_bps / 10_000.0
        return price * (1.0 + f)

    def _slip_sell(self, price: float) -> float:
        f = self.slippage_bps / 10_000.0
        return price * (1.0 - f)

    def _equity(self, cash: float, pos: _OpenPosition | None, mark_close: float) -> float:
        if pos is None:
            return cash
        if pos.side == SignalSide.LONG:
            return cash + pos.size * mark_close
        return cash - pos.size * mark_close

    def _open_long(self, cash: float, sig: Signal, close: float) -> tuple[_OpenPosition, float] | None:
        if sig.price is None:
            return None
        fill = self._slip_buy(close)
        if fill <= 0 or cash <= 0:
            return None
        cost_per_unit = fill * (1.0 + self.commission_rate)
        size = cash / cost_per_unit
        if size <= 0:
            return None
        spent = size * cost_per_unit
        cash_out = cash - spent
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

    def _open_short(self, cash: float, sig: Signal, close: float) -> tuple[_OpenPosition, float] | None:
        if sig.price is None:
            return None
        fill = self._slip_sell(close)
        if fill <= 0 or cash <= 0:
            return None
        notional_cap = cash
        size = notional_cap / (fill * (1.0 + self.commission_rate))
        if size <= 0:
            return None
        proceeds = size * fill * (1.0 - self.commission_rate)
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

    def _close_long(
        self,
        symbol: str,
        pos: _OpenPosition,
        exit_time: int,
        exit_price: float,
        reason: str,
        trades: list[Trade],
        cash: float,
    ) -> float:
        fill = self._slip_sell(exit_price)
        proceeds = pos.size * fill * (1.0 - self.commission_rate)
        cash_in = cash + proceeds
        paid_at_entry = pos.size * pos.entry_price * (1.0 + self.commission_rate)
        pnl = proceeds - paid_at_entry
        notional = pos.size * pos.entry_price
        pnl_pct = (pnl / notional * 100.0) if notional else 0.0
        trades.append(
            Trade(
                symbol=symbol,
                side=pos.side,
                entry_time=pos.entry_time,
                entry_price=pos.entry_price,
                size=pos.size,
                exit_time=exit_time,
                exit_price=fill,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                pnl=pnl,
                pnl_pct=pnl_pct,
                meta={"exit_reason": reason},
            )
        )
        return cash_in

    def _close_short(
        self,
        symbol: str,
        pos: _OpenPosition,
        exit_time: int,
        exit_price: float,
        reason: str,
        trades: list[Trade],
        cash: float,
    ) -> float:
        fill = self._slip_buy(exit_price)
        buy_cost = pos.size * fill * (1.0 + self.commission_rate)
        cash_in = cash - buy_cost
        proceeds_entry = pos.size * pos.entry_price * (1.0 - self.commission_rate)
        pnl = proceeds_entry - buy_cost
        notional = pos.size * pos.entry_price
        pnl_pct = (pnl / notional * 100.0) if notional else 0.0
        trades.append(
            Trade(
                symbol=symbol,
                side=pos.side,
                entry_time=pos.entry_time,
                entry_price=pos.entry_price,
                size=pos.size,
                exit_time=exit_time,
                exit_price=fill,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                pnl=pnl,
                pnl_pct=pnl_pct,
                meta={"exit_reason": reason},
            )
        )
        return cash_in

    def _try_exit(
        self,
        symbol: str,
        pos: _OpenPosition,
        ts: int,
        high: float,
        low: float,
        close: float,
        trades: list[Trade],
        cash: float,
    ) -> tuple[float, _OpenPosition | None]:
        if pos.side == SignalSide.LONG:
            if low <= pos.stop_loss:
                nc = self._close_long(symbol, pos, ts, pos.stop_loss, "sl", trades, cash)
                return nc, None
            if high >= pos.take_profit:
                nc = self._close_long(symbol, pos, ts, pos.take_profit, "tp", trades, cash)
                return nc, None
            if pos.time_stop_at is not None and ts >= pos.time_stop_at:
                nc = self._close_long(symbol, pos, ts, close, "time", trades, cash)
                return nc, None
        else:
            if high >= pos.stop_loss:
                nc = self._close_short(symbol, pos, ts, pos.stop_loss, "sl", trades, cash)
                return nc, None
            if low <= pos.take_profit:
                nc = self._close_short(symbol, pos, ts, pos.take_profit, "tp", trades, cash)
                return nc, None
            if pos.time_stop_at is not None and ts >= pos.time_stop_at:
                nc = self._close_short(symbol, pos, ts, close, "time", trades, cash)
                return nc, None
        return cash, pos

    def run(
        self,
        symbol: str,
        prepared: pd.DataFrame,
        strategy: StrategySignalSource,
    ) -> BacktestResult:
        if prepared.empty:
            return BacktestResult(
                symbol=symbol,
                interval=strategy.TF_EXEC,
                trades=[],
                equity_curve=[],
                metrics={"error": "empty prepared frame"},
            )

        prefix = strategy.TF_EXEC
        cash = self.initial_capital
        pos: _OpenPosition | None = None
        trades: list[Trade] = []
        equity_curve: list[tuple[int, float]] = []

        n = len(prepared)
        iterator = range(n)
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Backtest {symbol}", unit="bar")

        for i in iterator:
            row = prepared.iloc[i]
            ts = int(row["open_time"])
            h = float(row[f"{prefix}_high"])
            l = float(row[f"{prefix}_low"])
            c = float(row[f"{prefix}_close"])

            if pos is not None:
                cash, pos = self._try_exit(symbol, pos, ts, h, l, c, trades, cash)

            if pos is None:
                sig = strategy.signal_at(prepared, i)
                if sig is not None and sig.side == SignalSide.LONG:
                    opened = self._open_long(cash, sig, c)
                    if opened:
                        pos, cash = opened
                elif sig is not None and sig.side == SignalSide.SHORT:
                    opened = self._open_short(cash, sig, c)
                    if opened:
                        pos, cash = opened

            equity_curve.append((ts, self._equity(cash, pos, c)))

        if pos is not None:
            last = prepared.iloc[-1]
            ts = int(last["open_time"])
            c = float(last[f"{prefix}_close"])
            if pos.side == SignalSide.LONG:
                cash = self._close_long(symbol, pos, ts, c, "eod", trades, cash)
            else:
                cash = self._close_short(symbol, pos, ts, c, "eod", trades, cash)
            pos = None
            equity_curve[-1] = (ts, cash)

        final_eq = equity_curve[-1][1] if equity_curve else self.initial_capital
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
        loss_pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None and t.pnl is not None and t.pnl <= 0]
        avg_return_win_pct = sum(win_pnl_pcts) / len(win_pnl_pcts) if win_pnl_pcts else 0.0
        avg_return_loss_pct = sum(loss_pnl_pcts) / len(loss_pnl_pcts) if loss_pnl_pcts else 0.0

        metrics: dict[str, Any] = {
            "initial_capital": self.initial_capital,
            "final_equity": final_eq,
            "total_return_pct": (final_eq / self.initial_capital - 1.0) * 100.0 if self.initial_capital else 0.0,
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
            "commission_rate": self.commission_rate,
            "slippage_bps": self.slippage_bps,
        }

        return BacktestResult(
            symbol=symbol,
            interval=prefix,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Flatten ``Trade`` objects into a DataFrame for analysis / CSV export."""
    rows: list[dict[str, Any]] = []
    for t in trades:
        meta = t.meta or {}
        rows.append(
            {
                "symbol": t.symbol,
                "side": t.side.value,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "exit_reason": meta.get("exit_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def read_trades_csv(path: str | Path) -> pd.DataFrame:
    """Load trades written by ``scripts/run_backtest.py`` (or compatible CSV)."""
    df = pd.read_csv(Path(path))
    if "exit_reason" not in df.columns:
        df["exit_reason"] = ""
    return df


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, float) and (v == float("inf") or v != v):  # inf or nan
            out[k] = None
        else:
            out[k] = v
    return out


def save_run_summary(
    path: str | Path,
    metrics: dict[str, Any],
    strategy_config: dict[str, Any],
    *,
    symbol: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write metrics + strategy config for :func:`compare_runs` / :func:`load_run_summary`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metrics": _json_safe_metrics(dict(metrics)),
        "strategy_config": dict(strategy_config),
    }
    if symbol is not None:
        payload["symbol"] = symbol
    if extra:
        payload["extra"] = extra
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze_trades(trades_df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise losing trades, exit reasons, and time-of-day distribution (UTC).

    Use with :func:`trades_to_dataframe` or :func:`read_trades_csv`.
    """
    if trades_df.empty:
        return {
            "n_total": 0,
            "n_losers": 0,
            "n_winners": 0,
            "error": "empty trades DataFrame",
        }

    df = trades_df.copy()
    if "exit_reason" not in df.columns:
        df["exit_reason"] = ""

    winners = df[df["pnl"] > 0]
    losers = df[df["pnl"] <= 0]
    n = len(df)
    n_w, n_l = len(winners), len(losers)

    entry_dt = pd.to_datetime(df["entry_time"], unit="ms", utc=True)
    df["entry_hour_utc"] = entry_dt.dt.hour
    df["entry_dow"] = entry_dt.dt.day_name()

    losers_by_reason = losers["exit_reason"].value_counts(dropna=False).to_dict()
    winners_by_reason = winners["exit_reason"].value_counts(dropna=False).to_dict()

    sl_share_losers = float((losers["exit_reason"] == "sl").sum()) / n_l if n_l else 0.0
    tp_share_losers = float((losers["exit_reason"] == "tp").sum()) / n_l if n_l else 0.0
    time_share_losers = float((losers["exit_reason"] == "time").sum()) / n_l if n_l else 0.0
    tp_share_winners = float((winners["exit_reason"] == "tp").sum()) / n_w if n_w else 0.0

    hourly_losers = losers.assign(_h=pd.to_datetime(losers["entry_time"], unit="ms", utc=True).dt.hour)
    loser_hour_dist = hourly_losers["_h"].value_counts().sort_index().to_dict()

    # SL distance as %% of entry (theoretical risk before slip)
    sl_risk_pct: list[float] = []
    for _, r in losers.iterrows():
        if r["exit_reason"] != "sl" or pd.isna(r.get("entry_price")):
            continue
        ep = float(r["entry_price"])
        sl = r.get("stop_loss")
        if sl is None or pd.isna(sl) or ep == 0:
            continue
        side = str(r.get("side", "long"))
        if side == "long":
            sl_risk_pct.append(abs(ep - float(sl)) / ep * 100.0)
        else:
            sl_risk_pct.append(abs(float(sl) - ep) / ep * 100.0)

    notes: list[str] = []
    if sl_share_losers > 0.55 and n_l >= 5:
        notes.append("Majority of losers hit stop_loss — SL may be too tight or noisy entries.")
    if time_share_losers > 0.35 and n_l >= 5:
        notes.append("Many losers exit on time_stop — consider longer hold or earlier invalidation.")
    if tp_share_winners < 0.25 and n_w >= 5:
        notes.append("Few winners reach TP — TP may be too far or trend legs short.")

    return {
        "n_total": n,
        "n_winners": n_w,
        "n_losers": n_l,
        "win_rate": n_w / n if n else 0.0,
        "losers_by_exit_reason": losers_by_reason,
        "winners_by_exit_reason": winners_by_reason,
        "sl_share_among_losers": sl_share_losers,
        "tp_share_among_losers": tp_share_losers,
        "time_share_among_losers": time_share_losers,
        "tp_share_among_winners": tp_share_winners,
        "loser_entry_hour_utc_counts": {int(k): int(v) for k, v in loser_hour_dist.items()},
        "avg_loser_pnl": float(losers["pnl"].mean()) if n_l else 0.0,
        "median_loser_pnl": float(losers["pnl"].median()) if n_l else 0.0,
        "sl_risk_pct_losers_hit_sl": sl_risk_pct,
        "mean_sl_risk_pct_when_sl": float(sum(sl_risk_pct) / len(sl_risk_pct)) if sl_risk_pct else None,
        "hypotheses": notes,
    }


def adjust_parameters(trade_analysis: dict[str, Any]) -> dict[str, Any]:
    """Alias for :func:`suggest_parameter_adjustments` (compat with task naming)."""
    return suggest_parameter_adjustments(trade_analysis)


def suggest_parameter_adjustments(
    analysis: dict[str, Any],
    strategy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Heuristic parameter deltas from :func:`analyze_trades`.

    Merge into your strategy config (see :func:`merge_parameter_adjustments`).
    """
    cfg = dict(strategy_config or analysis.get("strategy_config") or {})
    sl_pct = float(cfg.get("stop_loss_pct", DEFAULT_BACKTEST_CONFIG["stop_loss_pct"]))
    tp_pct = float(cfg.get("take_profit_pct", DEFAULT_BACKTEST_CONFIG["take_profit_pct"]))
    rsi_long_max = float(cfg.get("m15_rsi_long_max", DEFAULT_BACKTEST_CONFIG["m15_rsi_long_max"]))
    rsi_short_min = float(cfg.get("m15_rsi_short_min", DEFAULT_BACKTEST_CONFIG["m15_rsi_short_min"]))
    time_h = int(cfg.get("time_stop_hours", DEFAULT_BACKTEST_CONFIG["time_stop_hours"]))
    swing_bars = int(cfg.get("h1_swing_bars", DEFAULT_BACKTEST_CONFIG["h1_swing_bars"]))

    out: dict[str, Any] = {}
    wr = float(analysis.get("win_rate", 0.0))
    sl_share = float(analysis.get("sl_share_among_losers", 0.0))
    time_share = float(analysis.get("time_share_among_losers", 0.0))
    tp_win_share = float(analysis.get("tp_share_among_winners", 0.0))

    if sl_share > 0.55 and analysis.get("n_losers", 0) >= 5:
        out["stop_loss_pct"] = round(min(sl_pct + 0.005, 0.06), 4)
    if wr < 0.45 and analysis.get("n_total", 0) >= 15:
        out["m15_rsi_long_max"] = round(max(rsi_long_max - 3.0, 25.0), 2)
        out["m15_rsi_short_min"] = round(min(rsi_short_min + 3.0, 75.0), 2)
    if time_share > 0.4 and analysis.get("n_losers", 0) >= 5:
        out["time_stop_hours"] = min(time_h + 12, 120)
    if tp_win_share < 0.2 and wr < 0.5 and analysis.get("n_winners", 0) >= 5:
        out["take_profit_pct"] = round(max(tp_pct - 0.01, 0.05), 4)

    if not out and analysis.get("n_losers", 0) >= 10:
        out["h1_swing_bars"] = min(swing_bars + 8, 120)

    out["_rationale"] = (
        f"sl_share_losers={sl_share:.2f}, win_rate={wr:.2f}, "
        f"time_losers={time_share:.2f}, tp_win_share={tp_win_share:.2f}"
    )
    return out


def merge_parameter_adjustments(
    base_config: dict[str, Any],
    adjustments: dict[str, Any],
) -> dict[str, Any]:
    """Apply non-meta keys from ``adjustments`` onto ``base_config`` with safe clamps."""
    skip = {"_rationale"}
    merged = {**base_config, **{k: v for k, v in adjustments.items() if k not in skip}}
    merged["stop_loss_pct"] = float(min(max(merged.get("stop_loss_pct", 0.02), 0.008), 0.08))
    # Keep take-profit in a realistic swing band (5–8%%) when merging auto-tune deltas
    merged["take_profit_pct"] = float(min(max(merged.get("take_profit_pct", 0.065), 0.05), 0.08))
    merged["m15_rsi_long_max"] = float(min(max(merged.get("m15_rsi_long_max", 52.0), 20.0), 65.0))
    merged["m15_rsi_short_min"] = float(min(max(merged.get("m15_rsi_short_min", 48.0), 35.0), 82.0))
    merged["h1_swing_bars"] = int(min(max(int(merged.get("h1_swing_bars", 48)), 12), 200))
    merged["time_stop_hours"] = int(min(max(int(merged.get("time_stop_hours", 48)), 8), 168))
    return merged


def compare_runs(run1_path: str | Path, run2_path: str | Path) -> pd.DataFrame:
    """
    Load two JSON summaries from :func:`save_run_summary` and compare key metrics.
    """
    a = load_run_summary(run1_path)
    b = load_run_summary(run2_path)
    ma = a.get("metrics") or {}
    mb = b.get("metrics") or {}
    keys = sorted(set(ma.keys()) | set(mb.keys()))
    rows: list[dict[str, Any]] = []
    for k in keys:
        v1, v2 = ma.get(k), mb.get(k)
        delta: float | None
        try:
            if v1 is not None and v2 is not None:
                delta = float(v2) - float(v1)
            else:
                delta = None
        except (TypeError, ValueError):
            delta = None
        rows.append({"metric": k, "run1": v1, "run2": v2, "delta": delta})
    return pd.DataFrame(rows)


def auto_tune(
    symbol: str,
    *,
    merged_dir: str | Path | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    initial_config: dict[str, Any] | None = None,
    target_win_rate: float = 0.5,
    max_iterations: int = 10,
    show_progress: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Manual-style tuning loop: run backtest → analyse losers → suggest params → re-run.

    Stops when ``win_rate >= target_win_rate`` or ``max_iterations`` is reached.
    """
    from backtest.data_loader import BacktestDataLoader
    from backtest.strategies.swing_strategy import SwingTradingStrategy

    base = {**DEFAULT_BACKTEST_CONFIG, **(initial_config or {})}
    strategy_keys = {
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
    }
    cfg_engine = {
        k: base[k]
        for k in ("initial_capital", "commission_rate", "slippage_bps")
        if k in base
    }
    strategy_cfg = {k: base[k] for k in strategy_keys if k in base}

    loader = BacktestDataLoader(merged_dir=merged_dir) if merged_dir else BacktestDataLoader()
    aligned = loader.load_aligned(symbol, ["M15", "H1", "H4"], start_ms=start_ms, end_ms=end_ms)
    if aligned.empty:
        return {"error": "empty aligned data", "aligned_rows": 0}

    iterations: list[dict[str, Any]] = []
    stopped = "max_iterations"
    out_dir = Path(output_dir) if output_dir else None
    last_run_config: dict[str, Any] = dict(strategy_cfg)

    for it in range(max_iterations):
        strat_cfg_i = merge_parameter_adjustments(strategy_cfg, {})
        last_run_config = dict(strat_cfg_i)
        strategy = SwingTradingStrategy(config=strat_cfg_i)
        prepared = strategy.prepare(aligned)
        engine = BacktestEngine(
            initial_capital=cfg_engine.get("initial_capital"),
            commission_rate=cfg_engine.get("commission_rate"),
            slippage_bps=cfg_engine.get("slippage_bps"),
            show_progress=show_progress,
        )
        result = engine.run(symbol, prepared, strategy)
        metrics = dict(result.metrics)
        wr = float(metrics.get("win_rate", 0.0))

        trades_df = trades_to_dataframe(result.trades)
        analysis = analyze_trades(trades_df)
        analysis["strategy_config"] = dict(strat_cfg_i)

        iter_payload: dict[str, Any] = {
            "iteration": it,
            "win_rate": wr,
            "profit_factor": metrics.get("profit_factor"),
            "n_trades": metrics.get("n_trades"),
            "strategy_config": dict(strat_cfg_i),
        }
        iterations.append(iter_payload)

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_run_summary(
                out_dir / f"tune_iter_{it:02d}.json",
                metrics,
                strat_cfg_i,
                symbol=symbol,
                extra={"iteration": it},
            )

        if wr >= target_win_rate:
            stopped = "target_win_rate"
            break

        adj = suggest_parameter_adjustments(analysis, strat_cfg_i)
        rationale = adj.pop("_rationale", "")
        if not adj:
            stopped = "no_adjustments"
            break
        strategy_cfg = merge_parameter_adjustments(strat_cfg_i, adj)
        iter_payload["adjustments_applied"] = adj
        iter_payload["adjustment_rationale"] = rationale

    return {
        "stopped_reason": stopped,
        "target_win_rate": target_win_rate,
        "iterations": iterations,
        "final_strategy_config": last_run_config,
        "final_win_rate": iterations[-1]["win_rate"] if iterations else 0.0,
    }
