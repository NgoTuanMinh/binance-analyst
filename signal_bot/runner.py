#!/usr/bin/env python3
"""
Poll Binance spot klines (15m / 1h / 4h), run the same SwingTradingStrategy as backtest,
and send entry alerts to Telegram. You place orders manually.

Env:
  TELEGRAM_BOT_TOKEN   — BotFather token
  TELEGRAM_CHAT_ID     — Target chat or channel id
  SIGNAL_SYMBOLS       — Comma-separated, default BTCUSDT
  SIGNAL_POLL_SEC      — Sleep between full scans, default 60
  SIGNAL_CACHE_DIR     — Parquet cache directory, default <repo>/data/signal_klines_cache
  SIGNAL_STATE_PATH    — Dedup state JSON, default <cache_dir>/signal_state.json

Run from repo root::

    export TELEGRAM_BOT_TOKEN=...
    export TELEGRAM_CHAT_ID=...
    python -m signal_bot.runner
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from backtest.data_loader import BacktestDataLoader
from backtest.models import Signal, SignalSide
from backtest.strategies.swing_strategy import SwingTradingStrategy

from signal_bot.klines_cache import load_mtf_cached
from signal_bot.telegram_notify import send_message


def _parse_symbols(s: str) -> list[str]:
    return [x.strip().upper() for x in s.replace(";", ",").split(",") if x.strip()]


def _fmt_ts_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _format_signal_message(symbol: str, sig: Signal) -> str:
    side = "LONG" if sig.side == SignalSide.LONG else "SHORT"
    emoji = "🟢" if sig.side == SignalSide.LONG else "🔴"

    def px(x: float) -> str:
        s = f"{x:.8f}".rstrip("0").rstrip(".")
        return s or "0"

    ep = float(sig.price or 0.0)
    sl = float(sig.meta.get("stop_loss", 0.0))
    tp = float(sig.meta.get("take_profit", 0.0))
    sl_pct = float(sig.meta.get("stop_loss_pct", 0.0)) * 100.0
    tp_pct = float(sig.meta.get("take_profit_pct", 0.0)) * 100.0
    ts_h = int(sig.meta.get("time_stop_ms", 0)) // (3600 * 1000)
    t_bar = _fmt_ts_utc(int(sig.time))
    return "\n".join(
        [
            f"{emoji} *{symbol}* `{side}`",
            f"Entry (M15 close): `{px(ep)}`",
            f"SL: `{px(sl)}` (~{sl_pct:.2f}%)",
            f"TP: `{px(tp)}` (~{tp_pct:.2f}%)",
            f"Time stop: ~{ts_h}h from bar open",
            f"M15 bar open: `{t_bar}`",
            "",
            "_Manual execution only — not financial advice._",
        ]
    )


def _load_state(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def evaluate_symbol(
    frames: dict[str, pd.DataFrame],
    strategy: SwingTradingStrategy,
    loader: BacktestDataLoader,
    now_ms: int,
) -> Signal | None:
    aligned = loader.align_timeframes(frames, base_interval="15m")
    if aligned.empty:
        return None
    ct_col = "15m_close_time"
    if ct_col not in aligned.columns:
        return None
    prepared = strategy.prepare(aligned)
    closed = prepared[prepared[ct_col] <= now_ms]
    if closed.empty:
        return None
    last_pos = int(closed.index[-1])
    return strategy.signal_at(prepared, last_pos)


def run_loop(
    *,
    symbols: list[str],
    cache_dir: Path,
    state_path: Path,
    poll_sec: float,
    token: str,
    chat_id: str,
    strategy_config: dict | None,
) -> None:
    strategy = SwingTradingStrategy(config=strategy_config)
    loader = BacktestDataLoader(merged_dir=_ROOT, use_cache=False)
    session = requests.Session()
    state = _load_state(state_path)

    while True:
        now_ms = int(time.time() * 1000)
        for symbol in symbols:
            try:
                frames = load_mtf_cached(cache_dir, symbol, session=session)
                sig = evaluate_symbol(frames, strategy, loader, now_ms)
                if sig is None:
                    continue
                key = f"{sig.side.value}:{sig.time}"
                if state.get(symbol) == key:
                    continue
                text = _format_signal_message(symbol, sig)
                send_message(token, chat_id, text, parse_mode="Markdown", session=session)
                state[symbol] = key
                _save_state(state_path, state)
            except Exception as e:
                print(f"[{symbol}] {e!s}", file=sys.stderr)
        time.sleep(max(5.0, float(poll_sec)))


def main() -> None:
    p = argparse.ArgumentParser(description="Swing signal Telegram poller (Binance + disk cache)")
    p.add_argument(
        "--symbols",
        default=os.environ.get("SIGNAL_SYMBOLS", "BTCUSDT"),
        help="Comma-separated symbols (default env SIGNAL_SYMBOLS or BTCUSDT)",
    )
    p.add_argument(
        "--poll-sec",
        type=float,
        default=float(os.environ.get("SIGNAL_POLL_SEC", "60")),
        help="Seconds between scans (default env SIGNAL_POLL_SEC or 60)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("SIGNAL_CACHE_DIR", str(_ROOT / "data" / "signal_klines_cache"))),
        help="Parquet cache directory",
    )
    p.add_argument(
        "--state-path",
        type=Path,
        default=os.environ.get("SIGNAL_STATE_PATH") or None,
        help="Dedup state JSON (default env SIGNAL_STATE_PATH or <cache-dir>/signal_state.json)",
    )
    p.add_argument(
        "--telegram-token",
        default=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        help="Override TELEGRAM_BOT_TOKEN",
    )
    p.add_argument(
        "--telegram-chat-id",
        default=os.environ.get("TELEGRAM_CHAT_ID", ""),
        help="Override TELEGRAM_CHAT_ID",
    )
    args = p.parse_args()
    token = (args.telegram_token or "").strip()
    chat_id = (args.telegram_chat_id or "").strip()
    if not token or not chat_id:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (or pass --telegram-*).", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir).resolve()
    state_path = Path(args.state_path).resolve() if args.state_path else (cache_dir / "signal_state.json")

    symbols = _parse_symbols(args.symbols)
    run_loop(
        symbols=symbols,
        cache_dir=cache_dir,
        state_path=state_path,
        poll_sec=args.poll_sec,
        token=token,
        chat_id=chat_id,
        strategy_config=None,
    )


if __name__ == "__main__":
    main()
