## Bot tín hiệu Telegram (`signal_bot/`)

Bot **polling** lấy nến **Binance Spot** (15m / 1h / 4h), cache **Parquet trên disk**, ghép đa khung giống backtest, rồi chạy **`SwingTradingStrategy`** với tham số mặc định từ `backtest/configs/default_config.py`. Bot **chỉ gửi tin nhắn** (entry / SL / TP / time stop); **không** đặt lệnh — bạn tự vào lệnh.

Liên hệ với phần feature: các **trade label** trong `target.py` (`would_trade_successful`, …) cũng bám logic swing strategy; bot live là lớp “cảnh báo thời gian thực” cùng một quy tắc vào/ra lý thuyết.

### Điều kiện

- **Repo root** `Binance-analyst` (để `import backtest`, `import signal_bot`).
- **Telegram**: tạo bot qua [@BotFather](https://t.me/BotFather), lấy **token**; **chat id** (cá nhân hoặc kênh) — ví dụ qua `https://api.telegram.org/bot<TOKEN>/getUpdates` sau khi nhắn bot một tin.

### Cài đặt nhanh (Linux / macOS / Windows có Python)

```bash
cd /path/to/Binance-analyst
pip install -r signal_bot/requirements.txt
cp signal_bot/.env.example .env
# Sửa .env: TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...
python -m signal_bot.runner
```

Có thể dùng `python-dotenv`: file **`.env`** đặt ở **thư mục gốc repo** (runner tự `load_dotenv` nếu đã cài gói).

### Cài đặt nhanh trên Debian 12

Chạy **bằng user thường** (không `sudo` cả script — để `.venv` thuộc đúng user). Script chỉ gọi `sudo` khi cài gói `apt`:

```bash
cd /path/to/Binance-analyst
bash scripts/setup_signal_bot_debian12.sh
```

Script sẽ: cài `python3`, `python3-venv`, `python3-pip`, …; tạo **`.venv`**; `pip install -r signal_bot/requirements.txt`; copy `.env` từ `signal_bot/.env.example` nếu chưa có; tạo `data/signal_klines_cache/`; (tuỳ chọn) gợi ý **systemd --user** unit `binance-signal-bot.service`.

Sau khi điền `.env`:

```bash
cd /path/to/Binance-analyst
source .venv/bin/activate
python -m signal_bot.runner
```

**Systemd user** (nếu đã cài unit):

```bash
systemctl --user enable --now binance-signal-bot.service
journalctl --user -u binance-signal-bot.service -f
```

Chạy nền khi không đăng nhập GUI:

```bash
sudo loginctl enable-linger "$USER"
```

### Biến môi trường (tóm tắt)

| Biến                 | Bắt buộc | Mô tả                                                                        |
| -------------------- | -------- | ---------------------------------------------------------------------------- |
| `TELEGRAM_BOT_TOKEN` | Có       | Token từ BotFather                                                           |
| `TELEGRAM_CHAT_ID`   | Có       | Chat hoặc channel id                                                         |
| `SIGNAL_SYMBOLS`     | Không    | CSV symbol, mặc định `BTCUSDT`                                               |
| `SIGNAL_POLL_SEC`    | Không    | Chu kỳ quét (giây), mặc định `60`                                            |
| `SIGNAL_CACHE_DIR`   | Không    | Thư mục Parquet cache; mặc định `data/signal_klines_cache` (dưới repo)       |
| `SIGNAL_STATE_PATH`  | Không    | File JSON chống gửi trùng cùng một bar; mặc định `<cache>/signal_state.json` |

CLI tương đương: `python -m signal_bot.runner --help` (`--symbols`, `--poll-sec`, `--cache-dir`, `--state-path`, `--telegram-token`, `--telegram-chat-id`).

### Cache & trạng thái trên disk

- **Nến**: `data/signal_klines_cache/{SYMBOL}_{15m|1h|4h}.parquet` (đã liệt kê trong `.gitignore` ở root repo).
- **Dedup tín hiệu**: `signal_state.json` (theo symbol, không spam lại cùng nến + cùng side).

### Ví dụ tin Telegram (minh họa)

Với LONG, entry ~65 000 USDT, SL/TP theo config mặc định (~3,5% / ~6,5%), time stop 96h, tin gửi dạng Markdown tương tự:

```text
🟢 *BTCUSDT* `LONG`
Entry (M15 close): `65000`
SL: `62725` (~3.50%)
TP: `69225` (~6.50%)
Time stop: ~96h from bar open
M15 bar open: `2025-03-25 21:15 (UTC+7)`  *(cùng thời điểm 14:15 UTC)*

_Manual execution only — not financial advice._
```

---

## Quick reference (EN)

- **Data path**: `crypto-data-pipeline/data/merged/{SYMBOL}/{SYMBOL}-{15m|1h|4h}.parquet` (via `BacktestDataLoader`).
- **Outputs**: default `features/data/{SYMBOL}_features.parquet` (script / `FeaturePipeline`); or any `--output-dir`.
- **Base TF**: **15m**; H1/H4 merged as-of on `open_time`.
- **Parallel many symbols**: `python -m features.pipeline --symbols-json ... --workers 4 --output-dir ...`
- **One-shot symbol write**: `build_and_write_symbol("BTCUSDT", merged_dir, out_dir)`.
