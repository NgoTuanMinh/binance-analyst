# Backtest (swing strategy + engine)

Module này đọc dữ liệu **đã merge** từ pipeline (`crypto-data-pipeline/data/merged/`), align đa khung thời gian (15m / 1h / 4h), chạy chiến lược swing và mô phỏng lệnh (SL / TP / time stop, phí, trượt giá).

## Điều kiện dữ liệu

Với mỗi symbol cần đủ 3 file Parquet (do pipeline tạo):

- `{merged_dir}/{SYMBOL}/{SYMBOL}-15m.parquet`
- `{merged_dir}/{SYMBOL}/{SYMBOL}-1h.parquet`
- `{merged_dir}/{SYMBOL}/{SYMBOL}-4h.parquet`

## Môi trường Python

```bash
pip install -r crypto-data-pipeline/requirements.txt
```

## Chạy CLI (khuyến nghị: `scripts/run_backtest.py`)

Từ thư mục gốc **`Binance-analyst`**:

```bash
cd /path/to/Binance-analyst
python scripts/run_backtest.py --symbols BTCUSDT --start 2023-01-01 --end 2023-06-01
```

### Chạy cả danh sách symbol từ `top_300_symbols.json`

File JSON là mảng string `["BTCUSDT", ...]` (hoặc object có key `symbols`). Dùng `--symbols-json`; nên thêm `--continue-on-error` để symbol thiếu Parquet không dừng cả batch, và `--batch-summary-csv` để gom kết quả một file:

```bash
python scripts/run_backtest.py \
  --symbols-json crypto-data-pipeline/config/top_300_symbols.json \
  --continue-on-error \
  --no-progress \
  --batch-summary-csv out/batch_summary.csv \
  --start 2024-01-01 --end 2024-12-31
```

Trên máy nhiều CPU, thêm `--workers N` để chạy **song song từng symbol** (chỉ khi có **từ 2 symbol** trở lên). Ví dụ `--workers 8` hoặc `--workers 0` (dùng `os.cpu_count()`). Chi tiết: mục **Song song CPU (`--workers`)** bên dưới.

Mỗi symbol cần đủ `15m` / `1h` / `4h` Parquet trong `merged_dir`; thiếu sẽ ghi `status=missing_parquet` trong summary.

Sau khi chạy **nhiều symbol** (từ JSON hoặc `--symbols A,B,C`), script **luôn in** block **trung bình / median** trên các symbol `status=ok` (win_rate, return %, profit_factor, tổng số lệnh, v.v.). Nếu có `--batch-summary-csv out/x.csv`, mặc định ghi thêm **`out/x_aggregates.json`**. Không cần CSV thì có thể chỉ dùng `--batch-aggregate-json out/agg.json`.

`python run_backtest.py` ở gốc repo gọi cùng CLI (tương thích ngược).

### Baseline + xuất CSV / JSON (để so sánh lần chạy)

```bash
python scripts/run_backtest.py --symbols BTCUSDT --start 2023-01-01 --end 2023-06-01 \
  --export-trades-csv out/trades.csv --export-summary out/baseline.json
```

Kết quả in ra gồm **win rate**, **profit factor**, **avg return** (theo % PnL mỗi lệnh thắng/thua).

### Phân tích file trades CSV

```bash
python scripts/run_backtest.py --analyze-trades-csv out/trades.csv
```

In JSON: lệnh thua theo `exit_reason`, phân bố giờ vào lệnh (UTC), giả thuyết (SL quét nhiều, time stop, v.v.).

### So sánh hai lần chạy (JSON từ `--export-summary`)

```bash
python scripts/run_backtest.py --compare-run1 out/baseline.json --compare-run2 out/tuned.json
```

### Vòng lặp điều chỉnh tự động (`auto_tune`)

Heuristic: phân tích lệnh thua → `suggest_parameter_adjustments` → chạy lại (tối đa N vòng) cho đến khi `win_rate >= target` hoặc hết iteration.

**Một symbol:**

```bash
python scripts/run_backtest.py --auto-tune --symbols BTCUSDT --start 2023-01-01 --end 2023-06-01 \
  --target-win-rate 0.5 --max-tune-iterations 10 --tune-output-dir out/tune --no-progress
```

**Nhiều symbol** (`--symbols-json` hoặc `BTC,ETH,...`): mỗi symbol một job `auto_tune`. Với **`--workers 1`** (mặc định từ config nếu không truyền CLI) các job chạy **tuần tự**. Với **`--workers > 1`** các symbol chạy **song song** (ProcessPool). Nên thêm `--continue-on-error` nếu một số symbol chưa có Parquet.

```bash
python scripts/run_backtest.py --auto-tune --symbols-json crypto-data-pipeline/config/top_300_symbols.json \
  --continue-on-error --no-progress --max-tune-iterations 5 \
  --tune-output-dir out/tune_all --tune-batch-json out/tune_all.json
```

Ví dụ máy mạnh, tune nhiều symbol song song:

```bash
python scripts/run_backtest.py --auto-tune \
  --symbols-json crypto-data-pipeline/config/top_300_symbols.json \
  --continue-on-error --workers 8 --no-progress --max-tune-iterations 5 \
  --tune-output-dir out/tune_all --tune-batch-json out/tune_all.json
```

- `--tune-output-dir`: **một symbol** → `tune_iter_XX.json` trực tiếp trong thư mục đó; **nhiều symbol** → `OUT/BTCUSDT/tune_iter_XX.json`, `OUT/ETHUSDT/...`
- `--tune-batch-json`: ghi toàn bộ kết quả (một object hoặc `{"results": [...]}`).
- Khi `--workers > 1`, script **chạy hết** mọi symbol (không dừng sớm như vòng tuần tự khi tắt `--continue-on-error`); muốn “dừng khi lỗi đầu tiên” thì dùng `--workers 1`.

### Song song CPU (`--workers`) và cấu hình mặc định

- **Phạm vi:** chỉ khi có **từ 2 symbol** trở lên — backtest batch hoặc `auto_tune` đa symbol. Một symbol vẫn chạy một process; vòng **từng nến** trong engine vẫn tuần tự.
- **`--workers N`:** số process song song. **`--workers 0`** = dùng hết logical CPU (`os.cpu_count()`).
- **Ưu tiên:** CLI `--workers` > biến môi trường **`BACKTEST_WORKERS`** (chỉ khi **không** truyền `--workers`) > **`parallel_workers`** trong `backtest/configs/default_config.py` (hằng `PARALLEL_WORKERS`).
- **RAM:** mỗi worker nạp **đủ bộ dữ liệu align** cho một symbol — máy nhiều core nhưng ít RAM thì hạ `--workers` (ví dụ 4–8 thay vì 32).

### Config tối ưu: workflow gợi ý (máy mạnh)

Ở đây “config tối ưu” là **bộ tham số chiến lược** (SL/TP, RSI, swing bars, …), không chỉ số worker.

1. **Tune một symbol đại diện** (thường BTC) trên khoảng thời gian bạn tin tưởng — mỗi lần `auto_tune` vẫn là nhiều iteration **trên một coin** (không song song CPU giữa các iteration):

```bash
python scripts/run_backtest.py --auto-tune \
  --symbols BTCUSDT \
  --start 2024-01-01 --end 2024-12-31 \
  --target-win-rate 0.52 \
  --max-tune-iterations 10 \
  --tune-output-dir out/tune_btc \
  --tune-batch-json out/tune_btc.json \
  --no-progress
```

Trong JSON đầu ra, lấy **`final_strategy_config`** (và có thể chép các giá trị vào `backtest/configs/default_config.py` làm mặc định cho các lần chạy sau).

2. **Tune nhiều symbol song song** trên máy mạnh: thêm `--workers` như ví dụ ở trên.

3. **Kiểm tra sau khi có param mới:** chạy backtest + xuất summary để so sánh với baseline:

```bash
python scripts/run_backtest.py --symbols BTCUSDT --start 2024-01-01 --end 2024-12-31 \
  --export-summary out/after_tune.json --no-progress
```

4. **Đo trên cả universe** (param “tối ưu trên BTC” có ổn rộng không) — batch nhiều symbol + worker:

```bash
python scripts/run_backtest.py \
  --symbols-json crypto-data-pipeline/config/top_300_symbols.json \
  --continue-on-error --workers 8 --no-progress \
  --batch-summary-csv out/batch_summary.csv \
  --start 2024-01-01 --end 2024-12-31
```

Đọc `batch_summary.csv`, block trung bình/median in ra console, và file `*_aggregates.json` (nếu có).

| Tham số CLI | Ý nghĩa |
|-------------|---------|
| `--symbols` | Danh sách `BTCUSDT,ETHUSDT` (mặc định `BTCUSDT`) |
| `--symbol` | Một symbol (alias) |
| `--merged-dir` | Thư mục merged |
| `--start` / `--end` | UTC `YYYY-MM-DD` |
| `--export-trades-csv` | CSV từng lệnh (cột `exit_reason`, …) |
| `--export-summary` | JSON cho `compare_runs` |
| `--auto-tune` | Bật `auto_tune` (một hoặc nhiều symbol) |
| `--target-win-rate` | Mặc định `0.5` |
| `--max-tune-iterations` | Mặc định `10` |
| `--tune-batch-json` | File JSON gộp kết quả tune |
| `--workers` | Song song đa symbol: số process (`0` = mặc định theo CPU); bỏ qua thì dùng env/config (xem mục trên) |
| `--continue-on-error` | Nhiều symbol: ghi lỗi và chạy tiếp; exit 1 nếu có symbol lỗi |

## API trong `backtest/engine.py`

- `trades_to_dataframe(trades)` — từ `BacktestResult.trades` sang `DataFrame`.
- `read_trades_csv(path)` — đọc CSV do CLI xuất.
- `analyze_trades(trades_df)` — thống kê lệnh thua / giờ UTC / tỷ lệ thoát SL-time-TP.
- `suggest_parameter_adjustments(analysis, strategy_config?)` — dict gợi ý tham số (kèm `_rationale`).
- `merge_parameter_adjustments(base, adjustments)` — gộp + clamp an toàn.
- `save_run_summary` / `load_run_summary` — JSON một lần chạy.
- `compare_runs(path1, path2)` — `DataFrame` so sánh metrics.
- `auto_tune(symbol, ...)` — vòng lặp tuning (tải data một lần).

## Windows (PowerShell)

```powershell
cd C:\path\to\Binance-analyst
.\crypto-data-pipeline\.venv\Scripts\Activate.ps1
python scripts/run_backtest.py --symbols BTCUSDT
```

## Cấu trúc module

- `data_loader.py` — `BacktestDataLoader`
- `strategies/swing_strategy.py` — `SwingTradingStrategy`
- `engine.py` — `BacktestEngine` + phân tích / tuning ở trên
- `models.py`, `configs/default_config.py` — mặc định đã tune (BTC 2024): SL 3.5%, TP 6.5% (mục tiêu band 5–8%), time stop 96h, RSI long ≤42 / short ≥60, `h1_swing_bars` 64.

## Gọi từ code

```python
from pathlib import Path
from backtest.data_loader import BacktestDataLoader
from backtest.strategies.swing_strategy import SwingTradingStrategy
from backtest.engine import BacktestEngine

loader = BacktestDataLoader(merged_dir=Path("crypto-data-pipeline/data/merged"))
aligned = loader.load_aligned("BTCUSDT", ["M15", "H1", "H4"])
strategy = SwingTradingStrategy()
prepared = strategy.prepare(aligned)
engine = BacktestEngine(show_progress=True)
result = engine.run("BTCUSDT", prepared, strategy)
print(result.metrics["win_rate"], result.metrics["profit_factor"])
```

Chạy với **gốc repo** trên `sys.path` (giống CLI).
