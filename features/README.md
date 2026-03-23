# Features (ML / AI)

## Mục đích — dùng để làm gì?

Thư mục **`features/`** phục vụ **chuẩn bị dữ liệu phân tích / học máy** trên crypto (Binance), từ OHLCV đã merge trong `crypto-data-pipeline`:

| Mục tiêu | Mô tả ngắn |
|----------|------------|
| **Huấn luyện & đánh giá ML** | Tạo ma trận **một dòng = một mốc thời gian (15m)** gồm nhiều chỉ báo kỹ thuật đa khung (M15/H1/H4), pattern giá, khái niệm SMC, regime thị trường, và **cột target** (return tương lai, phân loại, risk-adjusted, v.v.) để đưa vào mô hình dự báo hướng/giá hoặc lọc tín hiệu. |
| **Nghiên cứu & so sánh feature** | Xuất Parquet theo symbol, dễ nối nhiều mã; có thể xếp hạng tương quan feature ↔ target (`FeaturePipeline.get_feature_importance_ranking`) làm điểm khởi đầu chọn biến. |
| **Gắn với chiến lược backtest** | Target kiểu **trade** (`would_trade_successful`, `optimal_sl` / `optimal_tp`, …) gắn với logic :class:`backtest.strategies.swing_strategy.SwingTradingStrategy` — hữu ích khi muốn học “tín hiệu giống bot” hoặc đánh giá khả năng thành công theo cùng quy tắc SL/TP/time-stop. |
| **Tự động hóa pipeline** | :class:`FeaturePipeline` và `scripts/generate_features.py` giúp **lặp lại** quy trình: load merged → sinh feature → lưu file — tránh làm tay từng symbol/khoảng thời gian. |

**Không phải**: thư mục này **không** tải dữ liệu thô từ Binance hay chạy lệnh giao dịch; nó đọc **Parquet đã merge** và chỉ **tính toán cột mới**.

---

Module layout:

| File | Role |
|------|------|
| `technical.py` | Low-level functions + :class:`TechnicalFeatures` (full M15/H1/H4 spec: EMA/SMA, MACD, ADX+DI, Supertrend (3,10)/(5,20), RSI, Stoch 14-3-3, Williams %R, CCI, BB+%B+bandwidth, ATR, Keltner, OBV, vol SMA, VP POC) |
| `price_action.py` | Candle patterns, S/R distance, swings, FVG |
| `smart_money.py` | SMC: daily/weekly liquidity, sweeps, order-block distance + volume strength, structure/BOS/CHoCH, Fib + OTE (+ legacy `smc_*`) |
| `market_regime.py` | ADX/ATR regime: `regime_trending`/`ranging`/`volatile`, `trend_strength`, `volatility_regime`, `regime_M15`/`H1`/`H4` |
| `target.py` | Horizons (`target_*_return`, `target_up_*`, `target_class`, risk-adjusted), legacy `target_fwd_ret_*`, swing-strategy trade labels (`would_trade_successful`, `optimal_*`) |
| `pipeline.py` | :class:`FeaturePipeline`, ``load_mtf_frames``, ``build_feature_matrix``, CLI / parallel helpers |

---

## Hướng dẫn sử dụng

### Điều kiện trước khi chạy

1. **Dữ liệu merged** đã có sẵn theo cấu trúc Parquet:
   - `crypto-data-pipeline/data/merged/{SYMBOL}/{SYMBOL}-15m.parquet`
   - Cùng symbol: `{SYMBOL}-1h.parquet`, `{SYMBOL}-4h.parquet` (khuyến nghị cho đủ chỉ báo MTF + target giao dịch).
2. **Python packages**: `pandas`, `numpy`, `pyarrow` (ghi/đọc Parquet).
3. Chạy lệnh từ **thư mục gốc repo** (`Binance-analyst`) hoặc đảm bảo `PYTHONPATH` trỏ vào root để `import features` hoạt động.

### Luồng dữ liệu (tóm tắt)

- **Khung thời gian gốc (base)** luôn là **15m** (`15m`): mỗi dòng = một nến 15 phút.
- **H1 / H4** được merge ngược theo `open_time` (as-of backward) lên timeline 15m.
- Cột chỉ báo có tiền tố `15m_`, `1h_`, `4h_` (từ :class:`TechnicalFeatures`).

### Cách 1 — Script khuyến nghị: `scripts/generate_features.py`

Dùng khi cần **nhiều symbol**, **lọc theo ngày**, và lưu ra **`features/data/`** (mặc định).

```bash
cd /path/to/Binance-analyst

# Top 100 symbol đầu (từ crypto-data-pipeline/config/top_300_symbols.json)
python scripts/generate_features.py \
  --symbols top_100 \
  --intervals 15m 1h 4h \
  --start 2020-01-01 \
  --end 2025-12-31

# Vài symbol cụ thể
python scripts/generate_features.py \
  --symbols BTCUSDT,ETHUSDT \
  --intervals 15m 1h 4h \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --output-dir features_output

# Bỏ cột target (chỉ feature)
python scripts/generate_features.py --symbols BTCUSDT --intervals 15m 1h 4h \
  --start 2024-01-01 --end 2024-02-01 --no-targets

# Bỏ target kiểu “trade” (swing strategy); vẫn giữ target return nếu không dùng --no-targets
python scripts/generate_features.py --symbols BTCUSDT --intervals 15m 1h 4h \
  --start 2024-01-01 --end 2024-02-01 --no-trade-targets

# Sau khi build, in top 20 |correlation| với một cột target (JSON)
python scripts/generate_features.py --symbols BTCUSDT --intervals 15m 1h 4h \
  --start 2024-01-01 --end 2024-03-01 \
  --correlation-target target_1h_return \
  --correlation-method pearson
```

**Tham số quan trọng**

| Tham số | Ý nghĩa |
|--------|---------|
| `--symbols` | `top_100`, `top_300`, CSV, hoặc đường dẫn file JSON (list hoặc `{"symbols": [...]}`) |
| `--intervals` | Phải gồm **`15m`**. Có thể ghi `M15`, `H1`, `H4` — sẽ được chuẩn hóa |
| `--start` / `--end` | UTC. Nếu chỉ ghi ngày (`YYYY-MM-DD`), **end** được hiểu là **hết ngày đó** |
| `--merged-dir` | Mặc định `crypto-data-pipeline/data/merged` |
| `--output-dir` | Mặc định `features/data` — mỗi symbol một file `{SYMBOL}_features.parquet` |
| `--strict` | Nếu **bật**: lỗi ngay khi một symbol **không** có đủ file Parquet merged; mặc định **tắt** → **bỏ qua** symbol thiếu dữ liệu (in cảnh báo ra stderr) |
| `--no-progress` | Tắt thanh `tqdm` (chỉ script `generate_features.py`; xem mục **Tiến trình** bên dưới) |

**Lưu ý:** Danh sách `top_100` / `top_300` có thể chứa symbol bạn **chưa tải/merge** — khi đó không phải lỗi code mà thiếu file; chạy không `--strict` sẽ chỉ xử lý symbol đủ `15m`/`1h`/`4h` Parquet.

### Cách 2 — Class `FeaturePipeline` (Python)

Phù hợp notebook / pipeline tùy biến (truyền thêm `**kwargs` vào `build_feature_matrix`).

```python
from pathlib import Path
from features.pipeline import FeaturePipeline

pl = FeaturePipeline(
    "BTCUSDT,ETHUSDT",           # hoặc ["BTCUSDT", "ETHUSDT"], hoặc "top_100"
    ["15m", "1h", "4h"],
    "2024-01-01",
    "2024-03-01",
    merged_dir=Path("crypto-data-pipeline/data/merged"),
    output_dir=Path("features/data"),
    # tuỳ chọn giống build_feature_matrix, ví dụ:
    # include_targets=True,
    # include_trade_targets=True,
    # target_horizons=(1, 4, 16, 96),
)

pl.load_data()                          # mặc định: bỏ qua symbol không đủ file merged
pl.load_data(skip_missing_symbols=False)  # fail ngay nếu thiếu Parquet (giống hành vi cũ)
pl.generate_all_features()  # technical + price_action + smart_money + regime + targets
paths = pl.save_features()  # dict symbol -> Path parquet

feature_cols = pl.get_feature_names()  # bỏ open_time, symbol, target_*
rank = pl.get_feature_importance_ranking("target_1h_return", method="pearson")
```

- **`get_feature_names()`**: mặc định **không** trả về cột `target_*` (dùng cho tập X). Bật `include_targets=True` nếu cần liệt kê đủ.
- **`get_feature_importance_ranking`**: xếp hạng theo **giá trị tuyệt đối correlation** với `target_col` trên toàn bộ dữ liệu đã concat (mọi symbol). Chỉ xét cột **số**.

### Cách 3 — Module `python -m features.pipeline` (nhiều worker)

Phù hợp xử lý **song song** theo list symbol (ít tùy chọn ngày hơn script trên; dùng `--start-ms` / `--end-ms` nếu cần).

```bash
python -m features.pipeline \
  --symbols-json crypto-data-pipeline/config/top_300_symbols.json \
  --workers 4 \
  --output-dir features_output

python -m features.pipeline --symbols BTCUSDT --workers 1 --output-dir features_output
```

### Cách 4 — API từng bước (một symbol)

Khi đã có `dict` DataFrame theo khung `15m` / `1h` / `4h`:

```python
from pathlib import Path
from features.pipeline import load_mtf_frames, build_feature_matrix

frames = load_mtf_frames(
    "BTCUSDT",
    Path("crypto-data-pipeline/data/merged"),
    timeframes=("M15", "H1", "H4"),
    start_ms=None,
    end_ms=None,
)
df = build_feature_matrix("BTCUSDT", frames)
```

### Ghi chú về target & trade labels

- **Target theo giờ/ngày** (`target_1h_return`, …) dùng số **nến 15m** tương ứng (4 / 16 / 96).
- **`would_trade_successful`, `optimal_sl`, `optimal_tp`, …** cần **đủ OHLCV 1h + 4h** đã merge; pipeline truyền `mtf_frames` tự động. Nếu thiếu H1/H4, các cột này có thể toàn **NaN** — dùng `--no-trade-targets` nếu không cần.
- **Các dòng cuối chuỗi** thường **NaN** ở cột target (không đủ future bars).

### Dependencies

`pandas`, `numpy`, `pyarrow` (Parquet).

### Tiến trình (progress)

- **`FeaturePipeline`**: ba bước **Load merged Parquet → Build feature matrix → Save Parquet** mỗi bước có thanh **`tqdm`** trên **stderr** (kèm tiền tố `[features]`).
- **Tắt progress**: `FeaturePipeline(..., show_progress=False)` hoặc script `scripts/generate_features.py --no-progress`; module CLI `python -m features.pipeline --no-progress`.
- **Song song** (`--workers` > 1): thanh tiến trình đếm số job **hoàn thành** (không phải thứ tự symbol).
- Nếu **chưa cài `tqdm`**, code in từng dòng `[features] … [i/N] SYMBOL` ra stderr (fallback).

---

## Quick reference (EN)

- **Data path**: `crypto-data-pipeline/data/merged/{SYMBOL}/{SYMBOL}-{15m|1h|4h}.parquet` (via `BacktestDataLoader`).
- **Outputs**: default `features/data/{SYMBOL}_features.parquet` (script / `FeaturePipeline`); or any `--output-dir`.
- **Base TF**: **15m**; H1/H4 merged as-of on `open_time`.
- **Parallel many symbols**: `python -m features.pipeline --symbols-json ... --workers 4 --output-dir ...`
- **One-shot symbol write**: `build_and_write_symbol("BTCUSDT", merged_dir, out_dir)`.
