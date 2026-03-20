# Crypto Data Pipeline cho Backtest Swing Trading

## Tong quan
Du an thu thap du lieu lich su tu Binance.vision de phuc vu backtest chien luoc swing trade tren cac khung M15, H1, H4.

## Tinh nang
- Tai du lieu tu dong tu Binance.vision
- Ho tro 200-300+ symbols
- 3 intervals: 15m, 1h, 4h
- Download song song, co resume support
- Kiem tra checksum, validate du lieu
- Merge va luu tru dang Parquet
- Bao cao chat luong du lieu
- Incremental updates

## Yeu cau he thong
- Python 3.9+
- Dung luong o cung: khoang 80-150 GB cho 300 symbols (tuy interval va time range)
- RAM: toi thieu 8GB, khuyen nghi 16GB

## Cai dat nhanh
```bash
cd crypto-data-pipeline
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python main.py --help
```

## Cach su dung
1) Lay danh sach top coins
```bash
python scripts/get_top_symbols.py --top 300 --output config/top_300_symbols.json
```

2) Tai du lieu
```bash
python main.py --start 2020-01-01 --end 2025-12-31 --symbols config/top_300_symbols.json --intervals 15m 1h 4h
```

Cap nhat du lieu moi:
```bash
python main.py --update-only --symbols config/top_300_symbols.json --intervals 1h
```

3) Kiem tra chat luong
```bash
python main.py --validate --symbols config/top_300_symbols.json --intervals 1h
```

4) Xuat bao cao
```bash
python main.py --report --symbols config/top_300_symbols.json --intervals 1h --format html
```

## Cau truc du lieu dau ra
```text
data/
├── raw/               # File ZIP goc
├── processed/         # File CSV da giai nen
│   └── BTCUSDT/
│       └── 1h/
│           ├── BTCUSDT-1h-2025-01-01.csv
│           └── ...
└── merged/            # File tong hop
    └── BTCUSDT/
        ├── BTCUSDT-1h.parquet
        ├── BTCUSDT-4h.parquet
        └── BTCUSDT-15m.parquet
```
