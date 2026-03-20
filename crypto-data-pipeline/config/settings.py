"""Static settings for the crypto data pipeline."""

from pathlib import Path

BASE_URL = "https://data.binance.vision"
DATA_TYPES = {"spot": ["klines"], "futures": ["klines"]}
INTERVALS = ["15m", "1h", "4h"]
SYMBOLS_FILE = "config/top_300_symbols.json"
DOWNLOAD_DIR = "data/raw"
EXTRACT_DIR = "data/processed"
MERGE_DIR = "data/merged"
LOG_DIR = "logs"
MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 2
MAX_WORKERS = 5
CHECKPOINT_FILE = "logs/pipeline_checkpoint.json"
CACHE_DIR = "data/cache"
REPORT_DIR = "logs/reports"
DEFAULT_VOLUME_MA_WINDOW = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
