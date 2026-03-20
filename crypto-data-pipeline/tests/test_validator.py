"""Tests for validator continuity checks."""

import pandas as pd

from src.validator import DataValidator


def test_check_data_continuity_detects_missing() -> None:
    validator = DataValidator("data/merged")
    # Missing 01:00 candle in 1h interval
    times = [
        1735689600000,  # 2025-01-01 00:00:00 UTC
        1735696800000,  # 2025-01-01 02:00:00 UTC
    ]
    df = pd.DataFrame(
        {
            "open_time": times,
            "open": [1, 1],
            "high": [2, 2],
            "low": [0.5, 0.5],
            "close": [1.2, 1.3],
            "volume": [10, 11],
        }
    )
    gaps = validator.check_data_continuity(df, "1h")
    assert len(gaps) >= 1


def test_check_data_continuity_with_nanoseconds_input() -> None:
    validator = DataValidator("data/merged")
    # Missing 01:00 candle in 1h interval, timestamps in nanoseconds
    times_ns = [
        1735689600000000000,  # 2025-01-01 00:00:00 UTC
        1735696800000000000,  # 2025-01-01 02:00:00 UTC
    ]
    df = pd.DataFrame(
        {
            "open_time": times_ns,
            "open": [1, 1],
            "high": [2, 2],
            "low": [0.5, 0.5],
            "close": [1.2, 1.3],
            "volume": [10, 11],
        }
    )
    gaps = validator.check_data_continuity(df, "1h")
    assert len(gaps) >= 1
