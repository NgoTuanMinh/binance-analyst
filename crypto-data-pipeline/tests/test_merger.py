"""Tests for merger module."""

from pathlib import Path

import pandas as pd

from src.merger import DataMerger


def test_merge_removes_duplicates(tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "BTCUSDT" / "1h"
    processed.mkdir(parents=True, exist_ok=True)
    merged = tmp_path / "merged"

    row1 = [1000, 1, 2, 0.5, 1.5, 10, 1999, 10, 1, 5, 5, 0]
    row2 = [1000, 1, 2, 0.5, 1.6, 12, 1999, 12, 1, 5, 5, 0]
    row3 = [2000, 2, 3, 1.5, 2.5, 20, 2999, 20, 1, 10, 10, 0]
    pd.DataFrame([row1, row3]).to_csv(processed / "a.csv", index=False, header=False)
    pd.DataFrame([row2]).to_csv(processed / "b.csv", index=False, header=False)

    merger = DataMerger(str(tmp_path / "processed"), str(merged))
    out = merger.merge_symbol_interval("BTCUSDT", "1h")
    assert len(out) == 2
    # keep last duplicate (row2 close=1.6)
    assert float(out[out["open_time"] == 1000]["close"].iloc[0]) == 1.6


def test_merge_normalizes_nanoseconds_to_milliseconds(tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "BTCUSDT" / "1h"
    processed.mkdir(parents=True, exist_ok=True)
    merged = tmp_path / "merged"

    # 2025-01-01 00:00:00 and 01:00:00 in nanoseconds
    row1 = [1735689600000000000, 1, 2, 0.5, 1.5, 10, 0, 10, 1, 5, 5, 0]
    row2 = [1735693200000000000, 2, 3, 1.5, 2.5, 20, 0, 20, 1, 10, 10, 0]
    pd.DataFrame([row1, row2]).to_csv(processed / "a.csv", index=False, header=False)

    merger = DataMerger(str(tmp_path / "processed"), str(merged))
    out = merger.merge_symbol_interval("BTCUSDT", "1h")
    assert int(out["open_time"].min()) == 1735689600000
    assert int(out["open_time"].max()) == 1735693200000
