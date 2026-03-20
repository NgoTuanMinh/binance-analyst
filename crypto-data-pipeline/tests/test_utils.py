"""Tests for utility helpers."""

import pytest

from src.utils import get_interval_minutes


def test_get_interval_minutes_valid() -> None:
    assert get_interval_minutes("15m") == 15
    assert get_interval_minutes("1h") == 60
    assert get_interval_minutes("4h") == 240


def test_get_interval_minutes_invalid() -> None:
    with pytest.raises(ValueError):
        get_interval_minutes("1w")
