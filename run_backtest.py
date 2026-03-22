#!/usr/bin/env python3
"""Backwards-compatible entry: delegates to ``scripts/run_backtest.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

runpy.run_path(str(_ROOT / "scripts" / "run_backtest.py"), run_name="__main__")
