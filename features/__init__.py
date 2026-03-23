"""
Feature engineering for ML / analytics.

Layout:

- ``technical.py`` — indicators (multi-timeframe aware)
- ``price_action.py`` — patterns, S/R, swings, FVG
- ``smart_money.py`` — SMC-style proxies
- ``market_regime.py`` — regime labels
- ``target.py`` — forward targets
- ``pipeline.py`` — load ``data/merged`` Parquet, build matrix, export Parquet (parallel symbols)
"""

from .market_regime import (
    REGIME_HIGH_VOL,
    REGIME_QUIET,
    REGIME_RANGE,
    REGIME_RANGING,
    REGIME_TREND_BEAR,
    REGIME_TREND_BULL,
    REGIME_TRENDING,
    REGIME_VOLATILE,
    market_regime,
)
from .pipeline import (
    FeaturePipeline,
    build_and_write_symbol,
    build_feature_matrix,
    filter_symbols_with_merged_data,
    iter_with_progress,
    load_mtf_frames,
    normalize_interval_labels,
    resolve_symbols_argument,
    run_parallel,
)
from .price_action import (
    candle_patterns,
    fvg_detection,
    support_resistance,
    swing_points,
)
from .smart_money import smart_money_features
from .target import make_targets
from .technical import (
    TechnicalFeatures,
    momentum_indicators,
    trend_indicators,
    volatility_indicators,
    volume_indicators,
)

__all__ = [
    "TechnicalFeatures",
    "trend_indicators",
    "momentum_indicators",
    "volatility_indicators",
    "volume_indicators",
    "candle_patterns",
    "support_resistance",
    "swing_points",
    "fvg_detection",
    "smart_money_features",
    "market_regime",
    "make_targets",
    "REGIME_QUIET",
    "REGIME_RANGING",
    "REGIME_TRENDING",
    "REGIME_VOLATILE",
    "REGIME_RANGE",
    "REGIME_TREND_BULL",
    "REGIME_TREND_BEAR",
    "REGIME_HIGH_VOL",
    "FeaturePipeline",
    "iter_with_progress",
    "filter_symbols_with_merged_data",
    "resolve_symbols_argument",
    "normalize_interval_labels",
    "load_mtf_frames",
    "build_feature_matrix",
    "build_and_write_symbol",
    "run_parallel",
]
