"""Reusable candlestick / pattern helpers shared by the strategies.

Every helper returns a boolean Series aligned to the input index and uses only
data at or before each bar (no look-ahead). Indexing convention matches the
document's scanner notation:  [-1] = previous bar, [0] = current bar.
Here we express that with vectorised ``.shift`` operations.
"""

from __future__ import annotations

import pandas as pd


def is_green(df: pd.DataFrame) -> pd.Series:
    return df["Close"] > df["Open"]


def is_red(df: pd.DataFrame) -> pd.Series:
    return df["Close"] < df["Open"]


def breakout_candle(df: pd.DataFrame) -> pd.Series:
    """Bar makes a higher high than the prior bar and closes up (doc breakout bar)."""
    return (df["Close"] > df["Open"]) & (df["High"] > df["High"].shift(1))


def rejection_below(df: pd.DataFrame, ref_close: pd.Series, ref_open: pd.Series) -> pd.Series:
    """Current close falls back below the reference bar's body (trap reversal)."""
    return (df["Close"] < ref_close) & (df["Close"] < ref_open)


def bearish_rsi_divergence(df: pd.DataFrame, rsi_col: str = "RSI14") -> pd.Series:
    """Price higher high but RSI lower/equal high vs the prior bar (doc §6.2)."""
    price_hh = df["High"] > df["High"].shift(1)
    rsi_lh = df[rsi_col] <= df[rsi_col].shift(1)
    return price_hh & rsi_lh


def close_in_upper_pct(df: pd.DataFrame, pct: float = 0.20) -> pd.Series:
    """Close sits in the upper `pct` of the bar's range (strength)."""
    rng = (df["High"] - df["Low"]).replace(0, pd.NA)
    pos = (df["Close"] - df["Low"]) / rng
    return pos >= (1.0 - pct)


def close_in_upper_half(df: pd.DataFrame, frac: float = 0.5) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, pd.NA)
    pos = (df["Close"] - df["Low"]) / rng
    return pos >= frac


def tight_range(df: pd.DataFrame, lookback: int = 4, max_pct: float = 0.015) -> pd.Series:
    """Rolling high-low range over `lookback` bars <= max_pct of price (consolidation)."""
    roll_high = df["High"].rolling(lookback).max()
    roll_low = df["Low"].rolling(lookback).min()
    rng_pct = (roll_high - roll_low) / df["Close"]
    return rng_pct <= max_pct


def consolidation_high(df: pd.DataFrame, lookback: int = 4) -> pd.Series:
    """Highest high over the prior `lookback` bars (the box top to break)."""
    return df["High"].shift(1).rolling(lookback).max()


def consolidation_low(df: pd.DataFrame, lookback: int = 4) -> pd.Series:
    """Lowest low over the prior `lookback` bars (the box bottom / SL)."""
    return df["Low"].shift(1).rolling(lookback).min()


def swing_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    return df["Low"].rolling(lookback).min()


def swing_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    return df["High"].rolling(lookback).max()
