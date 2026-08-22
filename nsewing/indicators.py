"""Technical indicators.

Prefers the ``ta`` library where it is solid, and hand-rolls the few things
``ta`` handles awkwardly (ATR-based helpers, Supertrend, volume MA). Every
function is pure: it takes an OHLCV DataFrame and returns a Series/DataFrame
aligned to the input index. No look-ahead — all use only past/current data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    _HAS_TA = True
except Exception:  # pragma: no cover
    _HAS_TA = False


# --------------------------------------------------------------------------
# Fallback implementations (used if `ta` is unavailable)
# --------------------------------------------------------------------------
def _rsi_fallback(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


# --------------------------------------------------------------------------
# Public indicator API
# --------------------------------------------------------------------------
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    if _HAS_TA:
        return RSIIndicator(close=close, window=window, fillna=False).rsi()
    return _rsi_fallback(close, window)


def ema(close: pd.Series, window: int) -> pd.Series:
    if _HAS_TA:
        return EMAIndicator(close=close, window=window, fillna=False).ema_indicator()
    return _ema(close, window)


def sma(close: pd.Series, window: int) -> pd.Series:
    if _HAS_TA:
        return SMAIndicator(close=close, window=window, fillna=False).sma_indicator()
    return close.rolling(window).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    if _HAS_TA:
        m = MACD(close=close, window_slow=slow, window_fast=fast,
                 window_sign=signal, fillna=False)
        return m.macd(), m.macd_signal(), m.macd_diff()
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, sig, macd_line - sig


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if _HAS_TA:
        return AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"],
                                window=window, fillna=False).average_true_range()
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # ta's ADX indexes into the series and errors on very short inputs; guard it.
    if _HAS_TA and len(df) > window * 2:
        try:
            return ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"],
                                window=window, fillna=False).adx()
        except Exception:
            pass
    return pd.Series(np.nan, index=df.index)


def bollinger(close: pd.Series, window: int = 20, dev: float = 2.0):
    """Return (upper, mid, lower)."""
    if _HAS_TA:
        bb = BollingerBands(close=close, window=window, window_dev=dev, fillna=False)
        return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return mid + dev * sd, mid, mid - dev * sd


def stochastic(df: pd.DataFrame, window: int = 14, smooth: int = 3):
    """Return (%K, %D)."""
    if _HAS_TA:
        so = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"],
                                  window=window, smooth_window=smooth, fillna=False)
        return so.stoch(), so.stoch_signal()
    low_min = df["Low"].rolling(window).min()
    high_max = df["High"].rolling(window).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    return k, k.rolling(smooth).mean()


def vol_ma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume moving average (the doc's VolMA20 baseline)."""
    return volume.rolling(window).mean()


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R: -100 * (HighestHigh - Close) / (HighestHigh - LowestLow).

    Oscillates between 0 (top of range) and -100 (bottom). Values below -80/-90
    indicate deeply oversold. Uses only past/current bars (no look-ahead).
    """
    highest = df["High"].rolling(window).max()
    lowest = df["Low"].rolling(window).min()
    rng = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - df["Close"]) / rng


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Supertrend line + direction (the standard ATR 10, multiplier 3 setting).

    Returns (line, direction) as two Series aligned to the input index:
      * line      : the trailing Supertrend stop level.
      * direction : +1 when price is ABOVE the line (uptrend / "green"),
                    -1 when price is BELOW it (downtrend / "red").

    Classic Wilder-style construction with band "locking": bands only tighten in
    the direction of the trend and flip when price closes across the line. Uses
    only past/current bars (each bar depends on the prior bar's line) — no
    look-ahead.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    atr_s = atr(df, period)
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr_s
    lower = hl2 - multiplier * atr_s

    n = len(df)
    line = np.full(n, np.nan)
    direction = np.full(n, 1)  # +1 up, -1 down
    up_arr = upper.to_numpy()
    lo_arr = lower.to_numpy()
    close_arr = close.to_numpy()

    final_upper = np.nan
    final_lower = np.nan
    for i in range(n):
        cu, cl, cc = up_arr[i], lo_arr[i], close_arr[i]
        if not np.isfinite(cu) or not np.isfinite(cl) or not np.isfinite(cc):
            # ATR not warmed up yet — carry raw bands, leave line NaN.
            final_upper, final_lower = cu, cl
            continue
        if i == 0 or not np.isfinite(line[i - 1]):
            final_upper, final_lower = cu, cl
            direction[i] = 1 if cc >= cl else -1
            line[i] = cl if direction[i] == 1 else cu
            continue
        # Lock bands: only move in the trend's favour.
        final_upper = cu if (cu < final_upper or close_arr[i - 1] > final_upper) else final_upper
        final_lower = cl if (cl > final_lower or close_arr[i - 1] < final_lower) else final_lower
        prev_dir = direction[i - 1]
        if prev_dir == 1:
            direction[i] = -1 if cc < final_lower else 1
        else:
            direction[i] = 1 if cc > final_upper else -1
        line[i] = final_lower if direction[i] == 1 else final_upper

    return (pd.Series(line, index=df.index),
            pd.Series(direction, index=df.index, dtype="int64"))


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume, added on up-closes, subtracted on
    down-closes. A rising OBV signals accumulation (buyers in control)."""
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()


def rel_strength(close: pd.Series, bench_close: pd.Series, window: int = 55) -> pd.Series:
    """Relative strength = stock return / benchmark return over `window` bars.

    >1 means the stock outperformed the benchmark over the window.
    """
    bench = bench_close.reindex(close.index).ffill()
    stock_ret = close / close.shift(window)
    bench_ret = bench / bench.shift(window)
    return stock_ret / bench_ret


def enrich(df: pd.DataFrame, bench_close: pd.Series | None = None) -> pd.DataFrame:
    """Attach the common indicator set used across strategies/screener."""
    out = df.copy()
    close = out["Close"]
    out["RSI14"] = rsi(close, 14)
    macd_line, macd_sig, macd_hist = macd(close)
    out["MACD"] = macd_line
    out["MACD_SIGNAL"] = macd_sig
    out["MACD_HIST"] = macd_hist
    out["EMA20"] = ema(close, 20)
    out["EMA50"] = ema(close, 50)
    out["EMA200"] = ema(close, 200)
    out["SMA20"] = sma(close, 20)
    out["ATR14"] = atr(out, 14)
    out["ADX14"] = adx(out, 14)
    out["VOLMA20"] = vol_ma(out["Volume"], 20)
    out["VOL_RATIO"] = out["Volume"] / out["VOLMA20"]
    st_line, st_dir = supertrend(out, 10, 3.0)
    out["SUPERT"], out["ST_DIR"] = st_line, st_dir
    out["OBV"] = obv(close, out["Volume"])
    bb_u, bb_m, bb_l = bollinger(close, 20, 2.0)
    out["BB_UPPER"], out["BB_MID"], out["BB_LOWER"] = bb_u, bb_m, bb_l
    k, d = stochastic(out, 14, 3)
    out["STOCH_K"], out["STOCH_D"] = k, d
    if bench_close is not None:
        out["RS55"] = rel_strength(close, bench_close, 55)
    return out
