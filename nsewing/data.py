"""Data layer: yfinance OHLCV fetch with an on-disk parquet cache.

yfinance history limits (important for swing design):
  * 1d / 1wk : many years available  -> the backbone for swing decisions
  * 1h       : ~730 days max         -> entry timing / short backtests only

Cache files live in ``config.CACHE_DIR`` keyed by symbol+interval.
"""

from __future__ import annotations

import os
import time

import pandas as pd

from . import config

try:
    import yfinance as yf
except Exception:  # pragma: no cover - import guard for early setup
    yf = None


def _make_session():
    """Build a curl_cffi session that works behind TLS-intercepting proxies.

    yfinance >=1.x uses curl_cffi internally. We hand it a session with the
    SSL/verify settings from config so the corporate-CA case is handled in one
    place. Returns None if curl_cffi is unavailable (yfinance falls back).
    """
    try:
        from curl_cffi import requests as cr
    except Exception:
        return None
    kwargs = {"impersonate": config.IMPERSONATE}
    if config.CA_BUNDLE:
        kwargs["verify"] = config.CA_BUNDLE      # secure: corporate root CA
    else:
        kwargs["verify"] = config.SSL_VERIFY     # False by default (intercepted TLS)
    return cr.Session(**kwargs)


_SESSION = _make_session()


# Sensible default look-back per interval (yfinance-friendly).
_DEFAULT_PERIOD = {"1d": "10y", "1wk": "10y", "1h": "720d"}

# Columns we standardise on.
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(symbol: str, interval: str) -> str:
    safe = symbol.replace("^", "_idx_").replace("&", "_and_").replace(".", "_")
    return os.path.join(config.CACHE_DIR, f"{safe}__{interval}.parquet")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns a MultiIndex (single-ticker download)."""
    if isinstance(df.columns, pd.MultiIndex):
        # Take the price-field level (first level) and drop the ticker level.
        df.columns = df.columns.get_level_values(0)
    return df


def get_ohlcv(
    symbol: str,
    interval: str = "1d",
    period: str | None = None,
    refresh: bool = False,
    max_age_hours: float = 12.0,
) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame indexed by datetime.

    Uses the parquet cache unless ``refresh`` is True or the cache is older
    than ``max_age_hours``.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed; run pip install -r requirements.txt")

    interval = interval.lower()
    period = period or _DEFAULT_PERIOD.get(interval, "5y")
    path = _cache_path(symbol, interval)

    if not refresh and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h <= max_age_hours:
            try:
                cached = pd.read_parquet(path)
                if not cached.empty:
                    return cached
            except Exception:
                pass  # fall through to re-fetch

    try:
        ticker = yf.Ticker(symbol, session=_SESSION)
        df = ticker.history(interval=interval, period=period, auto_adjust=False)
    except Exception:
        df = None
    if df is None or df.empty:
        # Return cache if we have any, else an empty frame.
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame(columns=_OHLCV)

    df = _flatten_columns(df)
    keep = [c for c in _OHLCV if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])

    try:
        df.to_parquet(path)
    except Exception:
        pass  # cache write is best-effort

    return df


def get_many(
    symbols: list[str],
    interval: str = "1d",
    period: str | None = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch several symbols; returns {symbol: DataFrame}. Skips empties."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = get_ohlcv(sym, interval=interval, period=period, refresh=refresh)
            if df is not None and not df.empty:
                out[sym] = df
        except Exception:
            continue
    return out


def get_vix(refresh: bool = False) -> pd.DataFrame:
    """India VIX daily series."""
    return get_ohlcv(config.VIX_TICKER, interval="1d", period="5y", refresh=refresh)


def latest_vix(refresh: bool = False) -> float | None:
    """Most recent India VIX close, or None if unavailable."""
    v = get_vix(refresh=refresh)
    if v is None or v.empty:
        return None
    return float(v["Close"].iloc[-1])
