"""Fundamentals via yfinance .info, plus the document's fundamental gate.

yfinance .info is flaky (rate limits, occasional 404s), so every field access
is defensive and cached. The gate is a *sanity filter* for swing trading — it
avoids blow-ups and surprise-earnings gaps, not a deep valuation screen.
"""

from __future__ import annotations

import time

import pandas as pd

from . import config
from .data import _SESSION

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

_CACHE: dict[str, dict] = {}
_CACHE_TS: dict[str, float] = {}
_TTL = 6 * 3600  # 6 hours


def get_fundamentals(symbol: str) -> dict:
    """Return a dict of key fundamentals; empty-ish on failure."""
    now = time.time()
    if symbol in _CACHE and now - _CACHE_TS.get(symbol, 0) < _TTL:
        return _CACHE[symbol]

    out = {
        "symbol": symbol, "market_cap": None, "pe": None, "roe": None,
        "debt_to_equity": None, "div_yield": None, "earnings_date": None,
        "sector": None, "name": symbol,
    }
    if yf is not None:
        try:
            t = yf.Ticker(symbol, session=_SESSION)
            info = t.info or {}
            out["market_cap"] = info.get("marketCap")
            out["pe"] = info.get("trailingPE")
            roe = info.get("returnOnEquity")
            out["roe"] = roe
            dte = info.get("debtToEquity")
            # yfinance reports D/E as a percentage (e.g. 45.0 => 0.45).
            out["debt_to_equity"] = (dte / 100.0) if dte is not None else None
            out["div_yield"] = info.get("dividendYield")
            out["sector"] = info.get("sector")
            out["name"] = info.get("shortName") or symbol
            # Next earnings date (event risk).
            try:
                cal = t.calendar
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        out["earnings_date"] = pd.to_datetime(ed[0]).date() if isinstance(ed, (list, tuple)) else pd.to_datetime(ed).date()
            except Exception:
                pass
        except Exception:
            pass

    _CACHE[symbol] = out
    _CACHE_TS[symbol] = now
    return out


def passes_gate(f: dict) -> tuple[bool, list[str]]:
    """Doc §2.2 fundamental gate. Missing data is treated leniently (pass with
    a note) rather than excluding a stock purely for a missing field."""
    g = config.FUND_GATE
    reasons = []
    ok = True

    mc = f.get("market_cap")
    if mc is not None and mc < g["min_market_cap"]:
        ok = False
        reasons.append(f"mcap ₹{mc/1e7:,.0f}Cr < ₹{g['min_market_cap']/1e7:,.0f}Cr")

    dte = f.get("debt_to_equity")
    if dte is not None and dte > g["max_debt_to_equity"]:
        ok = False
        reasons.append(f"D/E {dte:.2f} > {g['max_debt_to_equity']}")

    return ok, reasons


def earnings_in_days(f: dict, horizon: int = 30) -> int | None:
    """Days until next earnings, or None. Negative/large means no event risk."""
    ed = f.get("earnings_date")
    if ed is None:
        return None
    try:
        delta = (pd.Timestamp(ed) - pd.Timestamp.now().normalize()).days
        return delta
    except Exception:
        return None


# ==========================================================================
# Quarter-on-quarter fundamental GROWTH score (no technicals).
# Uses yfinance quarterly income statement: Operating Income, EPS, Revenue.
# Score rewards CONSISTENT QoQ growth + a positive latest quarter.
# ==========================================================================
_GROWTH_CACHE: dict[str, dict] = {}
_GROWTH_TS: dict[str, float] = {}

# Row-name candidates in yfinance's quarterly income statement (order = priority).
_ROWS = {
    "operating_income": ["Operating Income", "Operating Revenue"],
    "eps": ["Diluted EPS", "Basic EPS"],
    "revenue": ["Total Revenue", "Operating Revenue"],
}
# Weights for the three metrics (Operating Income / EPS / Revenue).
GROWTH_WEIGHTS = {"operating_income": 0.40, "eps": 0.35, "revenue": 0.25}


def _pick_row(stmt: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for n in names:
        if n in stmt.index:
            return stmt.loc[n]
    return None


def _metric_score(series: pd.Series) -> tuple[float, dict]:
    """Score one metric's QoQ growth: consistency (how many of the last ~4 QoQ
    steps rose) blended with whether the latest quarter is up.

    yfinance columns are newest-first, so we reverse to chronological order.
    Returns (0..1 score, detail dict).
    """
    if series is None:
        return 0.0, {"available": False}
    vals = [v for v in series.tolist() if pd.notna(v)]
    vals = vals[::-1]  # chronological (oldest -> newest)
    if len(vals) < 2:
        return 0.0, {"available": False}
    deltas = []
    for a, b in zip(vals[:-1], vals[1:]):
        if a == 0:
            continue
        deltas.append((b - a) / abs(a))
    if not deltas:
        return 0.0, {"available": False}
    rose = sum(1 for d in deltas if d > 0)
    consistency = rose / len(deltas)                 # fraction of quarters that grew
    latest_up = 1.0 if deltas[-1] > 0 else 0.0
    latest_pct = deltas[-1] * 100
    score = 0.6 * consistency + 0.4 * latest_up      # steady growth weighted over one-offs
    return score, {"available": True, "consistency": round(consistency, 2),
                   "latest_qoq_pct": round(latest_pct, 1),
                   "quarters_up": f"{rose}/{len(deltas)}"}


def growth_score(symbol: str) -> dict:
    """Return {score: 0..100, passes: bool, detail: {...}} from QoQ growth of
    Operating Income, EPS, and Revenue. Purely fundamental — no price/technicals."""
    now = time.time()
    if symbol in _GROWTH_CACHE and now - _GROWTH_TS.get(symbol, 0) < _TTL:
        return _GROWTH_CACHE[symbol]

    out = {"score": 0.0, "passes": False, "detail": {}, "available": False}
    if yf is not None:
        try:
            t = yf.Ticker(symbol, session=_SESSION)
            stmt = t.quarterly_income_stmt
            if stmt is not None and not stmt.empty:
                weighted = 0.0
                detail = {}
                any_avail = False
                for key, names in _ROWS.items():
                    row = _pick_row(stmt, names)
                    s, d = _metric_score(row)
                    detail[key] = d
                    if d.get("available"):
                        any_avail = True
                        weighted += GROWTH_WEIGHTS[key] * s
                out["available"] = any_avail
                out["score"] = round(weighted * 100, 1)
                # Passes the hard filter if the weighted growth score clears 50
                # (i.e. more growth than not across the weighted metrics).
                out["passes"] = any_avail and out["score"] >= 50.0
                out["detail"] = detail
        except Exception:
            pass

    _GROWTH_CACHE[symbol] = out
    _GROWTH_TS[symbol] = now
    return out
