"""'Why this stock?' — a plain-language, per-condition breakdown of why a given
stock triggered a strategy's signal on its latest signal bar.

Re-evaluates each rule from the strategy's ``_entries`` logic on the signal bar
and reports pass/value/threshold, then adds the computed trade levels, R:R, ATR,
a fundamentals snapshot, and the sector-rotation quadrant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import data as datamod
from . import fundamentals as fund
from . import indicators as ind
from . import sectors as sectmod
from .strategies import Strategy, get_strategy


def _fmt(x, nd=2):
    try:
        return round(float(x), nd)
    except Exception:
        return x


def _check(label, ok, value, threshold):
    return {"check": label, "pass": bool(ok), "value": value, "threshold": threshold}


def _bull_trap_checks(d, i, p):
    prev = i - 1
    checks = []
    checks.append(_check("Prior bar was a breakout candle (higher high, closed up)",
                         d["Close"].iloc[prev] > d["Open"].iloc[prev] and d["High"].iloc[prev] > d["High"].iloc[prev-1],
                         None, None))
    checks.append(_check("Reversal bar closes back below prior body",
                         d["Close"].iloc[i] < d["Close"].iloc[prev] and d["Close"].iloc[i] < d["Open"].iloc[prev],
                         _fmt(d["Close"].iloc[i]), f"< {_fmt(d['Open'].iloc[prev])}"))
    vb = d["Volume"].iloc[prev] / d["VOLMA20"].iloc[prev]
    checks.append(_check("Breakout-bar volume", vb >= p["vol_break"], f"{_fmt(vb)}x", f">= {p['vol_break']}x"))
    vr = d["Volume"].iloc[i] / d["VOLMA20"].iloc[i]
    checks.append(_check("Reversal-bar volume", vr >= p["vol_rev"], f"{_fmt(vr)}x", f">= {p['vol_rev']}x"))
    if p.get("require_macd"):
        checks.append(_check("MACD below signal (bearish)", d["MACD"].iloc[i] < d["MACD_SIGNAL"].iloc[i],
                             _fmt(d["MACD"].iloc[i]), f"< {_fmt(d['MACD_SIGNAL'].iloc[i])}"))
    return checks


def _accumulation_checks(d, i, p):
    box = int(p["box"])
    checks = []
    roll_high = d["High"].iloc[i-box:i].max()
    roll_low = d["Low"].iloc[i-box:i].min()
    rng_pct = (roll_high - roll_low) / d["Close"].iloc[i]
    checks.append(_check(f"Prior {box}-bar range is tight", rng_pct <= p["range_pct"],
                         f"{_fmt(rng_pct*100)}%", f"<= {_fmt(p['range_pct']*100)}%"))
    vr = d["Volume"].iloc[i] / d["VOLMA20"].iloc[i]
    checks.append(_check("Breakout-bar volume expansion", vr >= p["vol_expand"],
                         f"{_fmt(vr)}x", f">= {p['vol_expand']}x"))
    checks.append(_check("Closed above the consolidation box", d["Close"].iloc[i] > roll_high,
                         _fmt(d["Close"].iloc[i]), f"> {_fmt(roll_high)}"))
    checks.append(_check("RSI > 50 (bullish momentum)", d["RSI14"].iloc[i] > 50,
                         _fmt(d["RSI14"].iloc[i]), "> 50"))
    checks.append(_check("MACD above signal", d["MACD"].iloc[i] > d["MACD_SIGNAL"].iloc[i],
                         _fmt(d["MACD"].iloc[i]), f"> {_fmt(d['MACD_SIGNAL'].iloc[i])}"))
    return checks


def _climax_checks(d, i, p):
    prev = i - 1
    checks = []
    cv = d["Volume"].iloc[prev] / d["VOLMA20"].iloc[prev]
    checks.append(_check("Prior bar: selling-climax volume", cv >= p["climax_vol"],
                         f"{_fmt(cv)}x", f">= {p['climax_vol']}x"))
    bv = d["Volume"].iloc[i] / d["VOLMA20"].iloc[i]
    checks.append(_check("Buyer-response volume", bv >= p["buy_vol"], f"{_fmt(bv)}x", f">= {p['buy_vol']}x"))
    checks.append(_check("Green recovery bar (close > prior close)", d["Close"].iloc[i] > d["Close"].iloc[prev],
                         _fmt(d["Close"].iloc[i]), f"> {_fmt(d['Close'].iloc[prev])}"))
    checks.append(_check("Prior RSI was oversold", d["RSI14"].iloc[prev] < p["rsi_max"],
                         _fmt(d["RSI14"].iloc[prev]), f"< {p['rsi_max']}"))
    return checks


def _williams_checks(d, i, p):
    checks = []
    checks.append(_check(f"Williams %R({p['period']}) crossed below the level",
                         d["WR"].iloc[i] < p["level"] and d["WR"].iloc[i-1] >= p["level"],
                         _fmt(d["WR"].iloc[i]), f"< {p['level']} (was {_fmt(d['WR'].iloc[i-1])})"))
    return checks


_CHECKERS = {
    "Bull Trap Reversal": _bull_trap_checks,
    "Accumulation Breakout": _accumulation_checks,
    "Volume Climax Reversal": _climax_checks,
    "Williams %R Oversold": _williams_checks,
}


def explain(symbol: str, strategy_name: str, interval: str = "1d",
            sensitivity: str = "relaxed") -> dict:
    """Return a structured explanation for the stock's latest signal."""
    strat: Strategy = get_strategy(strategy_name, sensitivity=sensitivity)
    df = datamod.get_ohlcv(symbol, interval)
    if df is None or df.empty:
        return {"error": f"No data for {symbol}."}
    bench = None
    if interval in ("1d", "1wk"):
        b = datamod.get_ohlcv(config.BENCHMARK, interval)
        bench = b["Close"] if not b.empty else None

    sigs = strat.generate_signals(df, bench_close=bench)
    if sigs.empty:
        return {"symbol": symbol, "strategy": strategy_name,
                "error": "This strategy has no signal on this stock/timeframe."}
    last = sigs.iloc[-1]
    # strat.prepare() runs ind.enrich() (base class) plus any strategy-specific
    # columns (e.g. WR for Williams %R), so all needed columns are present.
    d = strat.prepare(df, bench_close=bench)
    i = d.index.get_loc(last["date"])

    checker = _CHECKERS.get(strategy_name)
    checks = checker(d, i, strat.params) if checker else []

    risk = abs(last["entry"] - last["stop"])
    reward = abs(last["entry"] - last["t1"])
    rr = reward / risk if risk > 0 else 0.0

    # Sector quadrant.
    sym2sec = {s: sec for sec, syms in config.SECTORS.items() for s in syms}
    sec = sym2sec.get(symbol)
    rrg = sectmod.rrg_coordinates(interval if interval in ("1d", "1wk") else "1d")
    quad = dict(zip(rrg["sector"], rrg["quadrant"])).get(sec, "Unknown") if not rrg.empty else "Unknown"

    f = fund.get_fundamentals(symbol)
    ed = fund.earnings_in_days(f)

    return {
        "symbol": symbol, "strategy": strategy_name, "side": last["side"],
        "signal_date": last["date"].date(), "interval": interval,
        "checks": checks,
        "levels": {
            "entry": _fmt(last["entry"]), "stop": _fmt(last["stop"]),
            "t1": _fmt(last["t1"]), "t2": _fmt(last["t2"]), "t3": _fmt(last["t3"]),
            "risk_per_share": _fmt(risk), "reward_to_t1": _fmt(reward), "rr_t1": _fmt(rr),
            "atr14": _fmt(d["ATR14"].iloc[i]),
        },
        "sector": {"name": sec, "quadrant": quad,
                   "in_trend": quad in ("Leading", "Improving")},
        "fundamentals": {
            "name": f.get("name"), "pe": _fmt(f.get("pe")) if f.get("pe") else None,
            "mcap_cr": round(f["market_cap"] / 1e7, 0) if f.get("market_cap") else None,
            "debt_eq": _fmt(f.get("debt_to_equity")) if f.get("debt_to_equity") is not None else None,
            "earnings_in_days": ed,
        },
    }
