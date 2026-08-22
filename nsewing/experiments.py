"""Experiment Lab — grid-search harness for the Supertrend strategy.

Sweeps combinations of universe × timeframe × stop/target bracket × indicator
preset × gates(on/off), running EACH through the SAME funnel + ₹ portfolio
engine the Backtest Lab uses (``pipeline.build_candidates`` +
``portfolio.run_portfolio``), so results are consistent and already verified for
look-ahead. Produces one metrics row per combination and ranks them by a
risk-adjusted Calmar score (CAGR ÷ |max drawdown|) with a minimum-trades floor.

Honesty note on gates: runs with the fundamental/growth gate ON use *today's*
yfinance fundamentals across all history (yfinance exposes no point-in-time
financials), so those rows are flagged ``fundamental_leak=True``. Gates-OFF runs
are 100% point-in-time (technicals + sector rotation are causal) and are the
clean benchmark.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config
from . import pipeline as pipe
from . import portfolio as pf
from . import sectors as sectmod
from . import data as datamod
from .strategies import get_strategy

STRATEGY = "Supertrend Sector Momentum"

# --------------------------------------------------------------------------
# Indicator presets — named filter-sets toggling the strategy's use_* params.
# Each is a dict of overrides applied on top of the 'relaxed' defaults.
# --------------------------------------------------------------------------
INDICATOR_PRESETS: dict[str, dict] = {
    "Baseline (ST+EMA+RSI+ADX+Vol+OBV+BO)": {},  # relaxed defaults as-is
    "Trend-only (ST+EMA)": {
        "use_rsi": False, "use_adx": False, "use_volume": False,
        "use_obv": False, "use_breakout": False},
    "+EMA200 long trend": {"use_ema200": True},
    "+MACD momentum": {"use_macd": True},
    "Momentum (RSI+ADX+MACD)": {
        "use_ema_structure": True, "use_macd": True,
        "use_volume": False, "use_obv": False, "use_breakout": False},
    "Volume+OBV breakout": {
        "use_rsi": False, "use_adx": False,
        "use_volume": True, "use_obv": True, "use_breakout": True},
    "+Weekly Supertrend": {"use_weekly": True},
    "Core Supertrend only": {
        "use_ema_structure": False, "use_rsi": False, "use_adx": False,
        "use_volume": False, "use_obv": False, "use_breakout": False},
}

# Stop / target brackets: (pct_stop, target_multiple). target% = stop% × mult.
STOP_BRACKETS: dict[str, tuple[float, float]] = {
    "5% / 10%": (0.05, 2.0),
    "8% / 16%": (0.08, 2.0),
    "10% / 20%": (0.10, 2.0),
    "12% / 24%": (0.12, 2.0),
}

INTERVALS = ["1d", "1wk"]
UNIVERSE_NAMES = list(config.UNIVERSES.keys())
GATE_MODES = ["ON", "OFF"]

# Fixed, sane simulation defaults (not swept — kept constant so combos compare).
FIXED = dict(window_years=3.0, max_hold=60, max_positions=12, risk_pct=0.03)

# Balanced default axis selection (~ a few hundred combos). All overridable.
DEFAULT_AXES = dict(
    universes=["NIFTY Midcap 50", "NIFTY Smallcap 50", "Midcap/Smallcap 150"],
    intervals=["1d", "1wk"],
    brackets=list(STOP_BRACKETS.keys()),
    presets=list(INDICATOR_PRESETS.keys()),
    gates=["ON", "OFF"],
)


def count_combos(axes: dict) -> int:
    return (len(axes["universes"]) * len(axes["intervals"]) * len(axes["brackets"])
            * len(axes["presets"]) * len(axes["gates"]))


def build_grid(axes: dict) -> list[dict]:
    """Cartesian product of the selected axes -> list of combo dicts."""
    combos = []
    for uni, itv, brk, pre, gate in itertools.product(
            axes["universes"], axes["intervals"], axes["brackets"],
            axes["presets"], axes["gates"]):
        combos.append({
            "universe": uni, "interval": itv, "bracket": brk,
            "preset": pre, "gates": gate,
        })
    return combos


# --------------------------------------------------------------------------
# Caches so we don't refetch data / recompute sector history per combo.
# --------------------------------------------------------------------------
def _sector_hist_cache() -> dict:
    return {}


def _run_one(combo: dict, sec_cache: dict) -> dict:
    """Run a single combination through the funnel + portfolio; return a row."""
    uni_name = combo["universe"]
    universe = config.UNIVERSES[uni_name]
    interval = combo["interval"]
    pct_stop, tmult = STOP_BRACKETS[combo["bracket"]]
    overrides = INDICATOR_PRESETS[combo["preset"]]
    gates_on = combo["gates"] == "ON"

    sec_interval = interval if interval in ("1d", "1wk") else "1d"
    if sec_interval not in sec_cache:
        sec_cache[sec_interval] = sectmod.quadrant_history(sec_interval)
    sec_hist = sec_cache[sec_interval] if gates_on else None

    bench = None
    if interval in ("1d", "1wk"):
        b = datamod.get_ohlcv(config.BENCHMARK, interval)
        bench = b["Close"] if b is not None and not b.empty else None

    r = pipe.build_candidates(
        STRATEGY, universe, interval=interval, sensitivity="relaxed",
        window_years=FIXED["window_years"],
        apply_fundamentals=gates_on, apply_growth=gates_on, apply_sector=gates_on,
        long_only=True, pct_stop=pct_stop, pct_target_mult=tmult, **overrides)

    strat = get_strategy(STRATEGY, sensitivity="relaxed",
                         pct_stop=pct_stop, pct_target_mult=tmult, **overrides)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=FIXED["window_years"] * 365.25)
    bt_universe = r.get("fund_pass", r["allowed"]) if gates_on else r["allowed"]

    pres = pf.run_portfolio(
        strat, r["data_map"], bench_close=bench, capital=config.DEFAULT_CAPITAL,
        risk_pct=FIXED["risk_pct"], max_positions=FIXED["max_positions"],
        max_hold=FIXED["max_hold"], start_date=start,
        allowed_symbols=bt_universe, scores=r["scores"],
        symbol_sectors=r.get("sym2sec") if gates_on else None,
        sector_history=sec_hist)
    m = pres["metrics"]
    cagr = m["cagr"]
    dd = abs(m["max_dd"])
    calmar = (cagr / dd) if dd > 1e-9 else (cagr / 0.01 if cagr else 0.0)
    return {
        "universe": uni_name, "interval": interval, "bracket": combo["bracket"],
        "preset": combo["preset"], "gates": combo["gates"],
        "fundamental_leak": gates_on,          # ON => uses today's fundamentals
        "trades": m["trades"],
        "win_%": round(m["win_rate"] * 100, 1),
        "PF": round(m["profit_factor"], 2),
        "total_%": round(m["total_return"] * 100, 1),
        "CAGR_%": round(cagr * 100, 1),
        "max_dd_%": round(m["max_dd"] * 100, 1),
        "calmar": round(calmar, 2),
    }


def run_grid(grid: list[dict], progress_cb=None) -> pd.DataFrame:
    """Run every combo; return a long-form DataFrame (one row per combo).

    ``progress_cb(done, total)`` is called after each combo for a UI progress bar.
    """
    sec_cache = _sector_hist_cache()
    rows = []
    total = len(grid)
    for i, combo in enumerate(grid, 1):
        try:
            rows.append(_run_one(combo, sec_cache))
        except Exception as e:  # keep the grid going; record the failure
            rows.append({
                "universe": combo["universe"], "interval": combo["interval"],
                "bracket": combo["bracket"], "preset": combo["preset"],
                "gates": combo["gates"], "fundamental_leak": combo["gates"] == "ON",
                "trades": 0, "win_%": 0.0, "PF": 0.0, "total_%": 0.0,
                "CAGR_%": 0.0, "max_dd_%": 0.0, "calmar": 0.0, "error": str(e)[:80]})
        if progress_cb:
            progress_cb(i, total)
    return pd.DataFrame(rows)


def rank(df: pd.DataFrame, min_trades: int = 20) -> pd.DataFrame:
    """Rank by risk-adjusted Calmar (CAGR ÷ |max_dd|), floor on trade count.

    Combos with fewer than ``min_trades`` are kept but flagged ``thin=True`` and
    sorted below all non-thin combos so a lucky 3-trade run can't win.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["thin"] = out["trades"] < min_trades
    # Sort: non-thin first, then by calmar desc, then CAGR desc.
    out = out.sort_values(
        by=["thin", "calmar", "CAGR_%"], ascending=[True, False, False]
    ).reset_index(drop=True)
    return out
