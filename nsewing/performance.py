"""Multi-window performance comparison across all strategies.

For each trailing window (e.g. last 30/60/90/120/180 days) and each strategy,
run the backtester restricting entries to that window and collect headline
stats. Returns tidy DataFrames the performance page can pivot/plot.

Note on short windows: annualised CAGR over 30 days is statistically noisy, so
we report BOTH total return over the window and its annualised equivalent, and
always show the trade count so thin samples are obvious.
"""

from __future__ import annotations

import pandas as pd

from . import backtest, config
from . import data as datamod
from . import pipeline as pipe
from . import portfolio as pf
from . import sectors as sectmod
from .strategies import enabled_strategies, get_strategy

# Swing-appropriate windows in DAYS (the strategy holds 30-90 days, so windows
# shorter than ~180d are dominated by unfinished trades and pure noise).
DEFAULT_WINDOWS = [180, 365, 730, 1095]


def _annualise(total_return: float, days: int) -> float:
    if days <= 0:
        return 0.0
    return (1.0 + total_return) ** (365.25 / days) - 1.0


def run_matrix(
    universe: list[str],
    interval: str = "1d",
    windows: list[int] = None,
    sensitivity: str = "relaxed",
    risk_pct: float = None,
    strategies: list[str] = None,
    today: pd.Timestamp = None,
    *,
    apply_fundamentals: bool = True,
    apply_growth: bool = True,
    apply_sector: bool = True,
    max_positions: int = 12,
    max_hold: int = 60,
    pct_stop: float | None = 0.10,
    pct_target_mult: float = 2.0,
    capital: float = None,
) -> pd.DataFrame:
    """Return a long-form DataFrame: one row per (strategy, window).

    This runs the SAME engine as the Backtest Lab — the layered funnel
    (fundamentals + QoQ-growth + point-in-time sector gate) followed by the ₹
    portfolio simulation with the universal 10% position cap — so the numbers
    match reality instead of the raw, unfiltered-signal proxy used before.
    Each window is converted to ``window_years`` and drives both the funnel and
    the simulation, exactly like the Backtest Lab's single window.
    """
    windows = windows or DEFAULT_WINDOWS
    risk_pct = risk_pct if risk_pct is not None else config.DEFAULT_RISK_PCT
    capital = capital if capital is not None else config.DEFAULT_CAPITAL
    strat_names = strategies or enabled_strategies()
    today = today or pd.Timestamp.today().normalize()

    bench = None
    if interval in ("1d", "1wk"):
        b = datamod.get_ohlcv(config.BENCHMARK, interval)
        bench = b["Close"] if not b.empty else None
    sec_interval = interval if interval in ("1d", "1wk") else "1d"
    sec_hist = sectmod.quadrant_history(sec_interval) if apply_sector else None

    rows = []
    for name in strat_names:
        for w in windows:
            years = w / 365.25
            start = today - pd.Timedelta(days=w)
            r = pipe.build_candidates(
                name, universe, interval=interval, sensitivity=sensitivity,
                window_years=years, apply_fundamentals=apply_fundamentals,
                apply_sector=apply_sector, apply_growth=apply_growth,
                long_only=True, pct_stop=pct_stop, pct_target_mult=pct_target_mult)
            strat = get_strategy(name, sensitivity=sensitivity,
                                 pct_stop=pct_stop, pct_target_mult=pct_target_mult)
            bt_universe = r.get("fund_pass", r["allowed"]) if apply_sector else r["allowed"]
            pres = pf.run_portfolio(
                strat, r["data_map"], bench_close=bench, capital=capital,
                risk_pct=risk_pct, max_positions=max_positions, max_hold=max_hold,
                start_date=start, allowed_symbols=bt_universe, scores=r["scores"],
                symbol_sectors=r.get("sym2sec") if apply_sector else None,
                sector_history=sec_hist)
            m = pres["metrics"]
            rows.append({
                "strategy": name,
                "window_days": w,
                "trades": m["trades"],
                "win_rate_%": round(m["win_rate"] * 100, 1),
                "total_return_%": round(m["total_return"] * 100, 2),
                # CAGR now comes from the portfolio's own (window-trimmed) equity
                # curve — not a naive extrapolation of a short window.
                "annualised_CAGR_%": round(m["cagr"] * 100, 1),
                "profit_factor": round(m["profit_factor"], 2),
                "expectancy_%": round(m["expectancy"] * 100, 2),
                "max_dd_%": round(m["max_dd"] * 100, 1),
            })
    return pd.DataFrame(rows)


def strategy_vs_indices(
    equity: pd.Series,
    capital: float,
    indices: dict[str, str] = None,
) -> pd.DataFrame:
    """Return a normalised (growth-of-1) frame comparing the strategy equity to
    each broad-market index over the strategy's date range.

    Columns: 'Strategy' + one per index (buy & hold). Index = dates.
    """
    indices = indices or config.COMPARE_INDICES
    if equity is None or len(equity) < 2:
        return pd.DataFrame()
    start, end = equity.index[0], equity.index[-1]
    cols = {"Strategy": equity / float(equity.iloc[0])}
    for name, ticker in indices.items():
        df = datamod.get_ohlcv(ticker, "1d")
        if df is None or df.empty:
            continue
        c = df["Close"]
        # tz-align: strategy index may be tz-aware.
        try:
            c = c.loc[(c.index >= start) & (c.index <= end)]
        except Exception:
            c.index = pd.to_datetime(c.index)
            c = c.loc[(c.index >= start) & (c.index <= end)]
        if c.empty:
            continue
        cols[name] = c / float(c.iloc[0])
    out = pd.DataFrame(cols).ffill()
    return out


def compare_returns_table(comp: pd.DataFrame) -> pd.DataFrame:
    """Total return % per series from a normalised growth frame."""
    if comp is None or comp.empty:
        return pd.DataFrame()
    rows = [{"series": c, "total_return_%": round((comp[c].dropna().iloc[-1] - 1) * 100, 1)}
            for c in comp.columns]
    return pd.DataFrame(rows).sort_values("total_return_%", ascending=False).reset_index(drop=True)


def pivot_metric(matrix: pd.DataFrame, metric: str = "total_return_%") -> pd.DataFrame:
    """Pivot to strategy (rows) x window (cols) for a chosen metric."""
    if matrix.empty:
        return matrix
    p = matrix.pivot(index="strategy", columns="window_days", values=metric)
    p.columns = [f"{c}d" for c in p.columns]
    return p
