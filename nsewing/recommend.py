"""Live recommendations (NEW module).

For each ENABLED strategy, scan a universe for signals that fired on the most
recent bar(s) — i.e. actionable *today* — rank them, and bucket into
**Strong Buy / Buy**. Every recommendation carries:

  * entry, stop-loss, targets (T1/T2/T3) and reward:risk,
  * a 0-100 conviction score (reuses screener._score_row),
  * the indicator readings AT the signal (RSI, MACD, ADX, EMA structure,
    volume ratio, Supertrend direction, Williams %R if present),
  * "probable hold days" — the strategy's historical average holding period on
    this universe (measured by a quick backtest), as a realistic expectation,
  * fundamentals gate + earnings-in-N-days flag.

This is decision-support on the strategies as designed — it does NOT touch the
strategies, backtester, or the Momentum Lab. It reuses screener/indicators/
backtest so the numbers agree with the rest of the app.

Buckets (by conviction score): Strong Buy >= 70, Buy >= 50, else Watch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt
from . import config
from . import data as datamod
from . import fundamentals as fund
from . import indicators as ind
from .screener import _score_row
from .strategies import enabled_strategies, get_strategy

STRONG_BUY, BUY = 70.0, 50.0


def _bucket(score: float) -> str:
    if score >= STRONG_BUY:
        return "STRONG BUY"
    if score >= BUY:
        return "BUY"
    return "WATCH"


def _avg_hold_days(strat, data_map, bench, max_hold: int) -> float:
    """Strategy's historical average holding period (calendar-ish, in bars) on
    this universe — used as 'probable hold days' guidance. Cheap: reuses the
    single-symbol R-multiple backtester over all symbols."""
    try:
        res = bt.run_backtest(strat, data_map, bench_close=bench,
                              max_hold_override=max_hold)
        m = res.get("metrics", {})
        n = m.get("trades", 0)
        if n and m.get("avg_bars_held"):
            return float(m["avg_bars_held"])
    except Exception:
        pass
    return float("nan")


def recommend_for_strategy(
    strategy_name: str,
    symbols: list[str],
    interval: str = "1d",
    sensitivity: str = "relaxed",
    recent_bars: int = 3,
    horizon: int = 30,
    with_fundamentals: bool = True,
) -> pd.DataFrame:
    """Ranked recommendation table for ONE strategy (fresh signals only)."""
    strat = get_strategy(strategy_name, sensitivity=sensitivity)
    bench = None
    if interval in ("1d", "1wk"):
        b = datamod.get_ohlcv(config.BENCHMARK, interval)
        bench = b["Close"] if not b.empty else None

    # Pull data once; reuse for signals, indicator snapshot, and avg-hold calc.
    data_map = {}
    for sym in symbols:
        df = datamod.get_ohlcv(sym, interval)
        if df is not None and not df.empty and len(df) >= 60:
            data_map[sym] = df

    avg_hold = _avg_hold_days(strat, data_map, bench, int(getattr(strat, "max_hold", 20)))

    rows = []
    for sym, df in data_map.items():
        sigs = strat.generate_signals(df, bench_close=bench)
        if sigs.empty:
            continue
        cutoff = df.index[-recent_bars]
        fresh = sigs[sigs["date"] >= cutoff]
        if fresh.empty:
            continue
        d = ind.enrich(df, bench_close=bench)
        # Williams %R column exists only if the strategy computed it in prepare().
        wr = None
        try:
            dd = strat.prepare(df, bench_close=bench)
            wr = dd["WR"] if "WR" in dd.columns else None
        except Exception:
            wr = None

        for _, s in fresh.iterrows():
            last = d.loc[s["date"]]
            entry, stop = s["entry"], s["stop"]
            risk = abs(entry - stop)
            reward = abs(entry - s["t1"])
            rr = reward / risk if risk > 0 else 0.0
            vr = float(last.get("VOL_RATIO", np.nan))
            score = _score_row(last, s["side"], rr, vr if np.isfinite(vr) else 1.0)
            row = {
                "signal_date": s["date"].date(),
                "symbol": sym,
                "recommendation": _bucket(score),
                "score": score,
                "side": s["side"],
                "entry": round(entry, 2),
                "stop_loss": round(stop, 2),
                "sl_%": round((stop / entry - 1) * 100, 2),
                "target1": round(s["t1"], 2),
                "target2": round(s["t2"], 2),
                "target3": round(s["t3"], 2),
                "t1_%": round((s["t1"] / entry - 1) * 100, 2),
                "R:R(T1)": round(rr, 2),
                "probable_hold_days": round(avg_hold, 0) if np.isfinite(avg_hold) else None,
                "max_hold_days": int(getattr(strat, "max_hold", 20)),
                # Indicator readings at the signal bar.
                "RSI14": round(float(last.get("RSI14", np.nan)), 1),
                "ADX14": round(float(last.get("ADX14", np.nan)), 1),
                "MACD>Signal": bool(last.get("MACD", 0) > last.get("MACD_SIGNAL", 0)),
                "Close>EMA50": bool(last.get("Close", 0) > last.get("EMA50", 0)),
                "Close>EMA200": bool(last.get("Close", 0) > last.get("EMA200", 0)),
                "Supertrend": ("up" if last.get("ST_DIR", 0) == 1 else "down"),
                "vol_ratio": round(vr, 2) if np.isfinite(vr) else None,
            }
            if wr is not None and s["date"] in wr.index:
                row["Williams%R"] = round(float(wr.loc[s["date"]]), 1)
            if with_fundamentals:
                f = fund.get_fundamentals(sym)
                ok, reasons = fund.passes_gate(f)
                ed = fund.earnings_in_days(f, horizon)
                row["fund_gate"] = "PASS" if ok else "FAIL"
                row["earnings_in_days"] = ed
                row["earnings_flag"] = ("⚠" if (ed is not None and 0 <= ed <= horizon) else "")
                row["sector"] = f.get("sector")
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Rank: Strong Buy first, then by score desc.
    order = {"STRONG BUY": 0, "BUY": 1, "WATCH": 2}
    out["_ord"] = out["recommendation"].map(order)
    out = out.sort_values(["_ord", "score"], ascending=[True, False]) \
             .drop(columns="_ord").reset_index(drop=True)
    return out


def recommend_all(
    symbols: list[str],
    interval: str = "1d",
    sensitivity: str = "relaxed",
    recent_bars: int = 3,
    horizon: int = 30,
    with_fundamentals: bool = True,
    strategies: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every enabled strategy; return {strategy_name: recommendation table}."""
    names = strategies or enabled_strategies()
    return {
        name: recommend_for_strategy(
            name, symbols, interval=interval, sensitivity=sensitivity,
            recent_bars=recent_bars, horizon=horizon,
            with_fundamentals=with_fundamentals)
        for name in names
    }


def save_csv(df: pd.DataFrame, strategy_name: str, universe_name: str,
             out_dir: str | None = None) -> str:
    """Save a recommendation table to a dated CSV under ``recommendations/``.

    Filename encodes the run date, strategy and universe so a daily history
    accumulates (e.g. recommendations/2026-08-21__Williams-R__Midcap-50.csv).
    Returns the path written."""
    import os
    out_dir = out_dir or os.path.join(config.PROJECT_DIR, "recommendations")
    os.makedirs(out_dir, exist_ok=True)
    # Date is taken from the data (max signal_date) to avoid Date.now in tests;
    # falls back to today's date string built from the frame if present.
    date_str = str(df["signal_date"].max()) if (df is not None and not df.empty
                                                 and "signal_date" in df) else "latest"
    safe_strat = strategy_name.replace(" ", "-").replace("%", "R").replace("/", "-")
    safe_uni = universe_name.replace(" ", "-").replace("/", "-")
    path = os.path.join(out_dir, f"{date_str}__{safe_strat}__{safe_uni}.csv")
    df.to_csv(path, index=False)
    return path
