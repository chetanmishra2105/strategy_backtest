"""Daily screener — run a strategy's scanner across a universe and rank the
fresh setups (signals firing on the most recent bar) into a candidate table.

Each candidate carries entry / stop / targets, reward:risk, volume ratio,
RSI/MACD state, a fundamental-gate flag, an earnings-in-N-days event flag, and
a composite 0-100 score for ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as datamod
from . import fundamentals as fund
from . import indicators as ind
from .strategies import get_strategy


def _score_row(d_last: pd.Series, side: str, rr: float, vol_ratio: float) -> float:
    """Composite 0-100 blend of trend/RS, momentum, volume, and R:R."""
    score = 0.0
    # Volume conviction (up to 30).
    score += min(vol_ratio / 3.0, 1.0) * 30
    # Reward:risk (up to 25), capped at 4R.
    score += min(rr / 4.0, 1.0) * 25
    # Momentum alignment (up to 20).
    rsi = d_last.get("RSI14", 50)
    if side == "long":
        score += np.clip((rsi - 40) / 40, 0, 1) * 20
    else:
        score += np.clip((60 - rsi) / 40, 0, 1) * 20
    # Trend/relative-strength (up to 25).
    rs = d_last.get("RS55", 1.0)
    if not np.isfinite(rs):
        rs = 1.0
    if side == "long":
        score += np.clip((rs - 0.9) / 0.4, 0, 1) * 25
    else:
        score += np.clip((1.1 - rs) / 0.4, 0, 1) * 25
    return float(round(score, 1))


def scan(
    strategy_name: str,
    symbols: list[str],
    interval: str = "1d",
    sensitivity: str = "relaxed",
    recent_bars: int = 3,
    with_fundamentals: bool = True,
    horizon: int = 30,
) -> pd.DataFrame:
    """Return a ranked DataFrame of fresh candidates.

    ``recent_bars`` = how many of the latest bars count as "actionable now".
    """
    strat = get_strategy(strategy_name, sensitivity=sensitivity)
    bench = datamod.get_ohlcv("^NSEI", interval).get("Close") \
        if interval in ("1d", "1wk") else None

    rows = []
    for sym in symbols:
        df = datamod.get_ohlcv(sym, interval)
        if df is None or df.empty or len(df) < 60:
            continue
        sigs = strat.generate_signals(df, bench_close=bench)
        if sigs.empty:
            continue
        cutoff = df.index[-recent_bars]
        fresh = sigs[sigs["date"] >= cutoff]
        if fresh.empty:
            continue

        d = ind.enrich(df, bench_close=bench)
        for _, s in fresh.iterrows():
            last = d.loc[s["date"]]
            entry, stop = s["entry"], s["stop"]
            risk = abs(entry - stop)
            reward = abs(entry - s["t1"])
            rr = reward / risk if risk > 0 else 0.0
            vol_ratio = float(last.get("VOL_RATIO", np.nan))
            row = {
                "symbol": sym, "strategy": strategy_name, "side": s["side"],
                "signal_date": s["date"].date(),
                "entry": round(entry, 2), "stop": round(stop, 2),
                "t1": round(s["t1"], 2), "t2": round(s["t2"], 2), "t3": round(s["t3"], 2),
                "R:R(T1)": round(rr, 2),
                "vol_ratio": round(vol_ratio, 2) if np.isfinite(vol_ratio) else None,
                "RSI": round(float(last.get("RSI14", np.nan)), 1),
                "score": _score_row(last, s["side"], rr, vol_ratio if np.isfinite(vol_ratio) else 1.0),
            }
            if with_fundamentals:
                f = fund.get_fundamentals(sym)
                ok, reasons = fund.passes_gate(f)
                ed = fund.earnings_in_days(f, horizon)
                row["fund_gate"] = "PASS" if ok else "FAIL"
                row["gate_notes"] = "; ".join(reasons)
                row["earnings_in_days"] = ed
                row["earnings_flag"] = ("⚠ earnings" if (ed is not None and 0 <= ed <= horizon) else "")
                row["sector"] = f.get("sector")
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return out
