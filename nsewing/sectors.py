"""Sector rotation — relative strength of each sector vs NIFTY.

We build sector composites from the constituent stocks in ``config.SECTORS``
(equal-weight normalised price), then compute:
  * a relative-strength heatmap over multiple trailing windows, and
  * RRG-style coordinates (RS-Ratio vs RS-Momentum) placing each sector in a
    Leading / Weakening / Lagging / Improving quadrant.

Building composites from constituents (rather than relying on sector-index
tickers, several of which yfinance does not expose reliably) keeps this robust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import data as datamod


def _base_symbol_to_sector() -> dict[str, str]:
    """Reverse of config.SECTORS: hardcoded large-cap symbol -> our sector."""
    out = {}
    for sec, syms in config.SECTORS.items():
        for s in syms:
            out[s] = sec
    return out


_BASE_S2S = None


def sector_for_symbol(symbol: str) -> str | None:
    """Best-effort sector for ANY symbol.

    1. Use the hardcoded large-cap map (config.SECTORS).
    2. Fall back to yfinance's GICS sector mapped via config.YF_SECTOR_MAP,
       so midcaps/smallcaps also get a sector (this fixes the 'None/Unknown'
       bug where the sector gate let everything through).
    """
    global _BASE_S2S
    if _BASE_S2S is None:
        _BASE_S2S = _base_symbol_to_sector()
    if symbol in _BASE_S2S:
        return _BASE_S2S[symbol]
    # yfinance fallback (import here to avoid a hard dependency at module load).
    try:
        from . import fundamentals as fund
        yf_sec = fund.get_fundamentals(symbol).get("sector")
        if yf_sec:
            return config.YF_SECTOR_MAP.get(yf_sec)
    except Exception:
        pass
    return None


def sector_composite(sector: str, interval: str = "1d") -> pd.Series:
    """Equal-weight normalised close for a sector's constituents."""
    stocks = config.SECTORS.get(sector, [])
    series = []
    for sym in stocks:
        df = datamod.get_ohlcv(sym, interval)
        if df is None or df.empty:
            continue
        c = df["Close"].dropna()
        if len(c) > 60:
            series.append(c / c.iloc[0])
    if not series:
        return pd.Series(dtype=float)
    mat = pd.concat(series, axis=1).ffill()
    return mat.mean(axis=1)


def all_composites(interval: str = "1d") -> dict[str, pd.Series]:
    return {s: sector_composite(s, interval) for s in config.SECTORS}


def rs_heatmap_table(interval: str = "1d") -> pd.DataFrame:
    """Trailing returns per sector over 1w / 1m / 3m plus RS vs NIFTY (1m)."""
    comps = all_composites(interval)
    bench = datamod.get_ohlcv(config.BENCHMARK, interval)["Close"]
    windows = {"1W": 5, "1M": 21, "3M": 63}
    rows = []
    for sec, comp in comps.items():
        if comp.empty:
            continue
        row = {"sector": sec}
        for label, w in windows.items():
            if len(comp) > w:
                row[f"{label} %"] = round((comp.iloc[-1] / comp.iloc[-w] - 1) * 100, 2)
            else:
                row[f"{label} %"] = np.nan
        # RS vs NIFTY over 1M.
        b = bench.reindex(comp.index).ffill()
        if len(comp) > 21 and len(b.dropna()) > 21:
            sec_ret = comp.iloc[-1] / comp.iloc[-21]
            ben_ret = b.iloc[-1] / b.iloc[-21]
            row["RS vs NIFTY (1M)"] = round(sec_ret / ben_ret, 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "1M %" in df:
        df = df.sort_values("1M %", ascending=False).reset_index(drop=True)
    return df


def rrg_coordinates(interval: str = "1d", tail: int = 5) -> pd.DataFrame:
    """RS-Ratio (x) and RS-Momentum (y) per sector, normalised around 100.

    RS-Ratio  > 100  => outperforming NIFTY.
    RS-Momentum> 100  => RS-Ratio is rising.
    Quadrant: (>100,>100)=Leading, (<100,>100)=Improving,
              (<100,<100)=Lagging, (>100,<100)=Weakening.
    """
    comps = all_composites(interval)
    bench = datamod.get_ohlcv(config.BENCHMARK, interval)["Close"]
    rows = []
    for sec, comp in comps.items():
        if comp.empty or len(comp) < 80:
            continue
        b = bench.reindex(comp.index).ffill()
        rs = (comp / b).dropna()
        if len(rs) < 70:
            continue
        # Normalise RS to a 100-centred ratio using a rolling z-score proxy.
        rs_ratio = 100 * rs / rs.rolling(63).mean()
        rs_mom = 100 * rs_ratio / rs_ratio.rolling(10).mean()
        rr = rs_ratio.iloc[-1]
        mm = rs_mom.iloc[-1]
        if not (np.isfinite(rr) and np.isfinite(mm)):
            continue
        if rr >= 100 and mm >= 100:
            quad = "Leading"
        elif rr < 100 and mm >= 100:
            quad = "Improving"
        elif rr < 100 and mm < 100:
            quad = "Lagging"
        else:
            quad = "Weakening"
        rows.append({"sector": sec, "rs_ratio": round(rr, 2),
                     "rs_momentum": round(mm, 2), "quadrant": quad,
                     "tail_x": [round(x, 2) for x in rs_ratio.iloc[-tail:].tolist()],
                     "tail_y": [round(y, 2) for y in rs_mom.iloc[-tail:].tolist()]})
    return pd.DataFrame(rows)


def _classify(rr: float, mm: float) -> str:
    if not (np.isfinite(rr) and np.isfinite(mm)):
        return "Unknown"
    if rr >= 100 and mm >= 100:
        return "Leading"
    if rr < 100 and mm >= 100:
        return "Improving"
    if rr < 100 and mm < 100:
        return "Lagging"
    return "Weakening"


def quadrant_history(interval: str = "1d") -> dict[str, pd.Series]:
    """Full historical RRG quadrant *time series* per sector (point-in-time).

    Same RS-Ratio / RS-Momentum math as ``rrg_coordinates`` but classified at
    EVERY date, not just the latest — so a backtest can ask "what quadrant was
    this sector in on the signal date?" and gate entries without look-ahead.

    Returns {sector: Series[str]} indexed by date.
    """
    comps = all_composites(interval)
    bench = datamod.get_ohlcv(config.BENCHMARK, interval)["Close"]
    out: dict[str, pd.Series] = {}
    for sec, comp in comps.items():
        if comp.empty or len(comp) < 80:
            continue
        b = bench.reindex(comp.index).ffill()
        rs = (comp / b).dropna()
        if len(rs) < 70:
            continue
        rs_ratio = 100 * rs / rs.rolling(63).mean()
        rs_mom = 100 * rs_ratio / rs_ratio.rolling(10).mean()
        quad = pd.Series(
            [_classify(r, m) for r, m in zip(rs_ratio.to_numpy(), rs_mom.to_numpy())],
            index=rs.index)
        out[sec] = quad
    return out


def quadrant_on(quad_series: pd.Series, date) -> str:
    """The sector's quadrant as of `date` (last known value on/before it)."""
    if quad_series is None or quad_series.empty:
        return "Unknown"
    try:
        idx = quad_series.index
        pos = idx.searchsorted(pd.Timestamp(date), side="right") - 1
        if pos < 0:
            return "Unknown"
        return str(quad_series.iloc[pos])
    except Exception:
        return "Unknown"


def rrg_figure(interval: str = "1d"):
    """Build the RRG scatter as a plotly figure."""
    import plotly.graph_objects as go
    df = rrg_coordinates(interval)
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="RRG — insufficient data")
        return fig
    colors = {"Leading": "#2ca02c", "Improving": "#1f77b4",
              "Lagging": "#d62728", "Weakening": "#ff7f0e"}
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(
            x=r["tail_x"], y=r["tail_y"], mode="lines+markers",
            line=dict(color=colors[r["quadrant"]], width=1),
            marker=dict(size=[6] * (len(r["tail_x"]) - 1) + [14]),
            name=r["sector"], text=r["sector"]))
    # Quadrant guide lines.
    fig.add_hline(y=100, line=dict(color="gray", dash="dot"))
    fig.add_vline(x=100, line=dict(color="gray", dash="dot"))
    fig.update_layout(
        title="Relative Rotation Graph (vs NIFTY 50)",
        xaxis_title="RS-Ratio (relative strength)  →  outperforming",
        yaxis_title="RS-Momentum  →  strengthening",
        height=560, margin=dict(l=50, r=20, t=50, b=40))
    # Annotate quadrants.
    return fig
