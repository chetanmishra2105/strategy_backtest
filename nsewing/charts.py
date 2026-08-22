"""Plotly chart builders for the Streamlit UI.

Every function returns a plotly Figure so pages can just call st.plotly_chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import indicators as ind


def candlestick(df: pd.DataFrame, title: str = "", overlays: bool = True,
                signals: pd.DataFrame | None = None) -> go.Figure:
    """Price candles + volume + RSI + MACD panels, with optional trade markers."""
    d = ind.enrich(df) if overlays else df
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.175, 0.175], vertical_spacing=0.03,
        subplot_titles=(title or "Price", "Volume", "RSI(14)", "MACD"),
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Price"), row=1, col=1)

    if overlays:
        for col, color in [("EMA20", "#1f77b4"), ("EMA50", "#ff7f0e"), ("EMA200", "#9467bd")]:
            if col in d:
                fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col,
                                         line=dict(width=1, color=color)), row=1, col=1)

    # Volume with its 20-MA.
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume",
                         marker_color="#888"), row=2, col=1)
    if "VOLMA20" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["VOLMA20"], name="VolMA20",
                                 line=dict(width=1, color="orange")), row=2, col=1)

    if "RSI14" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["RSI14"], name="RSI",
                                 line=dict(color="#2ca02c")), row=3, col=1)
        fig.add_hline(y=70, line=dict(dash="dot", color="red"), row=3, col=1)
        fig.add_hline(y=30, line=dict(dash="dot", color="green"), row=3, col=1)

    if "MACD" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD",
                                 line=dict(color="#1f77b4")), row=4, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD_SIGNAL"], name="Signal",
                                 line=dict(color="#ff7f0e")), row=4, col=1)
        fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"], name="Hist",
                             marker_color="#bbb"), row=4, col=1)

    # Signal markers (entry points).
    if signals is not None and not signals.empty:
        pts = signals.set_index("date")
        yv = d["Close"].reindex(pts.index)
        color = "red" if (signals["side"].iloc[0] == "short") else "lime"
        symbol = "triangle-down" if signals["side"].iloc[0] == "short" else "triangle-up"
        fig.add_trace(go.Scatter(x=pts.index, y=yv, mode="markers", name="Signal",
                                 marker=dict(size=11, color=color, symbol=symbol,
                                             line=dict(width=1, color="black"))),
                      row=1, col=1)

    fig.update_layout(height=780, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02),
                      margin=dict(l=40, r=20, t=40, b=20))
    return fig


def trade_chart(df: pd.DataFrame, trades: pd.DataFrame, symbol: str) -> go.Figure:
    """Price with entry/exit markers for a single symbol's trades."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
        close=df["Close"], name=symbol))
    if trades is not None and not trades.empty:
        t = trades[trades["symbol"] == symbol]
        fig.add_trace(go.Scatter(x=t["entry_date"], y=t["entry_price"], mode="markers",
                                 name="Entry", marker=dict(color="blue", size=9, symbol="circle")))
        wins = t[t["ret_pct"] > 0]
        losses = t[t["ret_pct"] <= 0]
        fig.add_trace(go.Scatter(x=wins["exit_date"], y=wins["exit_price"], mode="markers",
                                 name="Exit (win)", marker=dict(color="green", size=9, symbol="x")))
        fig.add_trace(go.Scatter(x=losses["exit_date"], y=losses["exit_price"], mode="markers",
                                 name="Exit (loss)", marker=dict(color="red", size=9, symbol="x")))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False,
                      title=f"Trades — {symbol}", margin=dict(l=40, r=20, t=40, b=20))
    return fig


def equity_curve(equity: pd.Series, benchmark: pd.Series | None = None) -> go.Figure:
    """Strategy equity vs a normalised buy&hold benchmark."""
    fig = go.Figure()
    if equity is not None and len(equity):
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                 name="Strategy", line=dict(color="#1f77b4", width=2)))
    if benchmark is not None and len(benchmark):
        b = benchmark.reindex(equity.index, method="ffill") if len(equity) else benchmark
        b = b / b.dropna().iloc[0]
        fig.add_trace(go.Scatter(x=b.index, y=b.values, name="Buy & Hold (NIFTY)",
                                 line=dict(color="#999", width=1, dash="dash")))
    fig.update_layout(height=380, title="Equity curve (growth of ₹1, position-sized)",
                      margin=dict(l=40, r=20, t=40, b=20))
    return fig


def equity_curve_rupees(equity: pd.Series, capital: float) -> go.Figure:
    """Portfolio value in ₹ over time, with the starting-capital baseline."""
    fig = go.Figure()
    if equity is not None and len(equity):
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                 name="Portfolio value", line=dict(color="#1f77b4", width=2)))
        fig.add_hline(y=capital, line=dict(color="#999", dash="dash"),
                      annotation_text=f"Start ₹{capital:,.0f}")
    fig.update_layout(height=380, title="Portfolio value (₹)",
                      margin=dict(l=60, r=20, t=40, b=20))
    return fig


def compare_curve(comp: pd.DataFrame) -> go.Figure:
    """Multi-line normalised comparison (strategy vs indices), distinct legend."""
    fig = go.Figure()
    palette = {"Strategy": "#1f77b4", "NIFTY 50": "#2ca02c",
               "NIFTY Midcap 150": "#ff7f0e", "NIFTY Smallcap 250": "#d62728"}
    for col in comp.columns:
        width = 3 if col == "Strategy" else 1.6
        dash = None if col == "Strategy" else "dot"
        fig.add_trace(go.Scatter(
            x=comp.index, y=(comp[col] - 1) * 100, name=col,
            line=dict(color=palette.get(col), width=width, dash=dash)))
    fig.add_hline(y=0, line=dict(color="#bbb", dash="dash"))
    fig.update_layout(height=440, title="Strategy vs broad-market indices — total return %",
                      yaxis_title="Cumulative return (%)",
                      legend=dict(orientation="h", y=1.04),
                      margin=dict(l=50, r=20, t=50, b=30))
    return fig


def drawdown_curve(equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    if equity is not None and len(equity):
        dd = equity / equity.cummax() - 1.0
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, name="Drawdown",
                                 fill="tozeroy", line=dict(color="#d62728")))
    fig.update_layout(height=280, title="Drawdown (%)",
                      margin=dict(l=40, r=20, t=40, b=20))
    return fig


def returns_hist(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades is not None and not trades.empty:
        fig.add_trace(go.Histogram(x=trades["ret_pct"] * 100, nbinsx=40,
                                   marker_color="#1f77b4"))
        fig.add_vline(x=0, line=dict(color="black", dash="dot"))
    fig.update_layout(height=300, title="Per-trade return distribution (%)",
                      margin=dict(l=40, r=20, t=40, b=20))
    return fig


def performance_heatmap(pivot: pd.DataFrame, title: str = "") -> go.Figure:
    """Heatmap of a strategy x window metric pivot (e.g. total return %)."""
    fig = go.Figure()
    if pivot is None or pivot.empty:
        return fig
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale="RdYlGn", zmid=0,
        text=pivot.round(2).values, texttemplate="%{text}",
        colorbar=dict(title="%")))
    fig.update_layout(height=340, title=title or "Performance by trailing window",
                      margin=dict(l=140, r=20, t=50, b=30))
    return fig


def monthly_heatmap(equity: pd.Series) -> go.Figure:
    """Monthly returns heatmap from the equity curve."""
    fig = go.Figure()
    if equity is None or len(equity) < 2:
        return fig
    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return fig
    dfm = monthly.to_frame("ret")
    dfm["year"] = dfm.index.year
    dfm["month"] = dfm.index.month
    pivot = dfm.pivot_table(index="year", columns="month", values="ret") * 100
    fig.add_trace(go.Heatmap(z=pivot.values, x=[str(m) for m in pivot.columns],
                             y=[str(y) for y in pivot.index],
                             colorscale="RdYlGn", zmid=0,
                             colorbar=dict(title="%")))
    fig.update_layout(height=320, title="Monthly returns (%)",
                      margin=dict(l=40, r=20, t=40, b=20))
    return fig
