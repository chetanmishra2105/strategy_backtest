"""Shared Streamlit helpers used across pages."""

from __future__ import annotations

import streamlit as st

from . import config
from . import data as datamod
from . import risk as riskmod


def sidebar_controls():
    """Render the global sidebar and return a settings dict."""
    st.sidebar.header("⚙️ Settings")
    capital = st.sidebar.number_input("Capital (₹)", min_value=10_000,
                                      value=int(config.DEFAULT_CAPITAL), step=50_000)

    vix = datamod.latest_vix()
    regime = riskmod.vix_regime(vix)
    if vix is not None:
        st.sidebar.metric("India VIX", f"{vix:.2f}", regime["label"])
        st.sidebar.caption(f"Suggested risk: **{regime['risk_pct']*100:.1f}%** — {regime['note']}")
    risk_pct = st.sidebar.slider("Risk per trade (%)", 0.25, 4.0,
                                 float(regime["risk_pct"] * 100), 0.25) / 100.0

    universe_name = st.sidebar.selectbox("Universe", list(config.UNIVERSES.keys()))
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1h"], index=0)
    sensitivity = st.sidebar.radio(
        "Signal sensitivity", ["relaxed", "strict"], index=0,
        help="'strict' = the document's literal thresholds (very few signals on "
             "daily data). 'relaxed' = tradeable defaults.")

    if st.sidebar.button("🔄 Refresh data cache"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared — data will re-download.")

    st.sidebar.divider()
    st.sidebar.caption("Data: yfinance (public). Not investment advice.")

    return {
        "capital": capital, "risk_pct": risk_pct, "vix": vix, "regime": regime,
        "universe_name": universe_name, "universe": config.UNIVERSES[universe_name],
        "interval": interval, "sensitivity": sensitivity,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def load_ohlcv(symbol: str, interval: str):
    return datamod.get_ohlcv(symbol, interval)


@st.cache_data(show_spinner=False, ttl=3600)
def load_bench(interval: str):
    if interval in ("1d", "1wk"):
        df = datamod.get_ohlcv(config.BENCHMARK, interval)
        return df["Close"] if not df.empty else None
    return None


def metric_row(metrics: dict, doc_stats: dict | None = None):
    """Render the key backtest metrics, with doc comparison if available."""
    c = st.columns(4)
    c[0].metric("Trades", metrics.get("trades", 0))
    c[1].metric("Win rate", f"{metrics.get('win_rate',0)*100:.1f}%")
    c[2].metric("Profit factor", f"{metrics.get('profit_factor',0):.2f}")
    c[3].metric("Expectancy/trade", f"{metrics.get('expectancy',0)*100:+.2f}%")
    c = st.columns(4)
    c[0].metric("CAGR", f"{metrics.get('cagr',0)*100:+.1f}%")
    c[1].metric("Max drawdown", f"{metrics.get('max_dd',0)*100:.1f}%")
    c[2].metric("Sharpe", f"{metrics.get('sharpe',0):.2f}")
    c[3].metric("Avg hold (bars)", f"{metrics.get('avg_bars_held',0):.1f}")

    if doc_stats:
        with st.expander("📄 How this compares to the document's published stats"):
            import pandas as pd
            comp = pd.DataFrame({
                "Metric": ["Trades", "Win rate", "Profit factor", "CAGR", "Max DD"],
                "This backtest": [
                    metrics.get("trades", 0),
                    f"{metrics.get('win_rate',0)*100:.1f}%",
                    f"{metrics.get('profit_factor',0):.2f}",
                    f"{metrics.get('cagr',0)*100:+.1f}%",
                    f"{metrics.get('max_dd',0)*100:.1f}%",
                ],
                "Document claim": [
                    doc_stats.get("trades", "—"),
                    f"{doc_stats.get('win_rate',0)*100:.1f}%",
                    f"{doc_stats.get('profit_factor','—')}",
                    f"{doc_stats.get('cagr',0)*100:.1f}%",
                    f"{doc_stats.get('max_dd',0)*100:.1f}%",
                ],
            })
            st.table(comp)
            st.caption(
                "Realistic, cost-adjusted, look-ahead-free results are typically **well below** "
                "published backtest claims. Treat the document's numbers as marketing, not a forecast.")
