"""Screener page — rank fresh setups across a universe for any strategy."""

from __future__ import annotations

import streamlit as st

from nsewing import charts, risk, screener, ui
from nsewing.strategies import enabled_strategies, get_strategy

st.set_page_config(page_title="Screener", page_icon="🔍", layout="wide")
settings = ui.sidebar_controls()

st.title("🔍 Daily Screener")
st.caption("Scan the universe for high-probability setups, ranked by a 0–100 composite score.")

strat_name = st.selectbox("Strategy", enabled_strategies())
recent = st.slider("Look-back window (bars counted as 'actionable now')", 1, 60, 10)
with_fund = st.checkbox("Include fundamentals + earnings flag (slower)", value=True)

if st.button("Run screener", type="primary"):
    with st.spinner("Scanning universe…"):
        df = screener.scan(strat_name, settings["universe"],
                           interval=settings["interval"],
                           sensitivity=settings["sensitivity"],
                           recent_bars=recent, with_fundamentals=with_fund)
    if df.empty:
        st.warning("No setups found in this window. Widen the look-back or switch to 'relaxed'.")
    else:
        st.success(f"{len(df)} candidate(s). Top of the list = highest composite score.")
        st.dataframe(df, use_container_width=True, hide_index=True)

        sym = st.selectbox("Chart & size a candidate", df["symbol"].tolist())
        if sym:
            data = ui.load_ohlcv(sym, settings["interval"])
            strat = get_strategy(strat_name, sensitivity=settings["sensitivity"])
            sigs = strat.generate_signals(data, bench_close=ui.load_bench(settings["interval"]))
            st.plotly_chart(charts.candlestick(data.tail(220), sym, signals=sigs.tail(20)),
                            use_container_width=True)
            row = df[df["symbol"] == sym].iloc[0]
            sz = risk.position_size(settings["capital"], settings["risk_pct"],
                                    row["entry"], row["stop"])
            c = st.columns(4)
            c[0].metric("Shares", f"{sz['shares']:,}")
            c[1].metric("₹ at risk", f"₹{sz['rupee_risk']:,.0f}")
            c[2].metric("Notional", f"₹{sz['notional']:,.0f}")
            c[3].metric("Score", f"{row['score']}")
