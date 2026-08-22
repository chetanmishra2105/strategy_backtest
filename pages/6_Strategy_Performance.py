"""Strategy Performance page — compare all strategies across trailing windows
(last 30 / 60 / 90 / 120 / 180 days) for the selected universe & interval."""

from __future__ import annotations

import streamlit as st

from nsewing import charts, performance, ui

st.set_page_config(page_title="Strategy Performance", page_icon="📊", layout="wide")
settings = ui.sidebar_controls()

st.title("📊 Strategy Performance Comparison")
st.caption("All strategies, run over trailing windows on the selected universe — through the **same "
           "funnel as the Backtest Lab** (fundamentals + QoQ-growth + point-in-time sector gate, "
           "10% position cap, real ₹ portfolio). Use it to see which strategy works on which cap "
           "segment over meaningful spans.")

st.info("**Windows are now swing-appropriate.** This strategy holds trades 30–90 days, so windows "
        "shorter than ~6 months are dominated by unfinished trades and pure noise — they've been "
        "removed. Annualised CAGR now comes from the portfolio's own equity curve (not a naive "
        "short-window extrapolation). Still check the **trades** count: a great number on <15 trades "
        "means little.")

WINDOW_LABELS = {"6 months": 180, "1 year": 365, "2 years": 730, "3 years": 1095}
picked = st.multiselect("Trailing windows", list(WINDOW_LABELS.keys()),
                        default=["6 months", "1 year", "2 years", "3 years"])
windows = [WINDOW_LABELS[p] for p in picked]

metric = st.selectbox(
    "Metric to highlight",
    ["total_return_%", "annualised_CAGR_%", "win_rate_%", "profit_factor",
     "expectancy_%", "trades"],
    index=0)

if st.button("Run comparison", type="primary"):
    if not windows:
        st.warning("Pick at least one window.")
    else:
        with st.spinner(f"Backtesting all strategies over {len(windows)} windows on "
                        f"{settings['universe_name']}…"):
            mat = performance.run_matrix(
                settings["universe"], interval=settings["interval"],
                windows=sorted(windows), sensitivity=settings["sensitivity"],
                risk_pct=settings["risk_pct"])
        if mat.empty:
            st.warning("No results — try a longer window or 'relaxed' sensitivity.")
        else:
            st.subheader(f"Heatmap — {metric} ({settings['universe_name']}, {settings['interval']})")
            pivot = performance.pivot_metric(mat, metric)
            st.plotly_chart(charts.performance_heatmap(pivot, f"{metric} by trailing window"),
                            use_container_width=True)

            st.subheader("Full results")
            st.dataframe(mat, use_container_width=True, hide_index=True)

            st.subheader("Pivot views")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Total return %")
                st.dataframe(performance.pivot_metric(mat, "total_return_%"),
                             use_container_width=True)
                st.caption("Win rate %")
                st.dataframe(performance.pivot_metric(mat, "win_rate_%"),
                             use_container_width=True)
            with c2:
                st.caption("Trades (sample size — watch for thin windows)")
                st.dataframe(performance.pivot_metric(mat, "trades"),
                             use_container_width=True)
                st.caption("Profit factor (>1 = net winning)")
                st.dataframe(performance.pivot_metric(mat, "profit_factor"),
                             use_container_width=True)

            st.caption("Tip: switch the **Universe** in the sidebar (Top 25 / NIFTY 50 / Midcap 50 / "
                       "Smallcap 50) and re-run to compare how each strategy performs across cap segments.")
