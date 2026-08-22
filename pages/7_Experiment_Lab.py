"""Experiment Lab — one-button grid search over Supertrend combinations.

Sweeps universe × timeframe × stop/target × indicator preset × gates(on/off),
runs each through the SAME funnel + ₹ portfolio engine as the Backtest Lab, and
ranks the results by risk-adjusted Calmar (CAGR ÷ |max drawdown|). The goal:
find, systematically, which combination actually works — instead of guessing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nsewing import charts, experiments as ex, ui

st.set_page_config(page_title="Experiment Lab", page_icon="🧪", layout="wide")
ui.sidebar_controls()  # keep the global sidebar; this page uses its own axes

st.title("🧪 Experiment Lab — find the best Supertrend combination")
st.caption("Sweeps many permutations of the Supertrend strategy and ranks them by **risk-adjusted "
           "return (Calmar = CAGR ÷ |max drawdown|)**. Each combination runs through the exact same "
           "funnel + ₹ portfolio backtest as the Backtest Lab, so the numbers are consistent.")

with st.expander("🔍 Is this look-ahead-free? (read me)", expanded=False):
    st.markdown(
        "- **Point-in-time (no look-ahead):** all technical indicators, the weekly Supertrend, "
        "next-bar entry, close-based stops, and the sector-rotation gate are computed using only "
        "data available *on each signal's own date*.\n"
        "- **The one compromise:** with **gates ON**, the fundamental & QoQ-growth filters use "
        "*today's* yfinance data across all history (yfinance has no historical point-in-time "
        "financials). Those rows are flagged **`fundamental_leak = True`** and are slightly "
        "optimistic.\n"
        "- **Gates OFF = the clean benchmark** — 100% point-in-time. Prefer those rows when judging "
        "a real, tradeable edge.")

st.divider()
st.subheader("① Choose what to sweep")
st.caption("Defaults are a balanced grid. Trim axes to run faster, or add to be exhaustive.")

d = ex.DEFAULT_AXES
c1, c2 = st.columns(2)
universes = c1.multiselect("Universes", ex.UNIVERSE_NAMES, default=d["universes"])
intervals = c2.multiselect("Timeframes", ex.INTERVALS, default=d["intervals"])
brackets = st.multiselect("Stop / Target brackets", list(ex.STOP_BRACKETS.keys()),
                          default=d["brackets"])
presets = st.multiselect("Indicator presets (which filters are on)",
                         list(ex.INDICATOR_PRESETS.keys()), default=d["presets"])
gates = st.multiselect("Gates (fundamentals + growth + sector rotation)", ex.GATE_MODES,
                       default=d["gates"],
                       help="ON = apply the fundamental/growth/sector funnel (uses today's "
                            "fundamentals → flagged as a leak). OFF = pure technical, fully "
                            "point-in-time.")

min_trades = st.slider("Minimum trades to rank a combo (thin combos are flagged & pushed down)",
                       5, 60, 20, 5)

axes = dict(universes=universes, intervals=intervals, brackets=brackets,
            presets=presets, gates=gates)
valid = all(axes.values())
n_combos = ex.count_combos(axes) if valid else 0
# Rough estimate: ~1-3s per combo depending on universe size & timeframe.
est_min = n_combos * 2.0 / 60.0
if not valid:
    st.warning("Pick at least one option in every axis.")
else:
    st.info(f"**≈ {n_combos} combinations** to run  ·  rough estimate **{est_min:.1f}–{est_min*1.8:.1f} min** "
            "(first run downloads/caches data; re-runs are faster).")

run = st.button("② Run experiment grid", type="primary", disabled=not valid)

# --------------------------------------------------------------------------
# Run the grid (cached in session_state so re-sorting doesn't re-run it).
# --------------------------------------------------------------------------
if run:
    grid = ex.build_grid(axes)
    bar = st.progress(0.0, text=f"Running 0 / {len(grid)} combinations…")

    def _cb(done, total):
        bar.progress(done / total, text=f"Running {done} / {total} combinations…")

    with st.spinner("Simulating every combination through the full funnel…"):
        res = ex.run_grid(grid, progress_cb=_cb)
    bar.empty()
    st.session_state["ex_results"] = res
    st.session_state["ex_min_trades"] = min_trades

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
if "ex_results" in st.session_state:
    res = st.session_state["ex_results"]
    mt = st.session_state.get("ex_min_trades", min_trades)
    ranked = ex.rank(res, min_trades=mt)

    if ranked.empty:
        st.warning("No results.")
        st.stop()

    st.divider()
    st.subheader("🏆 Best combination (risk-adjusted)")
    # Best = top non-thin row; fall back to overall top if all are thin.
    non_thin = ranked[~ranked["thin"]]
    best = (non_thin.iloc[0] if not non_thin.empty else ranked.iloc[0])
    leak_txt = "⚠️ uses today's fundamentals (slightly optimistic)" if best["fundamental_leak"] \
        else "✅ clean / point-in-time"
    k = st.columns(5)
    k[0].metric("CAGR", f"{best['CAGR_%']:+.1f}%")
    k[1].metric("Calmar", f"{best['calmar']:.2f}")
    k[2].metric("Profit factor", f"{best['PF']:.2f}")
    k[3].metric("Max drawdown", f"{best['max_dd_%']:.1f}%")
    k[4].metric("Trades", int(best["trades"]))
    st.markdown(
        f"**{best['preset']}** on **{best['universe']}** ({best['interval']}), "
        f"stop/target **{best['bracket']}**, gates **{best['gates']}** — {leak_txt}.")

    # Best CLEAN (gates-OFF) combo, called out separately since it's the honest one.
    clean = non_thin[~non_thin["fundamental_leak"]]
    if not clean.empty:
        cb = clean.iloc[0]
        st.success(
            f"**Best clean (point-in-time) combo:** {cb['preset']} on {cb['universe']} "
            f"({cb['interval']}), {cb['bracket']} → **{cb['CAGR_%']:+.1f}% CAGR**, "
            f"Calmar {cb['calmar']:.2f}, PF {cb['PF']:.2f}, DD {cb['max_dd_%']:.1f}%, "
            f"{int(cb['trades'])} trades. This is the number to trust.")

    st.subheader("Top 15 combinations")
    show_cols = ["universe", "interval", "bracket", "preset", "gates",
                 "fundamental_leak", "trades", "win_%", "PF", "total_%",
                 "CAGR_%", "max_dd_%", "calmar", "thin"]
    st.dataframe(ranked[show_cols].head(15), use_container_width=True, hide_index=True)

    st.subheader("Heatmap")
    hc = st.columns(3)
    metric = hc[0].selectbox("Colour by", ["CAGR_%", "calmar", "PF", "total_%", "win_%"], index=0)
    row_ax = hc[1].selectbox("Rows", ["preset", "universe", "bracket", "interval"], index=0)
    col_ax = hc[2].selectbox("Columns", ["bracket", "interval", "universe", "preset"], index=0)
    if row_ax != col_ax:
        # Average the metric over any other axes for the 2-D view.
        piv = res.pivot_table(index=row_ax, columns=col_ax, values=metric, aggfunc="mean")
        st.plotly_chart(charts.performance_heatmap(piv, f"Mean {metric} by {row_ax} × {col_ax}"),
                        use_container_width=True)
    else:
        st.caption("Pick different Rows and Columns to draw the heatmap.")

    st.subheader("Full results")
    st.dataframe(ranked[show_cols], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download results as CSV",
                       ranked.to_csv(index=False).encode("utf-8"),
                       file_name="supertrend_experiment_results.csv", mime="text/csv")

    st.caption("**How to read this:** prefer high **Calmar** (return per unit of drawdown) among "
               "rows with enough **trades** and `fundamental_leak = False`. A high CAGR with a huge "
               "drawdown or 3 trades is not a real edge. If the best clean combo is still modest, "
               "that is the honest ceiling of this strategy on these universes — not a bug.")
