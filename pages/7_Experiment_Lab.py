"""Experiment Lab — one-button grid search over a chosen strategy's combinations.

Pick a **strategy** at the top, then sweep its parameter grid:

  * **Supertrend Sector Momentum** — universe × timeframe × stop/target ×
    indicator preset × gates(on/off), each run through the SAME funnel + ₹
    portfolio engine as the Backtest Lab.
  * **Cross-sectional Momentum** — universe × lookback × hold-count × rebalance ×
    layer-config, each run through the SAME walk-forward (out-of-sample) harness
    as the Momentum Lab.

Both rank by risk-adjusted Calmar (CAGR ÷ |max drawdown|) with a min-trades floor,
and the winning combo per strategy is logged to ``benchmark/benchmark.md`` (it
only overwrites when a new run beats the previous best CAGR).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nsewing import benchmark as bm, charts, experiments as ex
from nsewing import momentum_experiments as mex, ui

st.set_page_config(page_title="Experiment Lab", page_icon="🧪", layout="wide")
settings = ui.sidebar_controls()  # keep the global sidebar; this page uses its own axes

st.title("🧪 Experiment Lab — find the best combination")

# --------------------------------------------------------------------------
# Strategy selector — drives which grid appears below.
# --------------------------------------------------------------------------
strategy = st.selectbox(
    "Strategy to sweep",
    [ex.STRATEGY, mex.STRATEGY],
    help="Pick which strategy's parameter combinations to grid-search. Each "
         "strategy has its own axes but the same ranking + benchmark logging.")

# ==========================================================================
# MOMENTUM BRANCH — cross-sectional momentum grid (walk-forward, OOS).
# Kept fully separate from the Supertrend code below (which is untouched).
# ==========================================================================
if strategy == mex.STRATEGY:
    st.caption("Sweeps momentum permutations and ranks them by **risk-adjusted return "
               "(Calmar = CAGR ÷ |max drawdown|)**. Each combination runs through the exact same "
               "**walk-forward, out-of-sample** harness as the Momentum Lab, so every number is "
               "what a live trader would actually have earned — not a curve fit to the past.")

    with st.expander("ℹ️ How each combo is scored (read me)", expanded=False):
        st.markdown(
            "- **Every number is out-of-sample.** Each combination runs the full walk-forward "
            "harness (9-month train / 3-month test folds, rolled forward); the reported CAGR / "
            "drawdown are stitched from the **test windows only**.\n"
            "- **Fixed while sweeping:** 12-1 skip (skip most-recent month), the 200-DMA trend "
            "filter, a 3-state HMM, and a 5-year history window — so combos compare fairly. The "
            "swept axes are the ones below.\n"
            "- **Layers axis** is the momentum analogue of gates: *Momentum only* → *+HMM regime "
            "filter* → *+HMM + vol-scaled sizing* (the same ablation ladder as the Momentum Lab).\n"
            "- **Slower than the Supertrend grid** — each combo runs many folds. Trim axes to keep "
            "run time sane.")

    st.divider()
    st.subheader("① Choose what to sweep")
    st.caption("Defaults are a balanced grid. Trim axes to run faster, or add to be exhaustive.")

    dm = mex.DEFAULT_AXES
    m1, m2 = st.columns(2)
    m_universes = m1.multiselect("Universes", mex.UNIVERSE_NAMES, default=dm["universes"])
    m_layers = m2.multiselect("Layer configs (momentum → +regime → +vol sizing)",
                              list(mex.LAYER_CONFIGS.keys()), default=dm["layers"])
    m_lookbacks = st.multiselect("Momentum lookback (months)", mex.LOOKBACK_MONTHS,
                                 default=dm["lookbacks"])
    m_topns = st.multiselect("Hold top-N stocks", mex.TOP_N_CHOICES, default=dm["top_ns"])
    m_rebs = st.multiselect("Rebalance frequency", list(mex.REBALANCE_MODES.keys()),
                            default=dm["rebalances"])

    m_min_trades = st.slider("Minimum trades to rank a combo (thin combos are flagged & pushed down)",
                             5, 60, 20, 5, key="mom_min_trades")

    m_axes = dict(universes=m_universes, lookbacks=m_lookbacks, top_ns=m_topns,
                  rebalances=m_rebs, layers=m_layers)
    m_valid = all(m_axes.values())
    m_n = mex.count_combos(m_axes) if m_valid else 0
    # Walk-forward is heavier than the Supertrend single-pass: ~4-8s per combo.
    m_est = m_n * 6.0 / 60.0
    if not m_valid:
        st.warning("Pick at least one option in every axis.")
    else:
        st.info(f"**≈ {m_n} combinations** to run  ·  rough estimate **{m_est:.1f}–{m_est*1.8:.1f} min** "
                "(walk-forward runs many folds per combo; first run downloads/caches data).")

    m_run = st.button("② Run experiment grid", type="primary", disabled=not m_valid)

    if m_run:
        grid = mex.build_grid(m_axes)
        bar = st.progress(0.0, text=f"Running 0 / {len(grid)} combinations…")

        def _mcb(done, total):
            bar.progress(done / total, text=f"Running {done} / {total} combinations…")

        with st.spinner("Simulating every combination through the walk-forward harness…"):
            m_res = mex.run_grid(grid, progress_cb=_mcb)
        bar.empty()
        st.session_state["mex_results"] = m_res
        st.session_state["mex_min_trades"] = m_min_trades

    if "mex_results" in st.session_state:
        m_res = st.session_state["mex_results"]
        mt = st.session_state.get("mex_min_trades", m_min_trades)
        ranked = mex.rank(m_res, min_trades=mt)

        if ranked is None or ranked.empty:
            st.warning("No results.")
            st.stop()

        st.divider()
        st.subheader("🏆 Best combination (risk-adjusted, out-of-sample)")
        non_thin = ranked[~ranked["thin"]]
        best = (non_thin.iloc[0] if not non_thin.empty else ranked.iloc[0])
        k = st.columns(5)
        k[0].metric("OOS CAGR", f"{best['CAGR_%']:+.1f}%")
        k[1].metric("Calmar", f"{best['calmar']:.2f}")
        k[2].metric("Sharpe", f"{best['sharpe']:.2f}")
        k[3].metric("Max drawdown", f"{best['max_dd_%']:.1f}%")
        k[4].metric("Trades", int(best["trades"]))
        st.markdown(
            f"**{best['layer']}** on **{best['universe']}**, lookback "
            f"**{best['lookback_m']}mo**, hold top **{int(best['top_n'])}**, "
            f"**{best['rebalance']}** rebalance.")

        # --- Log the champion to benchmark/benchmark.md (all-time best) -------
        best_params = dict(
            universe=best["universe"], lookback_m=int(best["lookback_m"]),
            top_n=int(best["top_n"]), rebalance=best["rebalance"], layer=best["layer"])
        best_metrics = dict(calmar=float(best["calmar"]), max_dd_pct=float(best["max_dd_%"]),
                            sharpe=float(best["sharpe"]), trades=int(best["trades"]))
        status = bm.update_best(
            mex.STRATEGY, best["CAGR_%"] / 100.0, best_params,
            metrics={"calmar": float(best["calmar"]), "max_dd_%": float(best["max_dd_%"]),
                     "sharpe": float(best["sharpe"]), "trades": int(best["trades"])},
            when=str(pd.Timestamp.today().date()))
        if status["updated"]:
            prev = status["previous_cagr"]
            prev_txt = f" (beat previous {prev*100:+.1f}%)" if prev is not None else " (first record)"
            st.success(f"🏅 New benchmark for **{mex.STRATEGY}**: {best['CAGR_%']:+.1f}% CAGR"
                       f"{prev_txt} — logged to `benchmark/benchmark.md`.")
        else:
            st.caption(f"Benchmark unchanged — the all-time best for {mex.STRATEGY} "
                       f"({status['previous_cagr']*100:+.1f}% CAGR) still stands.")

        st.subheader("Top 15 combinations")
        show_cols = ["universe", "lookback_m", "top_n", "rebalance", "layer",
                     "trades", "win_%", "total_%", "CAGR_%", "max_dd_%",
                     "sharpe", "calmar", "thin"]
        show_cols = [c for c in show_cols if c in ranked.columns]
        st.dataframe(ranked[show_cols].head(15), use_container_width=True, hide_index=True)

        st.subheader("Heatmap")
        hc = st.columns(3)
        metric = hc[0].selectbox("Colour by", ["CAGR_%", "calmar", "sharpe", "total_%"], index=0)
        row_ax = hc[1].selectbox("Rows", ["layer", "universe", "lookback_m", "top_n", "rebalance"], index=0)
        col_ax = hc[2].selectbox("Columns", ["universe", "lookback_m", "top_n", "rebalance", "layer"], index=0)
        if row_ax != col_ax:
            piv = m_res.pivot_table(index=row_ax, columns=col_ax, values=metric, aggfunc="mean")
            st.plotly_chart(charts.performance_heatmap(piv, f"Mean {metric} by {row_ax} × {col_ax}"),
                            use_container_width=True)
        else:
            st.caption("Pick different Rows and Columns to draw the heatmap.")

        st.subheader("Full results")
        st.dataframe(ranked[show_cols], use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download results as CSV",
                           ranked.to_csv(index=False).encode("utf-8"),
                           file_name="momentum_experiment_results.csv", mime="text/csv")

        st.caption("**How to read this:** prefer high **Calmar** (return per unit of drawdown) among "
                   "rows with enough **trades**. Because every number here is out-of-sample, a modest "
                   "best combo is the honest ceiling of momentum on these universes — not a bug.")

    st.stop()  # ← momentum branch ends here; Supertrend code below is untouched.

# ==========================================================================
# SUPERTREND BRANCH — the original grid (unchanged).
# ==========================================================================
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

    # --- Log the champion to benchmark/benchmark.md (all-time best) -----------
    # Prefer the clean (point-in-time) combo as the honest benchmark; fall back
    # to the best overall combo if every row used the fundamentals leak.
    champ = clean.iloc[0] if not clean.empty else best
    champ_params = dict(
        universe=champ["universe"], interval=champ["interval"], bracket=champ["bracket"],
        preset=champ["preset"], gates=champ["gates"])
    status = bm.update_best(
        ex.STRATEGY, champ["CAGR_%"] / 100.0, champ_params,
        metrics={"calmar": float(champ["calmar"]), "max_dd_%": float(champ["max_dd_%"]),
                 "PF": float(champ["PF"]), "trades": int(champ["trades"]),
                 "clean": bool(not champ["fundamental_leak"])},
        when=str(pd.Timestamp.today().date()))
    if status["updated"]:
        prev = status["previous_cagr"]
        prev_txt = f" (beat previous {prev*100:+.1f}%)" if prev is not None else " (first record)"
        st.success(f"🏅 New benchmark for **{ex.STRATEGY}**: {champ['CAGR_%']:+.1f}% CAGR"
                   f"{prev_txt} — logged to `benchmark/benchmark.md`.")
    else:
        st.caption(f"Benchmark unchanged — the all-time best for {ex.STRATEGY} "
                   f"({status['previous_cagr']*100:+.1f}% CAGR) still stands.")

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
