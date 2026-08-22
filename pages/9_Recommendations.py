"""📋 Recommendations — today's actionable picks per strategy.

For each ENABLED strategy this scans the chosen universe for signals firing on
the most recent bar(s), ranks them **Strong Buy → Buy → Watch**, and shows for
every stock: entry, stop-loss, targets, reward:risk, the indicator readings at
the signal, and the strategy's probable holding period. Pick any stock to chart
it with SL/target lines; save the day's list to a dated CSV.

New, self-contained page — it does not modify any strategy, the Backtest Lab, or
the Momentum Lab. It reuses nsewing.recommend (which reuses the screener,
indicators and backtester), so numbers agree with the rest of the app.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from nsewing import config, indicators as ind, recommend as rec, ui
from nsewing.strategies import enabled_strategies

st.set_page_config(page_title="Recommendations", page_icon="📋", layout="wide")
settings = ui.sidebar_controls()

st.title("📋 Recommendations — today's picks per strategy")
st.caption("Each enabled strategy is scanned for signals on the most recent bar(s). Picks are ranked "
           "**Strong Buy → Buy → Watch** by a 0–100 conviction score, with stop-loss, targets, the "
           "indicators at the signal, and a probable holding period. Not investment advice — always "
           "check the chart and your own risk before acting.")

enabled = enabled_strategies()
with st.expander("ℹ️ What the columns mean / how buckets are decided", expanded=False):
    st.markdown(
        f"- **Enabled strategies scanned:** {', '.join(enabled)}. "
        "To scan more, set them `True` in `nsewing/config.py → STRATEGY_ENABLED`.\n"
        "- **Buckets:** STRONG BUY = score ≥ 70, BUY = score ≥ 50, else WATCH. Score blends volume "
        "conviction, reward:risk, momentum (RSI) and relative strength.\n"
        "- **stop_loss / target1-3:** the exact levels for the trade, computed from the entry. `sl_%` "
        "and `t1_%` are those distances in percent.\n"
        "- **probable_hold_days:** the strategy's *historical average* holding period on this universe "
        "(from a quick backtest) — a realistic expectation, not a guarantee. `max_hold_days` is the "
        "hard cap.\n"
        "- **Indicator columns (RSI14, ADX14, MACD>Signal, EMA structure, Supertrend, vol_ratio, "
        "Williams%R):** the readings **on the signal bar**, i.e. what the stock looked like when the "
        "strategy flagged it.\n"
        "- **⚠ earnings_flag:** results are due inside the horizon — extra event risk.")

# --- Controls ---------------------------------------------------------------
c = st.columns(4)
recent = c[0].selectbox("Signal freshness (last N bars)", [1, 2, 3, 5, 10], index=2,
                        help="How many of the most recent bars count as 'actionable now'. "
                             "1 = only today's bar; 5 = any signal in the last 5 bars.")
with_fund = c[1].checkbox("Fundamentals + earnings flag", value=True)
horizon = c[2].selectbox("Earnings horizon (days)", [15, 30, 45], index=1)
only_actionable = c[3].checkbox("Hide 'Watch' (show Buy/Strong Buy only)", value=False)

run = st.button("Scan for recommendations", type="primary")

if run:
    with st.spinner(f"Scanning {len(settings['universe'])} stocks across {len(enabled)} strategies…"):
        res = rec.recommend_all(
            settings["universe"], interval=settings["interval"],
            sensitivity=settings["sensitivity"], recent_bars=recent,
            horizon=horizon, with_fundamentals=with_fund)
    st.session_state["rec_res"] = res
    st.session_state["rec_ctx"] = dict(universe_name=settings["universe_name"],
                                       interval=settings["interval"],
                                       only_actionable=only_actionable)

if "rec_res" not in st.session_state:
    st.info("Set your options and click **Scan for recommendations**.")
    st.stop()

res = st.session_state["rec_res"]
ctx = st.session_state["rec_ctx"]
only_act = only_actionable

# --- Per-strategy sections --------------------------------------------------
total_picks = 0
for strat_name, df in res.items():
    st.divider()
    if df is None or df.empty:
        st.subheader(f"🔹 {strat_name}")
        st.info("No fresh signals in this window for this strategy/universe.")
        continue

    view = df.copy()
    if only_act:
        view = view[view["recommendation"] != "WATCH"]
    n_strong = (df["recommendation"] == "STRONG BUY").sum()
    n_buy = (df["recommendation"] == "BUY").sum()
    total_picks += n_strong + n_buy

    st.subheader(f"🔹 {strat_name}  ·  {n_strong} Strong Buy · {n_buy} Buy")
    if view.empty:
        st.caption("(only Watch-level signals — hidden by your filter)")
        continue

    # Colour the recommendation column.
    def _style(v):
        return {"STRONG BUY": "background-color:#1a7d1a;color:white",
                "BUY": "background-color:#2ca02c;color:white",
                "WATCH": "background-color:#eee;color:#555"}.get(v, "")
    show_cols = ["signal_date", "symbol", "recommendation", "score", "side",
                 "entry", "stop_loss", "sl_%", "target1", "target2", "target3",
                 "t1_%", "R:R(T1)", "probable_hold_days", "max_hold_days",
                 "RSI14", "ADX14", "MACD>Signal", "Close>EMA50", "Close>EMA200",
                 "Supertrend", "vol_ratio", "Williams%R",
                 "fund_gate", "earnings_in_days", "earnings_flag", "sector"]
    show_cols = [c for c in show_cols if c in view.columns]
    styled = view[show_cols].style.map(_style, subset=["recommendation"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- CSV: save-to-disk + download ---------------------------------------
    cc = st.columns([1, 1, 2])
    if cc[0].button(f"💾 Save CSV to disk", key=f"save_{strat_name}"):
        path = rec.save_csv(df, strat_name, ctx["universe_name"])
        st.success(f"Saved → `{path}`")
    cc[1].download_button(
        "⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
        file_name=f"{df['signal_date'].max()}__{strat_name.replace(' ','-').replace('%','R')}.csv",
        mime="text/csv", key=f"dl_{strat_name}")
    cc[2].caption("Save-to-disk writes to the `recommendations/` folder with the date + strategy + "
                  "universe in the filename, so a daily history builds up automatically.")

    # --- Per-stock chart drill-down -----------------------------------------
    with st.expander(f"📈 Chart a stock from {strat_name}", expanded=False):
        pick = st.selectbox("Stock", view["symbol"].tolist(), key=f"pick_{strat_name}")
        prow = view[view["symbol"] == pick].iloc[0]
        dfp = ui.load_ohlcv(pick, ctx["interval"])
        if dfp is None or dfp.empty:
            st.info("Price data unavailable.")
        else:
            enr = ind.enrich(dfp)
            enr_win = enr.tail(180)  # last ~9 months for context
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
                                subplot_titles=(f"{pick} — {prow['recommendation']} "
                                                f"(score {prow['score']})", "RSI(14)", "MACD"))
            fig.add_trace(go.Candlestick(
                x=enr_win.index, open=enr_win["Open"], high=enr_win["High"],
                low=enr_win["Low"], close=enr_win["Close"], name="Price"), row=1, col=1)
            for col, color in [("EMA50", "#ff7f0e"), ("EMA200", "#9467bd")]:
                if col in enr_win:
                    fig.add_trace(go.Scatter(x=enr_win.index, y=enr_win[col], name=col,
                                             line=dict(width=1, color=color)), row=1, col=1)
            # Entry / SL / target lines.
            fig.add_hline(y=prow["entry"], line=dict(color="blue", dash="dash"),
                          annotation_text=f"Entry ₹{prow['entry']}", row=1, col=1)
            fig.add_hline(y=prow["stop_loss"], line=dict(color="red", dash="dot"),
                          annotation_text=f"SL ₹{prow['stop_loss']}", row=1, col=1)
            for tk, col in [("target1", "#2ca02c"), ("target2", "#1a7d1a"), ("target3", "#0f5d0f")]:
                if tk in prow:
                    fig.add_hline(y=prow[tk], line=dict(color=col, dash="dot"),
                                  annotation_text=f"{tk} ₹{prow[tk]}", row=1, col=1)
            if "RSI14" in enr_win:
                fig.add_trace(go.Scatter(x=enr_win.index, y=enr_win["RSI14"], name="RSI",
                                         line=dict(color="#2ca02c")), row=2, col=1)
                fig.add_hline(y=70, line=dict(dash="dot", color="red"), row=2, col=1)
                fig.add_hline(y=30, line=dict(dash="dot", color="green"), row=2, col=1)
            if "MACD" in enr_win:
                fig.add_trace(go.Scatter(x=enr_win.index, y=enr_win["MACD"], name="MACD",
                                         line=dict(color="#1f77b4")), row=3, col=1)
                fig.add_trace(go.Scatter(x=enr_win.index, y=enr_win["MACD_SIGNAL"], name="Signal",
                                         line=dict(color="#ff7f0e")), row=3, col=1)
            fig.update_layout(height=640, xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h", y=1.03),
                              margin=dict(l=50, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"**Trade plan** — {prow['side'].upper()} · entry **₹{prow['entry']}** · "
                f"SL **₹{prow['stop_loss']}** ({prow['sl_%']}%) · "
                f"T1 **₹{prow['target1']}** ({prow['t1_%']}%) · R:R **{prow['R:R(T1)']}** · "
                f"probable hold **{prow.get('probable_hold_days','?')} days** "
                f"(max {prow.get('max_hold_days','?')}).")

st.divider()
if total_picks == 0:
    st.warning("No Buy/Strong-Buy signals right now across the enabled strategies. That itself is "
               "information — the setups aren't present today. Try a wider universe or 'relaxed' "
               "sensitivity in the sidebar.")
else:
    st.success(f"**{total_picks} actionable (Buy/Strong Buy) recommendations** across "
               f"{len([d for d in res.values() if d is not None and not d.empty])} strategies. "
               "Save each strategy's CSV to build a dated history.")
st.caption("Reminder: recommendations use current data and each strategy's rules; they are a "
           "starting point for your own analysis, not a guarantee. Honour your stop-loss.")
