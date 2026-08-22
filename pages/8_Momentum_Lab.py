"""Momentum Lab — cross-sectional momentum + HMM regime + vol sizing, validated
out-of-sample with a walk-forward harness.

This is a NEW page. It does not touch the existing strategies, funnel, or
Backtest Lab. The engine here RANKS the whole universe and rotates into the
strongest names (trades individual stocks — the index is only the benchmark),
which is the systematic-fund approach to beating a midcap index.

Read it top to bottom:
  ① Settings — universe, window, momentum knobs, HMM states.
  ② Ablation ladder — Step 1 (momentum) → +HMM regime → +vol sizing, EVERY row
     measured on out-of-sample walk-forward test folds. CAGR *and* drawdown.
  ③ OOS equity curves vs the buy-&-hold index (the 25% bar to beat).
  ④ Per-fold table + current holdings + current market regime.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from nsewing import (charts, config, data as datamod, momentum, performance,
                     regime as rg, ui, walkforward as wf)

st.set_page_config(page_title="Momentum Lab", page_icon="🚀", layout="wide")
settings = ui.sidebar_controls()

st.title("🚀 Momentum Lab — cross-sectional momentum, validated out-of-sample")
st.caption("Ranks the whole universe by trailing momentum, holds the strongest names, and rebalances "
           "monthly — trading **individual stocks** (the index is only the benchmark). Layer an **HMM "
           "regime filter** (a Markov model on the index) and **volatility-scaled sizing** on top, and "
           "measure each layer's contribution on **walk-forward, out-of-sample** data — the only numbers "
           "worth trusting.")

with st.expander("ℹ️ How to read this (and why it's honest)", expanded=False):
    st.markdown(
        "- **Trades are in individual stocks.** Cross-sectional momentum ranks every stock against "
        "the others each month and holds the top N. The index can't be ranked against itself — it's "
        "the **benchmark to beat** (your ~25% midcap bar).\n"
        "- **The three pieces are one system, not three strategies.** Momentum is the engine; the HMM "
        "regime filter decides *when* to be invested; vol-scaling decides *how much* per name. Only "
        "momentum trades on its own.\n"
        "- **Full transparency (Step ⑤).** Every out-of-sample trade is listed — stock, entry/exit "
        "date, days held, return, and WHY it exited (rank_exit = fell out of the top-N; regime_exit = "
        "went to cash in a Bear regime; stop_loss/target = only if you turn on the optional bands). "
        "Pick any stock to see its chart with the exact buy/sell points and the indicators at entry.\n"
        "- **Every number is out-of-sample.** The walk-forward harness trains on 9 months, tests on the "
        "next 3, and rolls forward. The reported curve is stitched from **test windows only** — what a "
        "live trader would actually have earned, not a curve fit to the past.\n"
        "- **Watch CAGR *and* drawdown.** The regime/vol layers usually don't add raw CAGR — they cut "
        "drawdown, which is what lets you hold the position. A layer that improves neither gets dropped.")

# ==========================================================================
# ① Settings
# ==========================================================================
st.divider()
st.subheader("① Settings")
c = st.columns(4)
n_years = c[0].selectbox("History window (years)", [3, 5, 7, 10], index=1)
top_n = c[1].slider("Hold top N stocks", 5, 30, 12)
lookback_m = c[2].selectbox("Momentum lookback (months)", [3, 6, 9, 12], index=3)
rebalance = c[3].selectbox("Rebalance", ["Monthly", "Weekly", "Quarterly"], index=0)

c = st.columns(4)
skip_recent = c[0].checkbox("Skip most-recent month (12-1)", value=True,
                            help="Classic momentum skips the latest ~21 days to avoid short-term "
                                 "mean-reversion contaminating the signal.")
trend_filter = c[1].checkbox("Trend filter (hold only above 200-DMA)", value=True)
n_states = c[2].selectbox("HMM regime states", [2, 3, 4], index=1,
                          help="Hidden Markov states fit on the index: 2=Bear/Bull, "
                               "3=Bear/Neutral/Bull, 4 adds Strong-Bull.")
train_test = c[3].selectbox("Walk-forward folds", ["9mo train / 3mo test",
                                                   "12mo train / 3mo test",
                                                   "6mo train / 3mo test"], index=0)

REB_MAP = {"Monthly": "ME", "Weekly": "W-FRI", "Quarterly": "QE"}
TT_MAP = {"9mo train / 3mo test": (9, 3), "12mo train / 3mo test": (12, 3),
          "6mo train / 3mo test": (6, 3)}
train_m, test_m = TT_MAP[train_test]
lookback = int(lookback_m * 21)
skip = 21 if skip_recent else 0

mom_params = dict(top_n=top_n, lookback=lookback, skip=skip,
                  rebalance=REB_MAP[rebalance], use_trend_filter=trend_filter)

# --- Optional stop-loss / target overlay -----------------------------------
st.markdown("**Optional: hard stop-loss / target** (off by default)")
sc = st.columns([1.4, 1, 1])
use_stops = sc[0].checkbox("Add SL / target to each holding", value=False,
                           help="Momentum normally exits a stock when it drops out of the top-N "
                                "ranking or the regime turns risk-off (exit reasons rank_exit / "
                                "regime_exit). Tick this to ALSO place a hard % stop and a target on "
                                "every position — you'll then see stop_loss / target exits in the "
                                "ledger. This CHANGES the returns, so it's opt-in.")
sl_pct = sc[1].selectbox("Stop-loss %", ["8%", "10%", "12%", "15%", "20%"], index=3,
                         disabled=not use_stops)
tmult = sc[2].selectbox("Target = SL ×", ["1×", "1.5×", "2×", "3×"], index=2, disabled=not use_stops)
if use_stops:
    _sl = {"8%": 0.08, "10%": 0.10, "12%": 0.12, "15%": 0.15, "20%": 0.20}[sl_pct]
    _tm = {"1×": 1.0, "1.5×": 1.5, "2×": 2.0, "3×": 3.0}[tmult]
    mom_params.update(use_stops=True, stop_pct=_sl, target_mult=_tm)
    st.caption(f"Each holding: stop **{sl_pct}** below entry · target **{_sl*_tm*100:.0f}%** above entry.")

run = st.button("Run walk-forward ablation", type="primary")

# ==========================================================================
# ② + ③ Ablation ladder (out-of-sample)
# ==========================================================================
if run:
    uni_name = settings["universe_name"]
    universe = settings["universe"]
    with st.spinner(f"Loading {len(universe)} symbols…"):
        dm = {}
        for s in universe:
            df = ui.load_ohlcv(s, "1d")
            if df is not None and not df.empty:
                dm[s] = df
        bench = ui.load_bench("1d")
    if len(dm) < 5:
        st.error("Not enough symbols with data in this universe.")
        st.stop()

    with st.spinner("Running walk-forward folds (momentum → +regime → +vol sizing)…"):
        lad = wf.ablation_ladder(
            dm, bench, capital=settings["capital"], train_months=train_m,
            test_months=test_m, n_years=n_years, momentum_params=mom_params,
            regime_states=n_states)
    st.session_state["mom_ladder"] = lad
    st.session_state["mom_ctx"] = dict(uni_name=uni_name, n_years=n_years,
                                       capital=settings["capital"])
    # Cache the data map for the holdings/regime panels below.
    st.session_state["mom_dm"] = dm
    st.session_state["mom_bench"] = bench
    st.session_state["mom_params"] = mom_params
    st.session_state["mom_states"] = n_states

if "mom_ladder" in st.session_state:
    lad = st.session_state["mom_ladder"]
    ctx = st.session_state["mom_ctx"]
    table = lad["table"]
    curves = lad["curves"]

    st.divider()
    st.subheader("② Ablation ladder — out-of-sample walk-forward")
    st.caption(f"{ctx['uni_name']} · {ctx['n_years']}-year history · each row is the stitched "
               "**test-fold** result. Watch CAGR **and** max drawdown move as each layer is added.")

    # Add Step 0 — buy & hold the benchmark indices over the same OOS span.
    oos_curve = next((v for v in curves.values() if v is not None and len(v)), None)
    baseline_rows = []
    if oos_curve is not None and len(oos_curve):
        comp = performance.strategy_vs_indices(oos_curve, ctx["capital"])
        for name in config.COMPARE_INDICES:
            if name in comp.columns:
                tot = float(comp[name].dropna().iloc[-1] - 1.0)
                days = max((oos_curve.index[-1] - oos_curve.index[0]).days, 1)
                cagr = (1 + tot) ** (365.25 / days) - 1.0
                dd = float((comp[name] / comp[name].cummax() - 1.0).min())
                baseline_rows.append({
                    "Configuration": f"0 · Buy & hold {name}",
                    "OOS CAGR %": round(cagr * 100, 1),
                    "OOS total %": round(tot * 100, 1),
                    "Max drawdown %": round(dd * 100, 1),
                    "Sharpe": None, "Calmar": None,
                    "Final ₹": round(ctx["capital"] * (1 + tot), 0)})
    full = pd.concat([pd.DataFrame(baseline_rows), table], ignore_index=True) \
        if baseline_rows else table

    st.dataframe(full, use_container_width=True, hide_index=True)
    st.caption("**Step 0** is the bar to beat (buy & hold the index). If no momentum row clears it on "
               "CAGR *or* on risk-adjusted return (Calmar), the edge isn't there — that's the honest "
               "verdict, not a bug. A layer (regime/vol) earns its place only if it improves CAGR or "
               "drawdown; if it improves neither, drop it.")

    # --- ③ OOS equity curves ------------------------------------------------
    st.subheader("③ Out-of-sample equity curves vs the index")
    fig = go.Figure()
    palette = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, (label, eq) in enumerate(curves.items()):
        if eq is None or eq.empty:
            continue
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=label,
                                 line=dict(width=2, color=palette[i % len(palette)])))
    # Overlay the benchmark index (normalised to starting capital).
    if oos_curve is not None and len(oos_curve):
        comp = performance.strategy_vs_indices(oos_curve, ctx["capital"])
        for name, color in [("NIFTY Midcap 150", "#ff7f0e"), ("NIFTY Smallcap 250", "#9467bd")]:
            if name in comp.columns:
                fig.add_trace(go.Scatter(x=comp.index, y=comp[name] * ctx["capital"],
                                         name=f"B&H {name}",
                                         line=dict(width=1.4, dash="dot", color=color)))
    fig.add_hline(y=ctx["capital"], line=dict(color="#bbb", dash="dash"),
                  annotation_text=f"Start ₹{ctx['capital']:,.0f}")
    fig.update_layout(height=460, title="Out-of-sample portfolio value (₹) — walk-forward test folds",
                      legend=dict(orientation="h", y=1.05),
                      margin=dict(l=60, r=20, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # --- ④ Per-fold detail, current holdings & regime -----------------------
    st.divider()
    st.subheader("④ Detail — folds, current holdings, market regime")

    # Rerun the full-config walk-forward once to expose per-fold rows for the
    # top (all-layers) configuration.
    dm = st.session_state.get("mom_dm")
    bench = st.session_state.get("mom_bench")
    mp = st.session_state.get("mom_params")
    ns = st.session_state.get("mom_states", 3)

    tabs = st.tabs(["Walk-forward folds", "Current holdings (today)", "Market regime (HMM)"])

    reg_labels = lad.get("regime_labels")  # reuse the ladder's PIT HMM (no refit)

    with tabs[0]:
        with st.spinner("Detailing folds for the full-stack configuration…"):
            full_wf = wf.walk_forward_momentum(
                dm, bench_close=bench, capital=ctx["capital"],
                train_months=train_m, test_months=test_m, n_years=ctx["n_years"],
                momentum_params={**mp, "vol_scaled": True}, use_regime=True,
                regime_states=ns, regime_labels=reg_labels)
        st.caption("Each fold trains on the earlier window and is scored only on its **test** window. "
                   "The stitched test returns are the out-of-sample curve above (Step 3).")
        if not full_wf["folds"].empty:
            st.dataframe(full_wf["folds"], use_container_width=True, hide_index=True)
        og = full_wf["oos_metrics"]; ig = full_wf["is_metrics"]
        cc = st.columns(2)
        cc[0].metric("Out-of-sample CAGR", f"{og['cagr']*100:+.1f}%")
        cc[1].metric("In-sample CAGR (full history)", f"{ig['cagr']*100:+.1f}%",
                     help="If in-sample is much higher than OOS, the strategy was flattering itself. "
                          "The gap is the optimism you'd have overpaid for.")

    with tabs[1]:
        st.caption("Today's book from the full-history momentum run (what you'd hold now).")
        with st.spinner("Computing current holdings…"):
            cur = momentum.run_momentum(
                dm, capital=ctx["capital"],
                start_date=pd.Timestamp.today().normalize() - pd.DateOffset(years=ctx["n_years"]),
                regime_series=reg_labels, vol_scaled=True, **mp)
        hold = cur["holdings"]
        if not hold.empty:
            latest = hold[hold["rebalance_date"] == hold["rebalance_date"].max()]
            st.dataframe(latest, use_container_width=True, hide_index=True)
        else:
            st.info("No holdings (universe went to cash / no qualifying names).")

    with tabs[2]:
        st.caption("The Hidden Markov Model's regime label for the index (point-in-time). The momentum "
                   "book goes to **cash** when the regime is risk-off (Bear).")
        reg = reg_labels if reg_labels is not None else (
            rg.regime_series_pit(bench, n_states=ns) if bench is not None and len(bench) else None)
        if reg is not None and len(reg):
            cur_label = reg.iloc[-1] if len(reg) else "—"
            st.metric("Current market regime", cur_label)
            # Plot price coloured by regime.
            rfig = go.Figure()
            rfig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="NIFTY",
                                      line=dict(color="#333", width=1)))
            colors = {"Bull": "#2ca02c", "Neutral": "#ff7f0e", "Bear": "#d62728",
                      "Strong-Bull": "#1a7d1a"}
            for lab, col in colors.items():
                mask = reg[reg == lab].index
                if len(mask):
                    rfig.add_trace(go.Scatter(
                        x=mask, y=bench.reindex(mask).values, name=lab, mode="markers",
                        marker=dict(size=4, color=col)))
            rfig.update_layout(height=380, title="NIFTY with HMM regime labels (point-in-time)",
                               legend=dict(orientation="h", y=1.05),
                               margin=dict(l=50, r=20, t=50, b=30))
            st.plotly_chart(rfig, use_container_width=True)
        else:
            st.info("Benchmark data unavailable.")

    # ======================================================================
    # ⑤ Trade ledger & per-stock drill-down (full transparency)
    # ======================================================================
    st.divider()
    st.subheader("⑤ Every trade — ledger & per-stock drill-down")
    st.caption("The complete out-of-sample trade list for a chosen configuration: which stock, entry "
               "and exit date, how long it was held, the return, and WHY it exited. Then pick any stock "
               "to see its price chart with the exact entry/exit points and the indicators at entry.")

    ledgers = lad.get("ledgers", {})
    cfg_choice = st.selectbox("Configuration to inspect",
                              list(ledgers.keys()),
                              index=len(ledgers) - 1 if ledgers else 0)
    ledger = ledgers.get(cfg_choice, pd.DataFrame())

    if ledger is None or ledger.empty:
        st.info("No trades recorded for this configuration in the window.")
    else:
        # --- Exit-reason summary --------------------------------------------
        st.markdown("**Exit-reason breakdown** — how trades ended")
        vc = ledger["exit_reason"].value_counts()
        reason_help = {
            "rank_exit": "dropped out of the top-N ranking at a rebalance",
            "regime_exit": "book moved to cash (regime turned risk-off / Bear)",
            "target": "hard target hit (only with SL/target on)",
            "stop_loss": "hard stop hit (only with SL/target on)",
            "open_at_end": "still held on the last day of the window",
        }
        rc = st.columns(len(vc))
        for i, (reason, cnt) in enumerate(vc.items()):
            won = ledger[(ledger["exit_reason"] == reason) & (ledger["pnl_rs"] > 0)]
            wr = (len(won) / cnt * 100) if cnt else 0
            rc[i].metric(reason, cnt, f"{wr:.0f}% win",
                         help=reason_help.get(reason, ""))

        k = st.columns(4)
        k[0].metric("Total trades", len(ledger))
        k[1].metric("Win rate", f"{(ledger['pnl_rs'] > 0).mean()*100:.0f}%")
        k[2].metric("Avg hold (days)", f"{ledger['days_held'].mean():.0f}")
        k[3].metric("Avg return / trade", f"{ledger['return_pct'].mean():+.1f}%")

        # --- Filterable full ledger -----------------------------------------
        st.markdown("**Full trade ledger** (out-of-sample)")
        fc = st.columns([1.2, 1.2, 1])
        syms_all = sorted(ledger["symbol"].unique().tolist())
        sym_filter = fc[0].multiselect("Filter by stock", syms_all, default=[])
        reason_filter = fc[1].multiselect("Filter by exit reason",
                                          sorted(ledger["exit_reason"].unique().tolist()),
                                          default=[])
        outcome = fc[2].selectbox("Outcome", ["All", "Winners", "Losers"], index=0)
        view = ledger.copy()
        if sym_filter:
            view = view[view["symbol"].isin(sym_filter)]
        if reason_filter:
            view = view[view["exit_reason"].isin(reason_filter)]
        if outcome == "Winners":
            view = view[view["pnl_rs"] > 0]
        elif outcome == "Losers":
            view = view[view["pnl_rs"] <= 0]
        led_cols = ["fold", "symbol", "entry_date", "exit_date", "days_held",
                    "entry_price", "stop_price", "target_price", "exit_price",
                    "invested_rs", "pnl_rs", "return_pct", "exit_reason",
                    "entry_score", "entry_regime"]
        st.dataframe(view[[c for c in led_cols if c in view.columns]],
                     use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download this ledger (CSV)",
                           ledger.to_csv(index=False).encode("utf-8"),
                           file_name=f"momentum_trades_{cfg_choice[:1]}.csv", mime="text/csv")

        # --- Per-stock dynamic chart ----------------------------------------
        st.markdown("**🔎 Drill into one stock** — price with entry/exit points + indicators at entry")
        pick_sym = st.selectbox("Stock", syms_all,
                                index=syms_all.index(ledger.iloc[-1]["symbol"])
                                if ledger.iloc[-1]["symbol"] in syms_all else 0)
        dm = st.session_state.get("mom_dm", {})
        sym_trades = ledger[ledger["symbol"] == pick_sym].reset_index(drop=True)
        df_sym = dm.get(pick_sym)
        if df_sym is None or df_sym.empty:
            st.info("Price data unavailable for this stock.")
        else:
            from nsewing import indicators as ind
            enr = ind.enrich(df_sym)
            # Window the chart to the traded span (+ padding) for readability.
            e0 = pd.Timestamp(sym_trades["entry_date"].min())
            e1 = pd.Timestamp(sym_trades["exit_date"].max())
            idx_tz = getattr(enr.index, "tz", None)
            if idx_tz is not None:
                e0 = e0.tz_localize(idx_tz); e1 = e1.tz_localize(idx_tz)
            pad = pd.Timedelta(days=45)
            enr_win = enr.loc[(enr.index >= e0 - pad) & (enr.index <= e1 + pad)]

            fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                 row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
                                 subplot_titles=(f"{pick_sym} — price with trades", "RSI(14)", "MACD"))
            fig2.add_trace(go.Candlestick(
                x=enr_win.index, open=enr_win["Open"], high=enr_win["High"],
                low=enr_win["Low"], close=enr_win["Close"], name="Price"), row=1, col=1)
            for col, color in [("EMA50", "#ff7f0e"), ("EMA200", "#9467bd")]:
                if col in enr_win:
                    fig2.add_trace(go.Scatter(x=enr_win.index, y=enr_win[col], name=col,
                                              line=dict(width=1, color=color)), row=1, col=1)
            # Entry / exit markers.
            def _to_idx_ts(d):
                t = pd.Timestamp(d)
                return t.tz_localize(idx_tz) if idx_tz is not None else t
            ent_x = [_to_idx_ts(d) for d in sym_trades["entry_date"]]
            fig2.add_trace(go.Scatter(x=ent_x, y=sym_trades["entry_price"], mode="markers",
                                      name="Entry", marker=dict(color="blue", size=11,
                                      symbol="triangle-up", line=dict(width=1, color="black"))),
                           row=1, col=1)
            win_t = sym_trades[sym_trades["pnl_rs"] > 0]
            los_t = sym_trades[sym_trades["pnl_rs"] <= 0]
            fig2.add_trace(go.Scatter(x=[_to_idx_ts(d) for d in win_t["exit_date"]],
                                      y=win_t["exit_price"], mode="markers", name="Exit (win)",
                                      marker=dict(color="green", size=11, symbol="x")), row=1, col=1)
            fig2.add_trace(go.Scatter(x=[_to_idx_ts(d) for d in los_t["exit_date"]],
                                      y=los_t["exit_price"], mode="markers", name="Exit (loss)",
                                      marker=dict(color="red", size=11, symbol="x")), row=1, col=1)
            if "RSI14" in enr_win:
                fig2.add_trace(go.Scatter(x=enr_win.index, y=enr_win["RSI14"], name="RSI",
                                          line=dict(color="#2ca02c")), row=2, col=1)
                fig2.add_hline(y=70, line=dict(dash="dot", color="red"), row=2, col=1)
                fig2.add_hline(y=30, line=dict(dash="dot", color="green"), row=2, col=1)
            if "MACD" in enr_win:
                fig2.add_trace(go.Scatter(x=enr_win.index, y=enr_win["MACD"], name="MACD",
                                          line=dict(color="#1f77b4")), row=3, col=1)
                fig2.add_trace(go.Scatter(x=enr_win.index, y=enr_win["MACD_SIGNAL"], name="Signal",
                                          line=dict(color="#ff7f0e")), row=3, col=1)
            fig2.update_layout(height=680, xaxis_rangeslider_visible=False,
                               legend=dict(orientation="h", y=1.03),
                               margin=dict(l=50, r=20, t=50, b=20))
            st.plotly_chart(fig2, use_container_width=True)

            # Indicators AT ENTRY for each trade in this stock.
            st.markdown(f"**Indicator readings at each entry — {pick_sym}**")
            snap_rows = []
            for _, t in sym_trades.iterrows():
                ets = _to_idx_ts(t["entry_date"])
                near = enr.index[enr.index <= ets]
                if not len(near):
                    continue
                r = enr.loc[near[-1]]
                snap_rows.append({
                    "entry_date": t["entry_date"], "exit_date": t["exit_date"],
                    "days_held": t["days_held"], "return_%": t["return_pct"],
                    "exit_reason": t["exit_reason"],
                    "Close": round(r["Close"], 2),
                    "RSI14": round(r.get("RSI14", float("nan")), 1),
                    "MACD>Signal": bool(r.get("MACD", 0) > r.get("MACD_SIGNAL", 0)),
                    "Close>EMA50": bool(r.get("Close", 0) > r.get("EMA50", 0)),
                    "Close>EMA200": bool(r.get("Close", 0) > r.get("EMA200", 0)),
                    "ADX14": round(r.get("ADX14", float("nan")), 1),
                    "vol/VOLMA20": round(r.get("VOL_RATIO", float("nan")), 2),
                    "momentum_score": t.get("entry_score"),
                    "regime": t.get("entry_regime"),
                })
            if snap_rows:
                st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)
                st.caption("These are the indicator values on the entry bar — so you can see what the "
                           "stock looked like (trend, momentum, volume, regime) at the moment the "
                           "strategy bought it, and connect that to how the trade turned out.")

    # ======================================================================
    # ⑥ Today's buy list — ranked recommendations from the momentum engine
    # ======================================================================
    st.divider()
    st.subheader("⑥ Today's buy list — ranked Strong Buy → Buy")
    st.caption("As of the latest bar, the momentum engine ranks the whole universe and would buy the "
               "top names. Pick which configuration to use — the ranking is the same; **HMM regime** "
               "tells you whether the engine would actually deploy (or sit in cash), and **vol-scaled** "
               "changes the suggested weights. Every pick shows entry, stop-loss, target, weight, "
               "probable hold days and the indicators now.")

    rec_dm = st.session_state.get("mom_dm", {})
    rec_bench = st.session_state.get("mom_bench")
    rec_mp = dict(st.session_state.get("mom_params", {}))
    rec_ns = st.session_state.get("mom_states", 3)
    reg_lab = lad.get("regime_labels")

    rc = st.columns(3)
    which = rc[0].selectbox("Configuration",
                            ["1 · Momentum only", "2 · + HMM regime filter",
                             "3 · + HMM + vol-scaled sizing"], index=2)
    rec_topn = rc[1].slider("How many to list", 5, 30, int(rec_mp.get("top_n", 12)), key="rec_topn")
    rec_stops = rc[2].checkbox("Show SL/target from % bands", value=bool(rec_mp.get("use_stops", False)),
                               help="If your run used the optional SL/target, those levels are shown; "
                                    "otherwise a 15% stop / 2× target is shown as guidance.")

    use_reg = which.startswith(("2", "3"))
    use_vol = which.startswith("3")
    rec_params = dict(
        lookback=rec_mp.get("lookback", 252), skip=rec_mp.get("skip", 21),
        top_n=rec_topn, rebalance=rec_mp.get("rebalance", "ME"),
        use_trend_filter=rec_mp.get("use_trend_filter", True),
        vol_scaled=use_vol,
        use_stops=rec_mp.get("use_stops", False) or rec_stops,
        stop_pct=rec_mp.get("stop_pct", 0.15),
        target_mult=rec_mp.get("target_mult", 2.0),
    )
    if st.button("Generate today's buy list", type="primary"):
        with st.spinner("Ranking the universe as of today…"):
            recdf = momentum.momentum_recommendations(
                rec_dm, capital=ctx["capital"],
                regime_series=reg_lab if use_reg else None, **rec_params)
        st.session_state["mom_reclist"] = recdf
        st.session_state["mom_recwhich"] = which

    if "mom_reclist" in st.session_state:
        recdf = st.session_state["mom_reclist"]
        if recdf is None or recdf.empty:
            st.info("No qualifying names right now (nothing above the trend filter with positive "
                    "momentum). That itself is a signal — the setups aren't present today.")
        else:
            if recdf.attrs.get("regime_risk_off"):
                st.warning(f"⚠️ The HMM regime is **{recdf.attrs.get('regime_label')}** (risk-off) as of "
                           "today — with the regime filter on, the engine would hold **cash**, not buy "
                           "these. The ranked list is shown for information; treat it as a watchlist "
                           "until the regime turns Bull/Neutral.")
            n_sb = (recdf["recommendation"] == "STRONG BUY").sum()
            n_b = (recdf["recommendation"] == "BUY").sum()
            st.markdown(f"**{n_sb} Strong Buy · {n_b} Buy** · as of {recdf['as_of_date'].iloc[0]}")

            def _stylerec(v):
                return {"STRONG BUY": "background-color:#1a7d1a;color:white",
                        "BUY": "background-color:#2ca02c;color:white",
                        "WATCH": "background-color:#eee;color:#555"}.get(v, "")
            rec_cols = ["as_of_date", "rank", "symbol", "recommendation", "momentum_score",
                        "entry", "stop_loss", "sl_%", "target", "target_%", "weight_%",
                        "alloc_₹", "probable_hold_days", "above_trend", "regime",
                        "RSI14", "ADX14", "MACD>Signal", "Close>EMA200", "Supertrend", "vol_ratio"]
            rec_cols = [c for c in rec_cols if c in recdf.columns]
            st.dataframe(recdf[rec_cols].style.map(_stylerec, subset=["recommendation"]),
                         use_container_width=True, hide_index=True)

            # CSV: save-to-disk (dated) + download.
            from nsewing import recommend as _rec
            cc = st.columns([1, 1, 2])
            if cc[0].button("💾 Save buy list to disk"):
                lbl = f"Momentum-{st.session_state.get('mom_recwhich','cfg')[:1]}"
                path = _rec.save_csv(recdf.rename(columns={"as_of_date": "signal_date"}),
                                     lbl, ctx["uni_name"])
                st.success(f"Saved → `{path}`")
            cc[1].download_button(
                "⬇️ Download buy list (CSV)", recdf.to_csv(index=False).encode("utf-8"),
                file_name=f"{recdf['as_of_date'].iloc[0]}__momentum_buylist.csv", mime="text/csv")
            cc[2].caption("Save-to-disk writes a dated file to `recommendations/` so you accumulate a "
                          "day-by-day history of what the engine flagged.")

            # Per-stock chart with entry / SL / target lines.
            st.markdown("**📈 Chart a pick** — price with entry / stop-loss / target lines")
            psym = st.selectbox("Stock", recdf["symbol"].tolist(), key="rec_pick")
            prow = recdf[recdf["symbol"] == psym].iloc[0]
            pdf = rec_dm.get(psym)
            if pdf is None or pdf.empty:
                st.info("Price data unavailable.")
            else:
                from nsewing import indicators as _ind
                enrp = _ind.enrich(pdf).tail(180)
                figr = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     row_heights=[0.72, 0.28], vertical_spacing=0.04,
                                     subplot_titles=(f"{psym} — {prow['recommendation']} "
                                                     f"(score {prow['momentum_score']}%)", "RSI(14)"))
                figr.add_trace(go.Candlestick(
                    x=enrp.index, open=enrp["Open"], high=enrp["High"],
                    low=enrp["Low"], close=enrp["Close"], name="Price"), row=1, col=1)
                for col, color in [("EMA50", "#ff7f0e"), ("EMA200", "#9467bd")]:
                    if col in enrp:
                        figr.add_trace(go.Scatter(x=enrp.index, y=enrp[col], name=col,
                                                  line=dict(width=1, color=color)), row=1, col=1)
                figr.add_hline(y=prow["entry"], line=dict(color="blue", dash="dash"),
                               annotation_text=f"Entry ₹{prow['entry']}", row=1, col=1)
                figr.add_hline(y=prow["stop_loss"], line=dict(color="red", dash="dot"),
                               annotation_text=f"SL ₹{prow['stop_loss']}", row=1, col=1)
                figr.add_hline(y=prow["target"], line=dict(color="#2ca02c", dash="dot"),
                               annotation_text=f"Target ₹{prow['target']}", row=1, col=1)
                if "RSI14" in enrp:
                    figr.add_trace(go.Scatter(x=enrp.index, y=enrp["RSI14"], name="RSI",
                                              line=dict(color="#2ca02c")), row=2, col=1)
                    figr.add_hline(y=70, line=dict(dash="dot", color="red"), row=2, col=1)
                    figr.add_hline(y=30, line=dict(dash="dot", color="green"), row=2, col=1)
                figr.update_layout(height=560, xaxis_rangeslider_visible=False,
                                   legend=dict(orientation="h", y=1.04),
                                   margin=dict(l=50, r=20, t=50, b=20))
                st.plotly_chart(figr, use_container_width=True)
                st.markdown(
                    f"**Plan:** BUY **{psym}** · entry **₹{prow['entry']}** · SL **₹{prow['stop_loss']}** "
                    f"({prow['sl_%']}%) · target **₹{prow['target']}** ({prow['target_%']}%) · "
                    f"suggested weight **{prow['weight_%']}%** (≈₹{prow['alloc_₹']:,.0f}) · "
                    f"probable hold **{prow['probable_hold_days']} days**.")

else:
    st.info("Set your options above and click **Run walk-forward ablation**.")
