"""Backtest Lab — layered selection flow + portfolio backtest.

Flow (top to bottom), driven by the sidebar timeframe & universe:
  Layer 1  Universe & timeframe
  Layer 2  Signals (which stocks fired)
  Layer 3  Fundamental gate (pass/fail + earnings risk)
  Layer 4  Sector rotation (kept vs dropped, with quadrant)
  Layer 5  Final ranked candidates  + "Why this stock?" explainer
  ── Run portfolio backtest (capital / risk% / max positions / max hold) ──
  Results: ₹ equity, drawdown, rich trade log, capital-allocation panel

Single-symbol mode gives a quick per-stock R-multiple view with trade markers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nsewing import (backtest, charts, config, explain, performance,
                     pipeline, portfolio, sectors, ui)
from nsewing.strategies import enabled_strategies, get_strategy

st.set_page_config(page_title="Backtest Lab", page_icon="🧪", layout="wide")
settings = ui.sidebar_controls()

st.title("🧪 Backtest Lab")
st.caption("Watch the funnel: timeframe → signals → fundamentals → sector-in-trend → ranked picks, "
           "then backtest as a real ₹ portfolio. Stops now fire on the **daily close** (no more "
           "same-day shakeouts) and trades can hold up to your chosen max.")

mode = st.radio("Mode", ["Portfolio (layered funnel)", "Single symbol (quick view)"],
                horizontal=True)
strat_name = st.selectbox("Strategy", enabled_strategies())

# ==========================================================================
# SINGLE-SYMBOL QUICK VIEW
# ==========================================================================
if mode.startswith("Single"):
    all_syms = sorted(set(config.NIFTY50 + config.TOP25_LIQUID +
                          config.MIDCAP50 + config.SMALLCAP50 +
                          list(config.INDICES.values())))
    symbol = st.selectbox("Symbol", all_syms,
                          index=all_syms.index("RELIANCE.NS") if "RELIANCE.NS" in all_syms else 0)
    max_hold = st.slider("Max hold (bars/days)", 5, 90, 60)
    if st.button("Run single-symbol backtest", type="primary"):
        with st.spinner("Backtesting…"):
            strat = get_strategy(strat_name, sensitivity=settings["sensitivity"])
            df = ui.load_ohlcv(symbol, settings["interval"])
            bench = ui.load_bench(settings["interval"])
            res = backtest.run_backtest(strat, {symbol: df}, bench_close=bench,
                                        risk_pct=settings["risk_pct"], max_hold_override=max_hold)
        m = res["metrics"]
        if m["trades"] == 0:
            st.warning("No trades. Try 'relaxed' sensitivity, a longer history, or another symbol.")
        else:
            ui.metric_row(m, res["doc_stats"])
            st.plotly_chart(charts.equity_curve(res["equity"], ui.load_bench(settings["interval"])),
                            use_container_width=True)
            st.plotly_chart(charts.trade_chart(df, res["trades"], symbol), use_container_width=True)
            with st.expander("Trade log"):
                st.dataframe(res["trades"], use_container_width=True, hide_index=True)
    st.stop()

# ==========================================================================
# PORTFOLIO / LAYERED FUNNEL
# ==========================================================================
st.divider()
# Analysis window is fixed at 1 year for both stock filtering and the portfolio
# simulation on this page.
window_yrs = 1
st.caption("📅 **Analysis window: last 1 year (fixed).** Everything — Layers 2–5 **and** the backtest — "
           "is bounded to the last 1 year. Layer 5 lists every signal in this window that passed the "
           "filters (a stock can appear on several dates); that is exactly what the simulation trades. "
           "(Fundamentals use current data as a proxy; the signal and sector-rotation checks are "
           "point-in-time.)")

c = st.columns(4)
apply_fund = c[0].checkbox("Fundamental sanity gate", value=True,
                           help="Market cap > ₹5,000 Cr and Debt/Equity < 1.5.")
apply_growth = c[1].checkbox("QoQ growth filter + rank", value=True,
                             help="Purely fundamental (no technicals): scores quarter-on-quarter growth "
                                  "in Operating Profit (40%), EPS (35%), Revenue (25%). Stocks must show "
                                  "growth to trade, and candidates are RANKED by this score.")
apply_sector = c[2].checkbox("Sector-in-trend gate (point-in-time)", value=True)
st.caption("📈 **Long-only** — short signals are ignored. Each signal's sector is checked **as of its "
           "own date**. Ranking is by the **fundamental QoQ-growth score** when the growth filter is on.")

# --- Exit method: % stop/target (default) vs Reward:Risk vs strategy default --
st.markdown("**Exit method** — where the stop-loss and target sit")
ec = st.columns([1.2, 1, 1, 1])
use_pct = ec[0].checkbox("Use % Stop-loss / Target", value=True,
                         help="Place the stop a fixed % below entry and the target a multiple of "
                              "that above it — wider brackets so swing trades hold for weeks instead "
                              "of being shaken out in 1–2 days. Uncheck to use Reward:Risk instead.")
sl_choice = ec[1].selectbox("Stop-loss %", ["5%", "8%", "10%", "12%", "15%"], index=2,
                            disabled=not use_pct)
tmult_choice = ec[2].selectbox("Target = SL ×", ["1×", "1.5×", "2×", "3×"], index=2,
                               disabled=not use_pct,
                               help="Target distance as a multiple of the stop %. 10% SL × 2 = 20% target.")
rr_choice = ec[3].selectbox(
    "Reward : Risk (if % off)", ["Strategy default", "1:1", "1:2", "1:3", "1:4", "1:5"],
    index=0, disabled=use_pct,
    help="Used only when '% Stop-loss / Target' is unchecked.")

SL_MAP = {"5%": 0.05, "8%": 0.08, "10%": 0.10, "12%": 0.12, "15%": 0.15}
TM_MAP = {"1×": 1.0, "1.5×": 1.5, "2×": 2.0, "3×": 3.0}
RR_MAP = {"Strategy default": None, "1:1": 1, "1:2": 2, "1:3": 3, "1:4": 4, "1:5": 5}
if use_pct:
    pct_stop = SL_MAP[sl_choice]
    pct_target_mult = TM_MAP[tmult_choice]
    rr_target = None
    st.caption(f"Stop **{sl_choice}** below entry · target **{SL_MAP[sl_choice]*TM_MAP[tmult_choice]*100:.0f}%** "
               f"above entry (= {sl_choice} × {tmult_choice}).")
else:
    pct_stop = None
    pct_target_mult = 2.0
    rr_target = RR_MAP[rr_choice]

# --- Trailing stop-loss (activates once the trade goes green) ---------------
st.markdown("**Trailing stop-loss** — locks in profit once the trade turns green")
tc = st.columns([1.2, 1, 1])
use_trail = tc[0].checkbox("Enable trailing stop", value=False,
                           help="Once price rises above entry by the trail distance, the stop follows "
                                "the peak up (never down). It exits on a daily close below the trailed "
                                "level. The fixed target still applies; whichever hits first wins.")
trail_unit = tc[1].selectbox("Trail unit", ["₹ (rupees)", "% (percent)"], index=1,
                             disabled=not use_trail,
                             help="₹ = fixed rupee step (careful: ₹2 is huge for a ₹12 stock, tiny for "
                                  "MRF). % scales fairly across all price ranges.")
if trail_unit.startswith("₹"):
    trail_val_choice = tc[2].selectbox("Trail by ₹", ["1", "2", "5", "10", "20", "50"], index=2,
                                       disabled=not use_trail)
    trail_amount = float(trail_val_choice) if use_trail else None
    trail_is_pct = False
else:
    trail_val_choice = tc[2].selectbox("Trail by %", ["1%", "2%", "3%", "5%"], index=1,
                                       disabled=not use_trail)
    trail_amount = ({"1%": 0.01, "2%": 0.02, "3%": 0.03, "5%": 0.05}[trail_val_choice]
                    if use_trail else None)
    trail_is_pct = True
if use_trail:
    unit_txt = f"₹{trail_val_choice}" if not trail_is_pct else trail_val_choice
    st.caption(f"Trailing stop **{unit_txt}** below the running peak, active once the trade is up "
               f"by {unit_txt}. Ratchets up only — never loosens.")

run_funnel = st.button("① Build candidate funnel", type="primary")

if run_funnel or "funnel" in st.session_state:
    if run_funnel:
        with st.spinner("Running layered selection…"):
            st.session_state["funnel"] = pipeline.build_candidates(
                strat_name, settings["universe"], interval=settings["interval"],
                sensitivity=settings["sensitivity"], window_years=window_yrs,
                apply_fundamentals=apply_fund, apply_sector=apply_sector,
                apply_growth=apply_growth, long_only=True,
                rr_target=rr_target, pct_stop=pct_stop, pct_target_mult=pct_target_mult,
                trail_amount=trail_amount, trail_is_pct=trail_is_pct)
            st.session_state["funnel_settings"] = dict(strat=strat_name,
                                                       universe=settings["universe_name"],
                                                       window_yrs=window_yrs,
                                                       rr_target=rr_target,
                                                       pct_stop=pct_stop,
                                                       pct_target_mult=pct_target_mult)
    r = st.session_state["funnel"]

    # ---- Layer 1 -------------------------------------------------------
    l1 = r["layer1_universe"]
    with st.expander(f"Layer 1 — Universe & timeframe  ·  {len(l1['have_data'])} stocks, "
                     f"{settings['interval']}", expanded=True):
        st.write(f"**{settings['universe_name']}** — {len(l1['have_data'])} with enough history"
                 + (f", {len(l1['thin'])} skipped (too little data)" if l1["thin"] else ""))
        if l1["thin"]:
            st.caption("Skipped: " + ", ".join(l1["thin"]))

    # ---- Layer 2 -------------------------------------------------------
    l2 = r["layer2_signals"]
    with st.expander(f"Layer 2 — Signals  ·  {len(l2)} stocks fired", expanded=True):
        if l2.empty:
            st.info("No signals in this window. Widen the look-back or use 'relaxed' sensitivity.")
        else:
            st.dataframe(l2, use_container_width=True, hide_index=True)

    # ---- Layer 3 -------------------------------------------------------
    l3 = r["layer3_fund"]
    with st.expander(f"Layer 3 — Fundamentals & QoQ growth  ·  "
                     f"{(l3['gate']=='PASS').sum() if len(l3) else 0} passed", expanded=True):
        if l3.empty:
            st.caption("Fundamental / growth gate skipped (unchecked).")
        else:
            st.dataframe(l3, use_container_width=True, hide_index=True)
            st.caption("**growth_score** (0–100) = quarter-on-quarter growth in Operating Profit (40%), "
                       "EPS (35%), Revenue (25%) — rewards steady growers, no technicals. A stock must "
                       "score ≥ 50 (and pass the sanity gate: mcap > ₹5,000 Cr, D/E < 1.5) to trade. "
                       "OpInc_QoQ / EPS_QoQ = latest quarter's % change; Rev_up_qtrs = quarters revenue "
                       "rose out of the last 4. ⚠ = earnings inside the horizon.")

    # ---- Layer 4 -------------------------------------------------------
    l4 = r["layer4_sector"]
    kept, dropped = l4["kept"], l4["dropped"]
    with st.expander(f"Layer 4 — Sector rotation  ·  {len(kept)} kept, {len(dropped)} dropped",
                     expanded=True):
        cc = st.columns(2)
        cc[0].markdown("**✅ Kept — sector Leading/Improving**")
        cc[0].dataframe(kept if not kept.empty else pd.DataFrame({"info": ["(none / gate off)"]}),
                        use_container_width=True, hide_index=True)
        cc[1].markdown("**❌ Dropped — sector Lagging/Weakening**")
        cc[1].dataframe(dropped if not dropped.empty else pd.DataFrame({"info": ["(none)"]}),
                        use_container_width=True, hide_index=True)
        st.caption("Only stocks whose sector has money flowing in are traded (long strategies). "
                   "Dropped names are shown so the filtering is transparent.")

    # ---- Layer 5 -------------------------------------------------------
    l5 = r["layer5_ranked"]
    with st.expander(f"Layer 5 — Signals in the last {window_yrs}y that passed every filter  ·  "
                     f"{len(l5)}", expanded=True):
        if l5.empty:
            st.info("No signals survived all layers in this window. Loosen a gate or widen the window.")
        else:
            st.caption(f"Every signal in the last **{window_yrs} year(s)** that passed fundamentals + "
                       "sector-in-trend **as of its own date**. A stock appears once **per signal date**, "
                       "so names repeat — this is exactly the list the simulation below trades.")
            st.dataframe(l5, use_container_width=True, hide_index=True)
            # Why this stock?
            sym = st.selectbox("🔎 Why this stock? — pick a candidate",
                               sorted(l5["symbol"].unique().tolist()))
            if sym:
                ex = explain.explain(sym, strat_name, settings["interval"],
                                     sensitivity=settings["sensitivity"])
                if "error" in ex:
                    st.warning(ex["error"])
                else:
                    st.markdown(f"**{ex['fundamentals'].get('name', sym)}** — {ex['side'].upper()} "
                                f"signal on {ex['signal_date']} ({ex['interval']})")
                    for ch in ex["checks"]:
                        icon = "✅" if ch["pass"] else "❌"
                        extra = ""
                        if ch["value"] is not None:
                            extra = f" — value **{ch['value']}**" + (f" (need {ch['threshold']})" if ch["threshold"] else "")
                        st.markdown(f"{icon} {ch['check']}{extra}")
                    lv = ex["levels"]
                    st.markdown(f"**Trade plan:** entry ₹{lv['entry']} · stop ₹{lv['stop']} "
                                f"(risk ₹{lv['risk_per_share']}/sh) · targets ₹{lv['t1']}/₹{lv['t2']}/₹{lv['t3']} "
                                f"· R:R to T1 = **{lv['rr_t1']}** · ATR14 ₹{lv['atr14']}")
                    sec = ex["sector"]
                    st.markdown(f"**Sector:** {sec['name']} — *{sec['quadrant']}* "
                                f"({'in trend ✅' if sec['in_trend'] else 'weak ❌'})")
                    fu = ex["fundamentals"]
                    st.markdown(f"**Fundamentals:** P/E {fu['pe']} · mcap ₹{fu['mcap_cr']:,}Cr "
                                f"· D/E {fu['debt_eq']} · earnings in {fu['earnings_in_days']} days"
                                if fu.get("mcap_cr") else "**Fundamentals:** (unavailable)")

    # ---- Run portfolio backtest ---------------------------------------
    st.divider()
    st.subheader("② Backtest the funnel as a ₹ portfolio")
    st.caption(f"Simulates the **same last-{window_yrs}-year signals** shown in Layer 5. The sector gate "
               "is point-in-time (checked as of each signal's date); fundamentals use current data as a "
               "proxy over the window.")
    pc = st.columns(4)
    cap = pc[0].number_input("Capital (₹)", 50_000, value=int(settings["capital"]), step=50_000)
    rp = pc[1].slider("Risk per trade (%)", 0.25, 4.0, settings["risk_pct"] * 100, 0.25) / 100
    maxpos = pc[2].slider("Max concurrent positions", 1, 15, 6)
    maxhold = pc[3].slider("Max hold (days)", 5, 90, 60,
                           help="For a swing trader, keep this high (45–90). A short cap forces "
                                "exits before targets are reached and turns winners into small "
                                "time-stop losses — the usual cause of a losing curve.")
    if maxhold < 30:
        st.warning(f"⚠️ Max hold = {maxhold} days is short for swing trading. Targets need weeks to "
                   "hit; a short cap dumps positions early and usually drags the result negative. "
                   "Try 45–60+.")
    if use_pct:
        st.caption(f"Exit bracket: stop **{sl_choice}**, target **{SL_MAP[sl_choice]*pct_target_mult*100:.0f}%** "
                   "(computed from each entry price).")
    elif rr_target:
        st.caption(f"Reward:Risk fixed at **1:{rr_target}** — profit target sits {rr_target}× the "
                   "stop distance from entry.")

    if st.button("Run portfolio backtest", type="primary"):
        with st.spinner("Simulating the portfolio bar by bar (point-in-time sector gate)…"):
            strat = get_strategy(strat_name, sensitivity=settings["sensitivity"],
                                 rr_target=rr_target, pct_stop=pct_stop,
                                 pct_target_mult=pct_target_mult,
                                 trail_amount=trail_amount, trail_is_pct=trail_is_pct)
            bench = ui.load_bench(settings["interval"])
            start = pd.Timestamp.today().normalize() - pd.DateOffset(years=window_yrs)
            # Fundamentals still gate the universe (current-data proxy); the
            # sector gate is now applied per-signal-date inside the sim, so we
            # pass the sector history rather than pre-filtering by today's rotation.
            sec_interval = settings["interval"] if settings["interval"] in ("1d", "1wk") else "1d"
            sec_hist = sectors.quadrant_history(sec_interval) if apply_sector else None
            sym_secs = r.get("sym2sec") if apply_sector else None
            # Universe = fundamentally-sound stocks; the sector gate is applied
            # per-signal-date inside the sim (point-in-time), not pre-filtered.
            bt_universe = r.get("fund_pass", r["allowed"]) if apply_sector else r["allowed"]
            pres = portfolio.run_portfolio(
                strat, r["data_map"], bench_close=bench, capital=cap, risk_pct=rp,
                max_positions=maxpos, max_hold=maxhold, start_date=start,
                allowed_symbols=bt_universe, scores=r["scores"],
                symbol_sectors=sym_secs, sector_history=sec_hist)
        st.session_state["pres"] = pres
        # Remember the window start (tz-matched to the equity index) for trimming.
        eq_idx = pres["equity"].index
        ws = start
        if len(eq_idx) and getattr(eq_idx, "tz", None) is not None:
            ws = start.tz_localize(eq_idx.tz)
        st.session_state["bt_window_start"] = ws

    if "pres" in st.session_state:
        pres = st.session_state["pres"]
        m = pres["metrics"]
        if m["trades"] == 0:
            st.warning("No trades — loosen the gates or widen the look-back.")
        else:
            k = st.columns(4)
            k[0].metric("Final value", f"₹{m.get('final_value_rs',0):,.0f}",
                        f"{m['total_return']*100:+.1f}%")
            k[1].metric("Total P&L", f"₹{m.get('total_pnl_rs',0):,.0f}")
            k[2].metric("CAGR", f"{m['cagr']*100:+.1f}%")
            k[3].metric("Max drawdown", f"{m['max_dd']*100:.1f}%")
            k = st.columns(4)
            k[0].metric("Trades", m["trades"])
            k[1].metric("Win rate", f"{m['win_rate']*100:.1f}%")
            k[2].metric("Profit factor", f"{m['profit_factor']:.2f}")
            k[3].metric("Avg hold (days)", f"{m['avg_bars_held']:.0f}")

            # Trim equity to the backtest window (drop the flat pre-start tail)
            # so charts and the index comparison align to the chosen 1/2/3/5-yr span.
            eq = pres["equity"]
            win_start = st.session_state.get("bt_window_start")
            if win_start is not None and not eq.empty:
                try:
                    eq = eq[eq.index >= win_start]
                except Exception:
                    pass
                if eq.empty:
                    eq = pres["equity"]

            st.plotly_chart(charts.equity_curve_rupees(eq, cap), use_container_width=True)
            st.plotly_chart(charts.drawdown_curve(eq), use_container_width=True)

            # --- Strategy vs broad-market indices (same window) ------------
            st.subheader("📈 Strategy vs NIFTY 50 / Midcap 150 / Smallcap 250 (same window)")
            with st.spinner("Fetching index returns…"):
                comp = performance.strategy_vs_indices(eq, cap)
            if not comp.empty:
                st.plotly_chart(charts.compare_curve(comp), use_container_width=True)
                st.dataframe(performance.compare_returns_table(comp),
                             use_container_width=True, hide_index=True)
                st.caption("Buy & hold on an index over the same period is the benchmark to beat. "
                           "If the strategy trails it, the edge isn't there — trade honestly on what "
                           "the data shows, not on hope.")

            st.subheader("Trade log")
            st.caption("Each row shows the plan (entry / **stop** / **target** price) and the outcome "
                       "(exit price, days held, ₹ and % P&L, why it exited).")
            _cols = ["symbol", "side", "entry_date", "exit_date", "days_held", "shares",
                     "entry_price", "stop_price", "target_price", "exit_price",
                     "invested_rs", "pnl_rs", "pnl_pct", "exit_reason", "score"]
            tlog = pres["trades"]
            st.dataframe(tlog[[c for c in _cols if c in tlog.columns]],
                         use_container_width=True, hide_index=True)
            st.caption("Note: an exit price beyond the stop (e.g. exit < stop on a long) means price "
                       "**closed** through the stop that day — a realistic gap, so that loss is a bit "
                       "larger than the planned risk. This is why realized reward:risk runs below the "
                       "nominal setting.")

            st.subheader("💰 How the money was divided")
            alloc = pres["alloc_log"]
            if not alloc.empty:
                busy_days = alloc.groupby("date").size().sort_values(ascending=False)
                pick = st.selectbox("Pick a signal day", busy_days.index.tolist())
                day = alloc[alloc["date"] == pick]
                st.dataframe(day[["symbol", "score", "entry", "stop", "decision", "reason",
                                  "shares", "invested_rs", "cash_after"]],
                             use_container_width=True, hide_index=True)
                st.caption("TAKEN = position opened. Skips explain why: no slot (position cap hit), "
                           "low rank, or not enough cash. Positions are sized by risk "
                           "(₹risk ÷ stop distance), capped by available cash per open slot.")
