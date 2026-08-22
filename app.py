"""NSE Swing-Trading System — Streamlit entry point.

Run:  streamlit run app.py
Pages live in ./pages (multipage app). This home page explains the system and
surfaces the live market snapshot.
"""

from __future__ import annotations

import streamlit as st

from nsewing import ui
from nsewing import _winpatch

# Silence the benign Windows asyncio ConnectionResetError (WinError 10054) that
# fires when a browser tab closes. Cosmetic only; no effect on any result.
_winpatch.apply()

st.set_page_config(page_title="NSE Swing System", page_icon="📈", layout="wide")

settings = ui.sidebar_controls()

st.title("📈 NSE Swing-Trading System")
st.caption("Backtest engine · daily screener · sector rotation — built on your three strategies.")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        """
        ### What this does
        - **Screener** — rank high-probability setups across a universe, with fundamentals and an
          earnings-risk flag.
        - **Backtest Lab** — layered funnel (signal → fundamentals → sector-in-trend) + a ₹ portfolio
          backtest over the last year, with equity, drawdown, trade log, and index comparison.
        - **Momentum Lab** — cross-sectional momentum + HMM regime filter + vol-sizing, validated
          out-of-sample (walk-forward), with a full per-trade ledger and per-stock drill-down.
        - **Recommendations** — today's actionable picks per strategy, ranked Strong Buy → Buy, with
          stop-loss, targets, indicators-at-signal, probable hold days, chart, and dated CSV export.
        - **Sector Rotation** — an RRG map + relative-strength heatmap to show where money is flowing.
        - **Playbook** — the written guide: selection, volume, the legal information edge, bear/sideways
          strategies, and risk.

        ### Honest ground rules
        1. **No system guarantees 4% in 7–30 days.** Edge = positive expectancy + tight risk + cutting
           losers fast. The 4% is a per-trade *target*, not a promise.
        2. **No "insider" news.** We use only public, legal signals. Trading on unpublished
           price-sensitive information violates SEBI PIT regulations.
        3. **Published backtest numbers are usually optimistic.** This tool reproduces your strategies
           honestly (costs, slippage, no look-ahead) so you see the *real* edge before risking money.
        """
    )
with col2:
    st.subheader("Live snapshot")
    if settings["vix"] is not None:
        r = settings["regime"]
        st.metric("India VIX", f"{settings['vix']:.2f}", r["label"])
        st.info(f"**Suggested risk/trade:** {r['risk_pct']*100:.1f}%\n\n{r['note']}")
    bench = ui.load_bench("1d")
    if bench is not None and len(bench) > 1:
        chg = (bench.iloc[-1] / bench.iloc[-2] - 1) * 100
        st.metric("NIFTY 50", f"{bench.iloc[-1]:,.0f}", f"{chg:+.2f}%")

st.divider()
st.markdown("👈 **Pick a page from the sidebar to begin.** Start with the **Backtest Lab**.")
