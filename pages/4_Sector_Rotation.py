"""Sector Rotation page — RRG map + relative-strength heatmap + drill-down."""

from __future__ import annotations

import streamlit as st

from nsewing import config, screener, sectors, ui
from nsewing.strategies import enabled_strategies

st.set_page_config(page_title="Sector Rotation", page_icon="🧭", layout="wide")
settings = ui.sidebar_controls()

st.title("🧭 Sector Rotation")
st.caption("Where is money flowing? Leading/Improving sectors favour longs; Lagging/Weakening favour "
           "caution or shorts.")

interval = "1d" if settings["interval"] == "1h" else settings["interval"]

tab_rrg, tab_heat = st.tabs(["🔄 RRG map", "🌡️ RS heatmap"])

with tab_rrg:
    with st.spinner("Building relative-rotation graph…"):
        fig = sectors.rrg_figure(interval)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Top-right = **Leading**, top-left = **Improving**, bottom-left = **Lagging**, "
               "bottom-right = **Weakening**. The large dot is the latest reading; the tail shows "
               "the recent path.")
    rrg = sectors.rrg_coordinates(interval)
    if not rrg.empty:
        st.dataframe(rrg[["sector", "rs_ratio", "rs_momentum", "quadrant"]],
                     use_container_width=True, hide_index=True)

with tab_heat:
    with st.spinner("Computing trailing returns…"):
        tbl = sectors.rs_heatmap_table(interval)
    if tbl.empty:
        st.warning("No sector data available.")
    else:
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.caption("Sorted by 1-month return. RS vs NIFTY > 1.0 = outperforming the index.")

        # Drill-down: pick a sector, screen its constituents.
        st.divider()
        st.subheader("Drill-down: screen a sector's stocks")
        sec = st.selectbox("Sector", list(config.SECTORS.keys()))
        strat_name = st.selectbox("Strategy", enabled_strategies())
        if st.button("Screen sector"):
            with st.spinner("Screening constituents…"):
                df = screener.scan(strat_name, config.SECTORS[sec], interval=interval,
                                   sensitivity=settings["sensitivity"],
                                   recent_bars=30, with_fundamentals=False)
            if df.empty:
                st.info("No fresh setups in this sector right now.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
