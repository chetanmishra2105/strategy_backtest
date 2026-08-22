"""Layered candidate-selection pipeline.

Runs the funnel the user described and returns EACH stage's survivors plus the
drop reasons, so the Backtest Lab can render it layer by layer:

    1. Data / timeframe   -> enough history?
    2. Signal             -> did the strategy fire (recently)?
    3. Fundamentals       -> passes the doc's gate? earnings soon?
    4. Sector-in-trend    -> is the stock's sector Leading/Improving (RRG)?
    5. Score & rank       -> composite score for ranking same-day signals

Reuses: strategy.generate_signals, indicators.enrich, fundamentals.*,
sectors.rrg_coordinates, screener._score_row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import data as datamod
from . import fundamentals as fund
from . import indicators as ind
from . import sectors as sectmod
from .screener import _score_row
from .strategies import Strategy, get_strategy

IN_TREND_QUADRANTS = {"Leading", "Improving"}


def _symbol_to_sector(symbols) -> dict[str, str]:
    """Resolve each symbol to a sector via sectors.sector_for_symbol (large-cap
    map first, then the yfinance GICS fallback), so midcaps/smallcaps also map."""
    return {s: sectmod.sector_for_symbol(s) for s in symbols}


def build_candidates(
    strategy_name: str,
    universe: list[str],
    interval: str = "1d",
    sensitivity: str = "relaxed",
    recent_bars: int = 10,
    apply_fundamentals: bool = True,
    apply_sector: bool = True,
    horizon: int = 30,
    rr_target: float | None = None,
    pct_stop: float | None = None,
    pct_target_mult: float = 2.0,
    long_only: bool = True,
    apply_growth: bool = True,
    trail_amount: float | None = None,
    trail_is_pct: bool = False,
    window_years: float = 2.0,
    **strategy_overrides,
) -> dict:
    """Return a dict of stage results for the layered UI.

    Keys: strategy, layer1_universe, layer2_signals (DataFrame), layer3_fund
    (DataFrame), layer4_sector {kept, dropped}, layer5_ranked (DataFrame),
    allowed (set of surviving symbols), scores (dict).

    ``window_years`` bounds EVERYTHING: only signals from the last N years are
    considered, and Layer 5 lists **every** in-window signal (a stock can appear
    multiple times, once per signal date) that also passed the fundamental gate
    and the sector-in-trend check **as of that signal's own date**. This makes
    the funnel show exactly what the simulation trades over the same window.

    ``rr_target`` (optional): fix targets at this reward:risk (1/2/3…).
    ``pct_stop`` (optional): fixed % stop from entry, target at pct×mult
    (takes precedence over rr_target). So candidate levels + backtest match.
    """
    strat: Strategy = get_strategy(strategy_name, sensitivity=sensitivity,
                                   rr_target=rr_target, pct_stop=pct_stop,
                                   pct_target_mult=pct_target_mult,
                                   trail_amount=trail_amount, trail_is_pct=trail_is_pct,
                                   **strategy_overrides)
    # Use a day-count offset so fractional windows (e.g. 0.5y) are supported;
    # DateOffset(years=...) rejects non-integers.
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=window_years * 365.25)
    bench = None
    if interval in ("1d", "1wk"):
        b = datamod.get_ohlcv(config.BENCHMARK, interval)
        bench = b["Close"] if not b.empty else None

    sym2sec = _symbol_to_sector(universe)

    # ---- Layer 1: data / timeframe ---------------------------------------
    have_data, thin = [], []
    data_map = {}
    for sym in universe:
        df = datamod.get_ohlcv(sym, interval)
        if df is None or df.empty or len(df) < 60:
            thin.append(sym)
            continue
        data_map[sym] = df
        have_data.append(sym)

    # ---- Layer 2: signal (ALL in-window signals, not just the latest) ----
    # all_sigs: one entry per (symbol, signal_date) inside the window. A stock
    # can appear many times. This is the raw trade list the sim will walk.
    sig_rows = []
    all_sigs = []            # list of per-signal dicts
    signalled_syms = set()   # symbols with >=1 in-window signal
    for sym in have_data:
        df = data_map[sym]
        sigs = strat.generate_signals(df, bench_close=bench)
        if sigs.empty:
            continue
        if long_only:
            sigs = sigs[sigs["side"] == "long"]
            if sigs.empty:
                continue
        # Window filter: only signals within the last N years (tz-safe compare).
        sd = pd.to_datetime(sigs["date"])
        cut = cutoff.tz_localize(sd.dt.tz) if getattr(sd.dt, "tz", None) is not None else cutoff
        sigs = sigs[sd.values >= cut.to_datetime64()]
        if sigs.empty:
            continue
        d = ind.enrich(df, bench_close=bench)
        signalled_syms.add(sym)
        for _, s in sigs.iterrows():
            last_row = d.loc[s["date"]]
            risk = abs(s["entry"] - s["stop"])
            reward = abs(s["entry"] - s["t1"])
            rr = reward / risk if risk > 0 else 0.0
            vr = float(last_row.get("VOL_RATIO", np.nan))
            all_sigs.append({
                "symbol": sym, "date": s["date"], "signal": s, "rr": rr,
                "vol_ratio": vr, "rsi": float(last_row.get("RSI14", np.nan)),
                "last_row": last_row,
            })
            sig_rows.append({
                "symbol": sym, "signal_date": s["date"].date(),
                "entry": round(s["entry"], 2), "stop": round(s["stop"], 2),
                "R:R(T1)": round(rr, 2),
                "vol_ratio": round(vr, 2) if np.isfinite(vr) else None,
            })
    layer2 = pd.DataFrame(sig_rows)

    # ---- Layer 3: fundamentals (sanity gate + QoQ growth) ---------------
    # growth_map holds each symbol's fundamental QoQ-growth score (0-100), used
    # both as a hard filter (must show growth) and as the RANKING score later.
    fund_rows = []
    fund_pass = set(signalled_syms)
    growth_map: dict[str, float] = {}
    if apply_fundamentals or apply_growth:
        fund_pass = set()
        for sym in signalled_syms:
            f = fund.get_fundamentals(sym)
            ok, reasons = fund.passes_gate(f) if apply_fundamentals else (True, [])
            ed = fund.earnings_in_days(f, horizon)
            g = fund.growth_score(sym) if apply_growth else {"score": None, "passes": True, "detail": {}}
            growth_map[sym] = g["score"] if g["score"] is not None else 0.0
            growth_ok = g["passes"] if apply_growth else True
            if not growth_ok and apply_growth:
                reasons = (reasons + [f"weak QoQ growth ({g['score']})"]) if reasons else [f"weak QoQ growth ({g['score']})"]
            passed = ok and growth_ok
            if passed:
                fund_pass.add(sym)
            gd = g.get("detail", {})
            fund_rows.append({
                "symbol": sym, "gate": "PASS" if passed else "FAIL",
                "growth_score": g["score"],
                "OpInc_QoQ": gd.get("operating_income", {}).get("latest_qoq_pct"),
                "EPS_QoQ": gd.get("eps", {}).get("latest_qoq_pct"),
                "Rev_up_qtrs": gd.get("revenue", {}).get("quarters_up"),
                "reason": "; ".join(reasons) if reasons else "",
                "mcap_cr": round(f["market_cap"] / 1e7, 0) if f.get("market_cap") else None,
                "debt_eq": round(f["debt_to_equity"], 2) if f.get("debt_to_equity") is not None else None,
                "earnings_in_days": ed,
                "earnings_flag": "⚠" if (ed is not None and 0 <= ed <= horizon) else "",
                "sector": sym2sec.get(sym, f.get("sector")),
            })
    layer3 = pd.DataFrame(fund_rows)

    # ---- Ranking score per symbol (growth-based when apply_growth) ------
    scores = {}
    for sym in signalled_syms:
        if apply_growth:
            scores[sym] = growth_map.get(sym, 0.0)
        else:
            # technical composite from that symbol's most recent in-window signal
            recent = [a for a in all_sigs if a["symbol"] == sym]
            info = recent[-1]
            vr = info["vol_ratio"] if np.isfinite(info["vol_ratio"]) else 1.0
            scores[sym] = _score_row(info["last_row"], info["signal"]["side"], info["rr"], vr)

    # ---- Layer 4 + 5: PER-SIGNAL sector gate & ranked candidates --------
    # For EACH in-window signal, check its sector's quadrant AS OF that signal's
    # date (point-in-time). Keep only fund-passed symbols whose sector was
    # Leading/Improving on that date. Layer 5 then lists every surviving signal
    # (a stock may appear multiple times) — exactly what the simulation trades.
    sec_hist = sectmod.quadrant_history(interval if interval in ("1d", "1wk") else "1d") \
        if apply_sector else {}

    kept, dropped, ranked_rows = [], [], []
    for a in all_sigs:
        sym = a["symbol"]
        if sym not in fund_pass:
            continue  # failed fundamentals/growth — not tradeable at all
        s = a["signal"]
        sec = sym2sec.get(sym)
        if apply_sector:
            q = sectmod.quadrant_on(sec_hist.get(sec), a["date"]) if sec else "Unknown"
        else:
            q = "Leading"  # gate off -> treat as in-trend
        base = {"symbol": sym, "signal_date": a["date"].date(),
                "sector": sec or "Unknown", "quadrant": q}
        if q in IN_TREND_QUADRANTS or (not apply_sector):
            kept.append(base)
            ranked_rows.append({
                "symbol": sym, "side": s["side"], "score": scores.get(sym, 0.0),
                "growth_score": growth_map.get(sym),
                "signal_date": a["date"].date(),
                "entry": round(s["entry"], 2), "stop": round(s["stop"], 2),
                "t1": round(s["t1"], 2), "t2": round(s["t2"], 2), "t3": round(s["t3"], 2),
                "R:R(T1)": round(a["rr"], 2),
                "sector": sec,
            })
        else:
            dropped.append(base)

    layer4 = {"kept": pd.DataFrame(kept), "dropped": pd.DataFrame(dropped)}
    layer5 = pd.DataFrame(ranked_rows) if ranked_rows else pd.DataFrame()
    if not layer5.empty:
        # Most recent first, then by growth/score — this is the sim's trade list.
        layer5 = layer5.sort_values(["signal_date", "score"], ascending=[False, False]).reset_index(drop=True)

    # Symbols that produced at least one in-window, fully-gated signal.
    sector_ok = set(layer5["symbol"]) if not layer5.empty else set()

    return {
        "strategy": strategy_name,
        "data_map": data_map,
        "window_years": window_years,
        "layer1_universe": {"have_data": have_data, "thin": thin},
        "layer2_signals": layer2,
        "layer3_fund": layer3,
        "layer4_sector": layer4,
        "layer5_ranked": layer5,
        "allowed": sector_ok,          # symbols with >=1 in-window gated signal
        "fund_pass": set(fund_pass),   # fundamentals-only set (sim universe)
        "sym2sec": sym2sec,            # symbol -> sector map (reuse in the sim)
        "scores": scores,
        "growth_map": growth_map,      # symbol -> fundamental QoQ-growth score
    }
