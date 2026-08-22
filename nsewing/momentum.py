"""Cross-sectional momentum engine (NEW — does not touch existing strategies).

The hedge-fund staple: instead of firing a per-stock technical signal, we RANK
the whole universe against each other and hold the strongest names, rebalancing
periodically. Trades are placed in **individual stocks** (never the index — the
index is only the benchmark to beat).

Design (all point-in-time / causal — no look-ahead):
  * On each rebalance date t, score every stock by its trailing momentum
    (e.g. 6-12 month return), optionally skipping the most recent month
    (the classic "12-1" momentum that skips short-term reversal).
  * Keep only names above their own long-term trend (e.g. Close > SMA(trend_ma))
    so we never buy a downtrending stock just because it fell least.
  * Hold the top ``top_n`` by score for the next period. Equal-weight by default;
    ``vol_scaled`` sizes inversely to each stock's volatility (risk parity-lite)
    so no single volatile name dominates portfolio risk.
  * An optional ``regime_series`` (from ``regime.py``, an HMM on the index) gates
    the WHOLE book to cash when the market regime is risk-off.

This is deliberately a **periodic-rebalance** simulator, distinct from the
event-driven ``backtest``/``portfolio`` engines — momentum is a portfolio-level
rotation, not a stop/target trade. Returns a ₹ equity curve + the same metrics
dict shape the rest of the app uses, so charts/comparison reuse works unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import indicators as ind


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def momentum_score(
    close: pd.Series, lookback: int = 252, skip: int = 21
) -> pd.Series:
    """Trailing total return over ``lookback`` bars, skipping the most recent
    ``skip`` bars (12-1 momentum). Uses only past data at each index point.

    Returns a Series aligned to ``close`` (NaN until warmed up).
    """
    past = close.shift(skip)
    ref = close.shift(lookback)
    return past / ref - 1.0


def _annualised_vol(close: pd.Series, window: int = 63) -> pd.Series:
    """Annualised daily-return volatility over ``window`` bars (for vol sizing)."""
    ret = close.pct_change()
    return ret.rolling(window).std() * np.sqrt(252)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def run_momentum(
    data_map: dict[str, pd.DataFrame],
    capital: float = None,
    start_date=None,
    end_date=None,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 15,
    rebalance: str = "ME",
    trend_ma: int = 200,
    use_trend_filter: bool = True,
    vol_scaled: bool = False,
    vol_window: int = 63,
    regime_series: pd.Series | None = None,
    risk_off_states: set | None = None,
    cost_frac: float | None = None,
    use_stops: bool = False,
    stop_pct: float = 0.15,
    target_mult: float = 2.0,
    data_map_ohlc: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Simulate a monthly-rebalanced cross-sectional momentum portfolio.

    Parameters mirror the levers a systematic equity fund tunes:
      lookback/skip     : momentum formation window (252/21 = "12-1 month").
      top_n             : how many names to hold.
      rebalance         : pandas offset alias for rebalance dates ('ME'=month-end,
                          'W-FRI'=weekly, 'QE'=quarterly).
      trend_ma          : long-term MA; with use_trend_filter, only hold names
                          trading above it (absolute-momentum overlay).
      vol_scaled        : True -> weight inversely to volatility; else equal-weight.
      regime_series     : optional index regime label per date (from regime.py).
                          When the label at a rebalance is in ``risk_off_states``
                          the book goes 100% to cash for that period.
      use_stops         : OPT-IN. When True, each held name also carries a hard
                          stop (stop_pct below entry) and a target (stop_pct ×
                          target_mult above entry); an intra-period daily High/Low
                          touch closes that leg to cash and is logged as
                          'stop_loss'/'target'. OFF by default so the pure-
                          momentum numbers are unchanged.
      data_map_ohlc     : optional {sym: OHLC df} used only when use_stops — needs
                          High/Low. Defaults to ``data_map`` (which already is OHLCV).

    Returns dict: equity (₹ Series), holdings (per-rebalance book), trades (a full
    per-position ledger: entry/exit date+price, days held, return %, ₹ P&L, exit
    reason), metrics (dict), rebalances (list of dates).

    Exit reasons in the ledger:
      rank_exit    — dropped out of the top-N at a rebalance,
      regime_exit  — book moved to cash because the regime turned risk-off,
      rebalance    — re-weighted at a rebalance while still held (closes the old
                     leg and opens a fresh one so each ledger row is one hold),
      stop_loss / target — only when use_stops is on,
      open_at_end  — still held on the last bar of the window.
    """
    capital = capital if capital is not None else config.DEFAULT_CAPITAL
    cost_frac = cost_frac if cost_frac is not None else (
        2 * config.COST_PER_SIDE + config.COST_SELL_EXTRA)
    risk_off_states = risk_off_states or {"Bear", "Risk-off"}

    # --- Build an aligned close-price panel across all symbols. -----------
    # Compute everything tz-naive internally (simplest, avoids mixed-tz joins),
    # but remember the source tz so the OUTPUT equity index can be restored to it
    # — the rest of the app (e.g. performance.strategy_vs_indices) compares the
    # equity index against tz-aware yfinance data and would otherwise error.
    closes = {}
    src_tz = None
    for sym, df in data_map.items():
        if df is None or df.empty or len(df) < max(lookback + skip, trend_ma) + 5:
            continue
        c = df["Close"].copy()
        c.index = pd.to_datetime(c.index)
        if getattr(c.index, "tz", None) is not None:
            if src_tz is None:
                src_tz = c.index.tz
            c.index = c.index.tz_localize(None)
        closes[sym] = c[~c.index.duplicated(keep="last")]
    if len(closes) < 2:
        return {"equity": pd.Series(dtype=float), "holdings": pd.DataFrame(),
                "metrics": _empty(), "rebalances": []}

    panel = pd.DataFrame(closes).sort_index()
    panel = panel.ffill()

    # Pre-compute score & trend & vol panels (vectorised per column).
    score_panel = panel.apply(lambda s: momentum_score(s, lookback, skip))
    trend_panel = panel.apply(lambda s: s.rolling(trend_ma).mean())
    vol_panel = panel.apply(lambda s: _annualised_vol(s, vol_window)) if vol_scaled else None

    # High/Low panels for intra-period stop/target checks (only when use_stops).
    high_panel = low_panel = None
    if use_stops:
        omap = data_map_ohlc or data_map
        highs, lows = {}, {}
        for sym in panel.columns:
            df = omap.get(sym)
            if df is None or df.empty or "High" not in df or "Low" not in df:
                continue
            h, lo = df["High"].copy(), df["Low"].copy()
            for s in (h, lo):
                s.index = pd.to_datetime(s.index)
                if getattr(s.index, "tz", None) is not None:
                    s.index = s.index.tz_localize(None)
            highs[sym] = h[~h.index.duplicated(keep="last")]
            lows[sym] = lo[~lo.index.duplicated(keep="last")]
        high_panel = pd.DataFrame(highs).reindex(panel.index).ffill()
        low_panel = pd.DataFrame(lows).reindex(panel.index).ffill()

    # --- Window the calendar. ---------------------------------------------
    idx = panel.index
    start_ts = _naive_ts(start_date)
    end_ts = _naive_ts(end_date)
    if start_ts is not None:
        idx = idx[idx >= start_ts]
    if end_ts is not None:
        idx = idx[idx <= end_ts]
    if len(idx) < 2:
        return {"equity": pd.Series(dtype=float), "holdings": pd.DataFrame(),
                "metrics": _empty(), "rebalances": []}

    # Rebalance dates = last available trading day on/before each period end.
    rebal_dates = _rebalance_dates(idx, rebalance)
    if not rebal_dates:
        rebal_dates = [idx[0]]

    reg = None
    if regime_series is not None and len(regime_series):
        reg = regime_series.copy()
        reg.index = pd.to_datetime(reg.index)
        if getattr(reg.index, "tz", None) is not None:
            reg.index = reg.index.tz_localize(None)

    # --- Walk the calendar, holding a fixed book between rebalances. ------
    # Book value / equity mechanics are UNCHANGED from the pure-momentum engine
    # (cash sleeve + per-symbol share legs) so the headline numbers are identical
    # when use_stops is off. A SEPARATE, observational ``ledger`` tracks each
    # continuous hold (entry date/price → exit) so we can show a per-trade table
    # without perturbing the equity curve.
    cash = capital
    legs: dict[str, float] = {}            # symbol -> shares held (book)
    ledger_open: dict[str, dict] = {}      # symbol -> open-hold record (observational)
    equity_points = []
    holdings_rows = []
    trades = []
    rebal_set = set(rebal_dates)
    last_date = idx[-1]

    def _ledger_close(sym, exit_date, exit_price, reason):
        pos = ledger_open.pop(sym)
        ep = pos["entry_price"]
        ret = (exit_price / ep - 1.0) if ep else 0.0
        gross = pos["invested"] * ret
        pnl = gross - cost_frac * pos["invested"]      # approx round-trip cost
        trades.append({
            "symbol": sym, "entry_date": pos["entry_date"].date(),
            "exit_date": pd.Timestamp(exit_date).date(),
            "days_held": int((pd.Timestamp(exit_date) - pos["entry_date"]).days),
            "entry_price": round(ep, 2),
            "stop_price": round(pos["stop"], 2) if pos["stop"] else None,
            "target_price": round(pos["target"], 2) if pos["target"] else None,
            "exit_price": round(float(exit_price), 2),
            "invested_rs": round(pos["invested"], 0),
            "pnl_rs": round(pnl, 0),
            "return_pct": round(ret * 100, 2),
            "exit_reason": reason,
            "entry_score": round(pos["entry_score"], 4) if pos["entry_score"] is not None and np.isfinite(pos["entry_score"]) else None,
            "entry_regime": pos["entry_regime"],
        })

    for date in idx:
        row = panel.loc[date]

        # --- Intra-period stop/target check (only when use_stops). --------
        # Moves the affected leg to cash in the BOOK and closes the ledger hold.
        if use_stops and legs and high_panel is not None:
            for sym in list(legs.keys()):
                pos = ledger_open.get(sym)
                if pos is None or date <= pos["entry_date"]:
                    continue  # no exit on the entry bar itself
                hi = high_panel.at[date, sym] if sym in high_panel.columns else np.nan
                lo = low_panel.at[date, sym] if sym in low_panel.columns else np.nan
                exit_px = reason = None
                if pos["target"] and np.isfinite(hi) and hi >= pos["target"]:
                    exit_px, reason = pos["target"], "target"
                elif pos["stop"] and np.isfinite(lo) and lo <= pos["stop"]:
                    exit_px, reason = pos["stop"], "stop_loss"
                if reason:
                    cash += legs[sym] * exit_px         # liquidate leg to cash
                    del legs[sym]
                    _ledger_close(sym, date, exit_px, reason)

        # --- Rebalance at period boundaries. ------------------------------
        if date in rebal_set:
            book_val = _book_value(cash, legs, row)
            weights_rs, picks, cash = _select_and_size(
                date, row, score_panel.loc[date], trend_panel.loc[date],
                vol_panel.loc[date] if vol_panel is not None else None,
                top_n, use_trend_filter, vol_scaled, reg, risk_off_states,
                book_val, cost_frac)
            regime_label = _regime_on(reg, date)
            new_syms = set(weights_rs.keys())

            # Ledger: close holds that left the book; keep continuous holds open.
            for sym in list(ledger_open.keys()):
                if sym not in new_syms:
                    px = row.get(sym, np.nan)
                    if np.isfinite(px):
                        reason = "rank_exit" if picks else "regime_exit"
                        _ledger_close(sym, date, px, reason)

            # Rebuild the BOOK legs from ₹ weights (unchanged mechanics).
            legs = {}
            for sym, rs in weights_rs.items():
                px = row.get(sym, np.nan)
                if np.isfinite(px) and px > 0 and rs > 0:
                    legs[sym] = rs / px
                    # Ledger: open a new hold, or keep the existing one (update
                    # invested/stops to the fresh weight but preserve entry).
                    if sym not in ledger_open:
                        ledger_open[sym] = {
                            "entry_date": date, "entry_price": float(px),
                            "invested": rs,
                            "stop": float(px) * (1 - stop_pct) if use_stops else None,
                            "target": float(px) * (1 + stop_pct * target_mult) if use_stops else None,
                            "entry_score": float(score_panel.loc[date].get(sym, np.nan)),
                            "entry_regime": regime_label,
                        }
                    else:
                        ledger_open[sym]["invested"] = rs

            if picks:
                for sym in picks:
                    holdings_rows.append({
                        "rebalance_date": date.date(), "symbol": sym,
                        "score": round(float(score_panel.loc[date].get(sym, np.nan)), 4),
                        "weight_rs": round(weights_rs.get(sym, 0.0), 0),
                        "regime": regime_label})
            else:
                holdings_rows.append({
                    "rebalance_date": date.date(), "symbol": "(cash)",
                    "score": None, "weight_rs": round(book_val, 0),
                    "regime": regime_label})

        equity_points.append((date, _book_value(cash, legs, row)))

    # Close any still-open ledger holds at the last bar (observational only).
    final_row = panel.loc[last_date]
    for sym in list(ledger_open.keys()):
        px = final_row.get(sym, np.nan)
        if np.isfinite(px):
            _ledger_close(sym, last_date, px, "open_at_end")

    equity = pd.Series({d: v for d, v in equity_points}).sort_index()
    # Restore the source timezone on the output index so downstream comparisons
    # against tz-aware market data (indices) don't raise a dtype-mismatch.
    if src_tz is not None and not equity.empty:
        equity.index = equity.index.tz_localize(src_tz)
    holdings = pd.DataFrame(holdings_rows)
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("exit_date").reset_index(drop=True)
    metrics = _metrics_from_equity(equity, capital)
    _augment_trade_metrics(metrics, trades_df)
    return {"equity": equity, "holdings": holdings, "trades": trades_df,
            "metrics": metrics,
            "rebalances": [d.date() for d in rebal_dates]}


# --------------------------------------------------------------------------
# Live recommendations — today's top-ranked names (what the engine would buy now)
# --------------------------------------------------------------------------
_REBAL_DAYS = {"ME": 21, "W-FRI": 5, "QE": 63}   # approx bars per period → hold


def momentum_recommendations(
    data_map: dict[str, pd.DataFrame],
    capital: float = None,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 15,
    rebalance: str = "ME",
    trend_ma: int = 200,
    use_trend_filter: bool = True,
    vol_scaled: bool = False,
    vol_window: int = 63,
    regime_series: pd.Series | None = None,
    risk_off_states: set | None = None,
    use_stops: bool = False,
    stop_pct: float = 0.15,
    target_mult: float = 2.0,
    as_of=None,
) -> pd.DataFrame:
    """Return TODAY's ranked buy list from the momentum engine.

    This is the live analogue of one rebalance: as of the latest bar (or
    ``as_of``), score every stock, keep those above the trend filter with
    positive momentum, rank them, and mark the top ``top_n`` as buys. Ranking is
    by momentum score; the strongest are **STRONG BUY**, the rest of the top-N
    **BUY**, and the next names **WATCH** (just outside the cut).

    Each row carries: rank, recommendation, entry (last close), momentum score,
    a suggested stop-loss & target (from ``stop_pct``/``target_mult`` if set, else
    a sensible 15%/2× default so the columns are always populated), the portfolio
    weight the engine would assign, probable hold days (the rebalance period),
    trend/regime status, and the indicator readings now. If the regime is
    risk-off, the list is returned but flagged so the user knows the engine would
    actually sit in cash.
    """
    capital = capital if capital is not None else config.DEFAULT_CAPITAL
    risk_off_states = risk_off_states or {"Bear", "Risk-off"}
    # Populate SL/target columns even when use_stops is off (as guidance).
    sl_frac = stop_pct if use_stops else 0.15
    tgt_frac = sl_frac * (target_mult if use_stops else 2.0)

    # Build tz-naive close panel + High/Low kept aside for indicator enrich.
    closes = {}
    for sym, df in data_map.items():
        if df is None or df.empty or len(df) < max(lookback + skip, trend_ma) + 5:
            continue
        c = df["Close"].copy()
        c.index = pd.to_datetime(c.index)
        if getattr(c.index, "tz", None) is not None:
            c.index = c.index.tz_localize(None)
        closes[sym] = c[~c.index.duplicated(keep="last")]
    if len(closes) < 2:
        return pd.DataFrame()

    panel = pd.DataFrame(closes).sort_index().ffill()
    score_panel = panel.apply(lambda s: momentum_score(s, lookback, skip))
    trend_panel = panel.apply(lambda s: s.rolling(trend_ma).mean())
    vol_panel = panel.apply(lambda s: _annualised_vol(s, vol_window))

    # As-of date (latest available by default).
    idx = panel.index
    if as_of is not None:
        a = _naive_ts(as_of)
        idx = idx[idx <= a]
    if len(idx) == 0:
        return pd.DataFrame()
    date = idx[-1]

    reg_label = None
    if regime_series is not None and len(regime_series):
        reg = regime_series.copy()
        reg.index = pd.to_datetime(reg.index)
        if getattr(reg.index, "tz", None) is not None:
            reg.index = reg.index.tz_localize(None)
        reg_label = _regime_on(reg, date)
    regime_risk_off = reg_label in risk_off_states if reg_label else False

    prow = panel.loc[date]
    srow = score_panel.loc[date]
    trow = trend_panel.loc[date]
    vrow = vol_panel.loc[date]

    # Candidate set: finite score & price, positive momentum, (optional) uptrend.
    cands = []
    for sym in srow.index:
        sc, px, tm = srow.get(sym), prow.get(sym), trow.get(sym)
        if not (np.isfinite(sc) and np.isfinite(px) and px > 0):
            continue
        above_trend = np.isfinite(tm) and px > tm
        if use_trend_filter and not above_trend:
            continue
        if sc <= 0:
            continue
        cands.append((sym, float(sc), float(px), above_trend))
    if not cands:
        return pd.DataFrame()
    cands.sort(key=lambda x: x[1], reverse=True)

    # Vol-scaled weights across the top-N (mirrors the engine's sizing).
    picks = cands[:top_n]
    if vol_scaled:
        inv = {s: (1.0 / vrow.get(s)) if np.isfinite(vrow.get(s)) and vrow.get(s) > 1e-6 else 0.0
               for s, _, _, _ in picks}
        tot = sum(inv.values()) or 1.0
        wmap = {s: inv[s] / tot for s, _, _, _ in picks}
    else:
        wmap = {s: 1.0 / len(picks) for s, _, _, _ in picks}

    # Strong Buy = top third of the buy list (min 1); rest of top-N = Buy;
    # the next few names (just outside) = Watch.
    n_strong = max(1, len(picks) // 3)
    hold_days = _REBAL_DAYS.get(rebalance, 21)

    rows = []
    for rank, (sym, sc, px, above_trend) in enumerate(cands[:top_n + 5], 1):
        in_book = rank <= top_n
        if not in_book:
            bucket = "WATCH"
        elif rank <= n_strong:
            bucket = "STRONG BUY"
        else:
            bucket = "BUY"
        stop = px * (1 - sl_frac)
        target = px * (1 + tgt_frac)
        w = wmap.get(sym, 0.0)
        # Indicator snapshot from the source OHLCV (enrich needs H/L/V).
        snap = {}
        df = data_map.get(sym)
        if df is not None and not df.empty:
            try:
                enr = ind.enrich(df)
                # align to the as-of date (tz-aware original index)
                e_idx = enr.index
                if getattr(e_idx, "tz", None) is not None:
                    target_ts = date.tz_localize(e_idx.tz)
                else:
                    target_ts = date
                near = e_idx[e_idx <= target_ts]
                if len(near):
                    r = enr.loc[near[-1]]
                    snap = {
                        "RSI14": round(float(r.get("RSI14", np.nan)), 1),
                        "ADX14": round(float(r.get("ADX14", np.nan)), 1),
                        "MACD>Signal": bool(r.get("MACD", 0) > r.get("MACD_SIGNAL", 0)),
                        "Close>EMA200": bool(r.get("Close", 0) > r.get("EMA200", 0)),
                        "Supertrend": "up" if r.get("ST_DIR", 0) == 1 else "down",
                        "vol_ratio": round(float(r.get("VOL_RATIO", np.nan)), 2),
                    }
            except Exception:
                snap = {}
        rows.append({
            "as_of_date": date.date(),
            "rank": rank,
            "symbol": sym,
            "recommendation": bucket,
            "momentum_score": round(sc * 100, 1),   # % trailing return
            "entry": round(px, 2),
            "stop_loss": round(stop, 2),
            "sl_%": round(-sl_frac * 100, 1),
            "target": round(target, 2),
            "target_%": round(tgt_frac * 100, 1),
            "weight_%": round(w * 100, 1),
            "alloc_₹": round(capital * w, 0),
            "probable_hold_days": hold_days,
            "above_trend": above_trend,
            "regime": reg_label,
            **snap,
        })
    out = pd.DataFrame(rows)
    out.attrs["regime_risk_off"] = regime_risk_off
    out.attrs["regime_label"] = reg_label
    return out


# --------------------------------------------------------------------------
# Selection & sizing (one rebalance)
# --------------------------------------------------------------------------
def _select_and_size(date, price_row, score_row, trend_row, vol_row,
                     top_n, use_trend_filter, vol_scaled, reg, risk_off_states,
                     book_val, cost_frac):
    """Return (weights_rs, picks, cash_rs) for this rebalance.

    ``weights_rs`` maps each held symbol -> ₹ allocated; ``cash_rs`` is the
    un-deployed sleeve. Book value is conserved: Σ weights_rs + cash_rs == book_val
    minus the turnover cost drag.
    """
    # Regime gate: whole book to cash when risk-off.
    label = _regime_on(reg, date)
    if label is not None and label in risk_off_states:
        return {}, [], book_val

    # Candidate set: finite score, positive price, (optionally) above trend.
    cand = []
    for sym in score_row.index:
        sc = score_row.get(sym)
        px = price_row.get(sym)
        if not (np.isfinite(sc) and np.isfinite(px) and px > 0):
            continue
        if use_trend_filter:
            tm = trend_row.get(sym)
            if not (np.isfinite(tm) and px > tm):
                continue
        cand.append((sym, float(sc)))
    if not cand:
        return {}, [], book_val

    cand.sort(key=lambda x: x[1], reverse=True)
    picks = [s for s, sc in cand[:top_n] if sc > 0]  # only positive momentum
    if not picks:
        return {}, [], book_val

    # Weights.
    if vol_scaled and vol_row is not None:
        inv = {}
        for s in picks:
            v = vol_row.get(s)
            inv[s] = (1.0 / v) if (np.isfinite(v) and v > 1e-6) else 0.0
        tot = sum(inv.values())
        w = ({s: inv[s] / tot for s in picks} if tot > 0
             else {s: 1.0 / len(picks) for s in picks})
    else:
        w = {s: 1.0 / len(picks) for s in picks}

    # Deployable capital after a round-trip cost drag on turnover (approx: charge
    # cost on the full notional each rebalance — conservative).
    deployable = book_val * (1.0 - cost_frac)
    weights_rs = {s: deployable * frac for s, frac in w.items()}
    cash = book_val - sum(weights_rs.values())
    return weights_rs, picks, cash


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _naive_ts(x):
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize(None) if t.tzinfo is not None else t


def _rebalance_dates(idx: pd.DatetimeIndex, rule: str) -> list:
    """Last available trading day within each period of ``idx`` (e.g. month-end
    trading day for rule='ME'). These are the dates the book is rebuilt on."""
    s = pd.Series(idx, index=pd.DatetimeIndex(idx))
    grp = s.resample(rule).last().dropna()
    return [pd.Timestamp(x) for x in grp.tolist()]


def _regime_on(reg: pd.Series | None, date) -> str | None:
    if reg is None or reg.empty:
        return None
    sub = reg.loc[reg.index <= date]
    return str(sub.iloc[-1]) if len(sub) else None


def _book_value(cash: float, legs: dict, price_row) -> float:
    """Book value = cash sleeve + Σ shares×current price for each held leg.

    If a symbol has no finite price on ``date`` (e.g. not listed yet / gap), the
    leg is carried at its most recent contribution being 0 — but since legs are
    rebuilt every rebalance from live prices, stale legs don't accumulate."""
    val = float(cash)
    for sym, shares in legs.items():
        px = price_row.get(sym, np.nan)
        if np.isfinite(px) and px > 0:
            val += shares * px
    return val


def _empty() -> dict:
    return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
            "total_return": 0.0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "avg_bars_held": 0.0, "final_value_rs": 0.0,
            "total_pnl_rs": 0.0}


def _augment_trade_metrics(m: dict, trades: pd.DataFrame) -> None:
    """Fill trade-level metrics (win rate, profit factor, avg hold) from the
    ledger. Equity-derived fields (cagr/max_dd/…) are left as computed."""
    if trades is None or trades.empty:
        return
    pnl = trades["pnl_rs"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    m["trades"] = int(len(trades))
    m["win_rate"] = len(wins) / len(trades) if len(trades) else 0.0
    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    m["profit_factor"] = (gross_win / gross_loss) if gross_loss > 1e-9 else np.inf
    m["avg_bars_held"] = float(trades["days_held"].mean())
    m["expectancy"] = float(pnl.mean())


def _metrics_from_equity(equity: pd.Series, capital: float) -> dict:
    m = _empty()
    if equity is None or len(equity) < 2:
        return m
    start_v, end_v = float(equity.iloc[0]), float(equity.iloc[-1])
    m["total_return"] = end_v / start_v - 1.0 if start_v > 0 else 0.0
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    m["cagr"] = (end_v / start_v) ** (1 / years) - 1.0 if (years > 0 and start_v > 0) else 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    m["max_dd"] = float(dd.min())
    ret = equity.pct_change().dropna()
    m["sharpe"] = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 1e-9 else 0.0
    downside = ret[ret < 0]
    m["sortino"] = float(ret.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 1e-9 else 0.0
    m["final_value_rs"] = end_v
    m["total_pnl_rs"] = end_v - capital
    return m
