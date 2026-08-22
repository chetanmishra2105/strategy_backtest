"""Portfolio backtester — models real rupees, position sizing, and how capital
is divided when several stocks signal on the same day.

Difference from ``backtest.run_backtest`` (R-multiple mode): this walks the
UNION calendar of all symbols day by day, holds at most ``max_positions`` open
trades, sizes each by risk (₹ risk ÷ per-share risk), and books ₹ P&L back to a
cash balance. It answers "how was the money split" and produces a ₹ equity curve
and a rich, per-trade log with invested amount, days held, and ₹/%% P&L.

Exit rules are shared with the single-symbol engine via ``backtest.resolve_exit``
(close-based stops, no entry-bar stop-out, T1/T2/T3 scaling) so both engines
agree on when a trade ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt
from . import config
from . import sectors as sectmod
from .strategies import Strategy


def _round_trip_cost_frac() -> float:
    return 2 * config.COST_PER_SIDE + config.COST_SELL_EXTRA


def run_portfolio(
    strategy: Strategy,
    data_map: dict[str, pd.DataFrame],
    bench_close: pd.Series | None = None,
    capital: float = None,
    risk_pct: float = None,
    max_positions: int = 6,
    max_hold: int = 60,
    start_date=None,
    allowed_symbols: set[str] | None = None,
    scores: dict[str, float] | None = None,
    symbol_sectors: dict[str, str] | None = None,
    sector_history: dict[str, "pd.Series"] | None = None,
    in_trend_quadrants: set[str] | None = None,
) -> dict:
    """Simulate a rupee portfolio.

    ``allowed_symbols`` (optional): restrict entries to these (the layered
    pipeline's survivors). ``scores``: symbol->score for ranking same-day signals.

    Point-in-time sector gate (optional): if ``sector_history`` (sector ->
    quadrant Series from ``sectors.quadrant_history``) and ``symbol_sectors``
    (symbol -> sector) are given, each candidate entry is only taken if the
    stock's sector was in ``in_trend_quadrants`` (default {Leading, Improving})
    AS OF the signal date. This removes the look-ahead of applying today's
    sector rotation to old history.

    ``start_date`` bounds the backtest window (e.g. last 3-5 years) so the
    fundamental gate — which uses only current yfinance data — isn't projected
    onto older history where it's meaningless.

    Returns dict: trades (DataFrame), equity (₹ Series), alloc_log (DataFrame),
    metrics (dict).
    """
    in_trend_quadrants = in_trend_quadrants or {"Leading", "Improving"}
    use_pit_sector = bool(sector_history and symbol_sectors)
    capital = capital if capital is not None else config.DEFAULT_CAPITAL
    risk_pct = risk_pct if risk_pct is not None else config.DEFAULT_RISK_PCT
    scores = scores or {}
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    if start_ts is not None and start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    cost_frac = _round_trip_cost_frac()

    # --- Pre-compute per-symbol signals and numpy arrays. -----------------
    sym_data = {}
    all_dates: set[pd.Timestamp] = set()
    for sym, df in data_map.items():
        if df is None or df.empty or len(df) < 60:
            continue
        if allowed_symbols is not None and sym not in allowed_symbols:
            continue
        sigs = strategy.generate_signals(df, bench_close=bench_close)
        if sigs.empty:
            continue
        idx = df.index
        pos_of_date = {ts: i for i, ts in enumerate(idx)}
        # Map each signal to its bar index; entry is next bar.
        sig_by_bar = {}
        for _, s in sigs.iterrows():
            si = pos_of_date.get(s["date"])
            if si is None or si + 1 >= len(idx):
                continue
            sig_by_bar[si] = s
        sym_data[sym] = {
            "idx": idx,
            "open": df["Open"].to_numpy(),
            "high": df["High"].to_numpy(),
            "low": df["Low"].to_numpy(),
            "close": df["Close"].to_numpy(),
            "pos_of_date": pos_of_date,
            "sig_by_bar": sig_by_bar,
        }
        all_dates.update(idx)

    if not sym_data:
        return {"trades": pd.DataFrame(), "equity": pd.Series(dtype=float),
                "alloc_log": pd.DataFrame(), "metrics": bt._empty_metrics()}

    timeline = sorted(all_dates)

    cash = capital
    open_positions = []          # list of dicts
    closed = []                  # completed trade dicts
    alloc_rows = []              # capital-allocation log
    equity_points = []           # (date, equity_₹)

    def portfolio_value(on_date):
        """Cash + marked-to-market value of open positions at on_date close.

        For each open position we reserved ``invested`` from cash; its current
        worth is that reserve adjusted by the open P&L (long gains when price
        rises, short gains when price falls)."""
        val = cash
        for p in open_positions:
            d = sym_data[p["symbol"]]
            i = d["pos_of_date"].get(on_date)
            px = d["close"][i] if i is not None else p["entry_price"]
            move = (px - p["entry_price"]) * p["shares"]
            open_pnl = move if p["side"] == "long" else -move
            val += p["invested"] + open_pnl
        return val

    for today in timeline:
        naive_today = today.tz_localize(None) if getattr(today, "tzinfo", None) else today
        if start_ts is not None and naive_today < start_ts:
            # Still mark equity so the curve starts flat.
            equity_points.append((today, cash))
            continue

        # --- 1) Manage/close open positions that exit on/after today. -----
        still_open = []
        for p in open_positions:
            if today >= p["exit_date"]:
                proceeds = p["shares"] * p["exit_price"]
                # Round-trip cost on the average of entry & exit notional.
                cost = cost_frac * (p["invested"] + proceeds) / 2
                if p["side"] == "long":
                    pnl_rs = (proceeds - p["invested"]) - cost
                else:  # short: profit when price falls
                    pnl_rs = (p["invested"] - proceeds) - cost
                # Release the reserved capital plus/minus the P&L back to cash.
                cash += p["invested"] + pnl_rs
                closed.append({
                    "symbol": p["symbol"], "side": p["side"],
                    "entry_date": p["entry_date"], "exit_date": p["exit_date"],
                    "days_held": int((p["exit_date"] - p["entry_date"]).days),
                    "shares": p["shares"], "entry_price": round(p["entry_price"], 2),
                    "stop_price": round(p["stop_price"], 2),
                    "target_price": round(p["target_price"], 2),
                    "exit_price": round(p["exit_price"], 2),
                    "invested_rs": round(p["invested"], 0),
                    "pnl_rs": round(pnl_rs, 0),
                    "pnl_pct": round(pnl_rs / p["invested"] * 100, 2) if p["invested"] else 0.0,
                    "exit_reason": p["exit_reason"], "score": round(p.get("score", 0), 1),
                })
            else:
                still_open.append(p)
        open_positions = still_open

        # --- 2) Collect today's fresh signals across symbols. -------------
        todays_signals = []
        for sym, d in sym_data.items():
            si = d["pos_of_date"].get(today)
            if si is None:
                continue
            s = d["sig_by_bar"].get(si)
            if s is None:
                continue
            if any(op["symbol"] == sym for op in open_positions):
                continue  # already holding this name
            # Point-in-time sector gate: was this stock's sector in-trend on the
            # signal date? (Skips the look-ahead of using today's rotation.)
            if use_pit_sector:
                sec = symbol_sectors.get(sym)
                quad = sectmod.quadrant_on(sector_history.get(sec), today) if sec else "Unknown"
                if quad not in in_trend_quadrants:
                    alloc_rows.append({
                        "date": today.date(), "symbol": sym,
                        "score": round(scores.get(sym, 0.0), 1),
                        "entry": None, "stop": None, "decision": "skipped",
                        "reason": f"sector {sec or '?'} was {quad} (not in-trend) on signal date",
                        "shares": 0, "invested_rs": 0, "cash_after": round(cash, 0)})
                    continue
            todays_signals.append((sym, s))

        if not todays_signals:
            equity_points.append((today, portfolio_value(today)))
            continue

        # Rank by score (desc).
        todays_signals.sort(key=lambda x: scores.get(x[0], 0.0), reverse=True)

        # --- 3) Open positions for top signals subject to slots & cash. ---
        for sym, s in todays_signals:
            slots_left = max_positions - len(open_positions)
            taken = False
            reason = ""
            entry_i = d_entry = None
            d = sym_data[sym]
            si = d["pos_of_date"][today]
            entry_i = si + 1
            entry_price = d["open"][entry_i]
            stop = s["stop"]
            per_share_risk = abs(entry_price - stop)

            if slots_left <= 0:
                reason = "no slot (max positions full)"
            elif per_share_risk <= 0:
                reason = "invalid stop geometry"
            else:
                # Risk-based size, capped by (a) an equal share of available
                # cash and (b) a hard ceiling of MAX_POSITION_PCT of total
                # capital in any single stock (universal concentration rule).
                rupee_risk = capital * risk_pct
                shares = int(rupee_risk // per_share_risk)
                notional = shares * entry_price
                max_notional = capital * config.MAX_POSITION_PCT   # 10% hard cap
                cash_cap = cash / max(slots_left, 1)
                spend_cap = min(cash_cap, max_notional)
                if notional > spend_cap:
                    shares = int(spend_cap // entry_price)
                    notional = shares * entry_price
                if shares <= 0:
                    reason = "insufficient cash / below 10% cap for 1 share"
                else:
                    # Resolve the exit up front (deterministic on historical data).
                    ei, epx, realized, ereason = bt.resolve_exit(
                        s["side"], entry_price, stop,
                        [s["t1"], s["t2"], s["t3"]], [s["w1"], s["w2"], s["w3"]],
                        d["high"], d["low"], d["close"], entry_i, int(max_hold), len(d["idx"]),
                        trail_amount=getattr(strategy, "trail_amount", None),
                        trail_is_pct=getattr(strategy, "trail_is_pct", False))
                    cash -= notional
                    open_positions.append({
                        "symbol": sym, "side": s["side"],
                        "entry_date": d["idx"][entry_i], "entry_price": entry_price,
                        "stop_price": stop, "target_price": s["t1"],
                        "shares": shares, "invested": notional, "collateral": notional,
                        "exit_date": d["idx"][ei], "exit_price": epx,
                        "exit_reason": ereason, "score": scores.get(sym, 0.0),
                    })
                    taken = True
                    reason = "opened"

            alloc_rows.append({
                "date": today.date(), "symbol": sym, "score": round(scores.get(sym, 0.0), 1),
                "entry": round(entry_price, 2), "stop": round(stop, 2),
                "decision": "TAKEN" if taken else "skipped", "reason": reason,
                "shares": shares if taken else 0,
                "invested_rs": round(notional, 0) if taken else 0,
                "cash_after": round(cash, 0),
            })

        equity_points.append((today, portfolio_value(today)))

    # --- Build outputs. ---------------------------------------------------
    tdf = pd.DataFrame(closed)
    if not tdf.empty:
        tdf = tdf.sort_values("exit_date").reset_index(drop=True)
    equity = pd.Series({d: v for d, v in equity_points}).sort_index()
    alloc = pd.DataFrame(alloc_rows)
    # Metrics use the equity trimmed to the backtest window. Before start_date
    # the curve is intentionally flat (no trades yet); leaving that long pre-
    # window tail in would divide CAGR over the wrong (far longer) horizon and
    # understate it badly. Trim to the window so CAGR/Sharpe reflect the period
    # actually traded.
    eq_metrics = equity
    if start_ts is not None and not equity.empty:
        idx_naive = equity.index.tz_localize(None) if getattr(equity.index, "tz", None) is not None else equity.index
        mask = idx_naive >= start_ts
        if mask.any():
            eq_metrics = equity[mask]
    metrics = _portfolio_metrics(tdf, eq_metrics, capital)
    return {"trades": tdf, "equity": equity, "alloc_log": alloc, "metrics": metrics}


def _portfolio_metrics(tdf: pd.DataFrame, equity: pd.Series, capital: float) -> dict:
    m = bt._empty_metrics()
    if tdf.empty or len(equity) < 2:
        return m
    n = len(tdf)
    wins = tdf[tdf["pnl_rs"] > 0]
    losses = tdf[tdf["pnl_rs"] <= 0]
    gross_win = wins["pnl_rs"].sum()
    gross_loss = abs(losses["pnl_rs"].sum())
    m["trades"] = n
    m["win_rate"] = len(wins) / n
    m["avg_win"] = wins["pnl_pct"].mean() / 100 if len(wins) else 0.0
    m["avg_loss"] = losses["pnl_pct"].mean() / 100 if len(losses) else 0.0
    m["best_trade"] = tdf["pnl_pct"].max() / 100
    m["worst_trade"] = tdf["pnl_pct"].min() / 100
    m["profit_factor"] = (gross_win / gross_loss) if gross_loss > 1e-9 else np.inf
    m["expectancy"] = tdf["pnl_rs"].mean() / capital
    m["avg_bars_held"] = tdf["days_held"].mean()

    start_v, end_v = equity.iloc[0], equity.iloc[-1]
    m["total_return"] = end_v / start_v - 1.0
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    m["cagr"] = (end_v / start_v) ** (1 / years) - 1.0 if years > 0 and start_v > 0 else 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    m["max_dd"] = dd.min()
    eq_ret = equity.pct_change().dropna()
    m["sharpe"] = (eq_ret.mean() / eq_ret.std() * np.sqrt(252)) if eq_ret.std() > 1e-9 else 0.0
    downside = eq_ret[eq_ret < 0]
    m["sortino"] = (eq_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 1e-9 else 0.0
    m["total_pnl_rs"] = tdf["pnl_rs"].sum()
    m["final_value_rs"] = end_v
    return m
