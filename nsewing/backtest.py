"""Event-based backtester with multi-target scaling exits.

Design choices for realism:
  * No look-ahead: a signal detected on bar i is executed at bar i+1's OPEN.
  * Costs: entry & exit both pay COST_PER_SIDE; exits additionally pay STT.
  * Scaling exits: partial size taken off at T1/T2/T3 (weights from the doc);
    after T1 the stop trails to break-even. A hard stop and a max-hold cap
    bound every trade to the 7-30 day swing horizon.
  * One open position per symbol (a new signal while in a trade is ignored).

Works for both long and short trades. Returns a Trade log and an equity
curve, plus a metrics dict mirroring the document's performance report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .strategies import Strategy


@dataclass
class Trade:
    symbol: str
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    bars_held: int
    ret_pct: float          # net return on the trade (after costs), signed for P&L
    exit_reason: str


def _apply_costs(gross_ret: float, side: str) -> float:
    """Convert a gross price move into a net return after round-trip costs."""
    # Round-trip cost as a fraction of notional (entry side + exit side + STT).
    cost = 2 * config.COST_PER_SIDE + config.COST_SELL_EXTRA
    return gross_ret - cost


def resolve_exit(side, entry, stop, targets, weights,
                 highs, lows, closes, entry_i, max_hold, n,
                 trail_amount=None, trail_is_pct=False):
    """Resolve a single position's exit (shared by single-symbol and portfolio
    backtests). Returns (exit_i, exit_price, realized_gross, exit_reason).

    Rules: targets fill intraday (limit-style); the stop fires only on a CLOSE
    beyond it and never on the entry bar (this is the fix for the same-day-exit
    bug). Scaling out at T1/T2/T3 with break-even trail after T1; residual
    closed at the final bar (max-hold).

    Trailing stop (optional): when ``trail_amount`` is set (a rupee amount, or a
    fraction if ``trail_is_pct``), the stop follows price once the trade is in
    profit by at least the trail distance. It only ever ratchets in the favourable
    direction (up for long, down for short), never loosens, and still fires on a
    daily CLOSE beyond it. No look-ahead: the trail uses only the peak/trough of
    bars seen so far.
    """
    remaining = 1.0
    realized = 0.0
    hit = [False, False, False]
    cur_stop = stop
    exit_i = None
    exit_reason = "max_hold"
    exit_price = None
    peak = entry      # highest price seen so far (long)
    trough = entry    # lowest price seen so far (short)
    use_trail = trail_amount is not None and trail_amount > 0

    def _trail_dist(ref):
        return ref * trail_amount if trail_is_pct else trail_amount

    last_bar = min(entry_i + max_hold, n - 1)
    for j in range(entry_i, last_bar + 1):
        hi, lo, cl = highs[j], lows[j], closes[j]
        # Neither stop NOR target may fill on the entry bar: we buy at this bar's
        # open, so acting on the same bar's high/low would be look-ahead and
        # produces bogus "0-day" trades. Both become active the next bar.
        allow_exit = j > entry_i

        if side == "long":
            for k in range(3):
                if allow_exit and not hit[k] and hi >= targets[k]:
                    realized += weights[k] * (targets[k] / entry - 1.0)
                    remaining -= weights[k]
                    hit[k] = True
                    if k == 0:
                        cur_stop = entry
            if remaining <= 1e-9:
                exit_i, exit_price, exit_reason = j, targets[2], "target"
                break
            # Trailing stop: once in profit by the trail distance, ratchet up.
            if use_trail:
                peak = max(peak, hi)
                dist = _trail_dist(peak)
                if peak >= entry + dist:                 # activate once green
                    cur_stop = max(cur_stop, peak - dist)
            if allow_exit and cl <= cur_stop:
                realized += remaining * (cl / entry - 1.0)
                exit_i, exit_price = j, cl
                exit_reason = "stop" if (not any(hit) and not use_trail) else "trail_stop"
                remaining = 0.0
                break
        else:  # short
            for k in range(3):
                if allow_exit and not hit[k] and lo <= targets[k]:
                    realized += weights[k] * (1.0 - targets[k] / entry)
                    remaining -= weights[k]
                    hit[k] = True
                    if k == 0:
                        cur_stop = entry
            if remaining <= 1e-9:
                exit_i, exit_price, exit_reason = j, targets[2], "target"
                break
            if use_trail:
                trough = min(trough, lo)
                dist = _trail_dist(trough)
                if trough <= entry - dist:               # activate once green (short)
                    cur_stop = min(cur_stop, trough + dist)
            if allow_exit and cl >= cur_stop:
                realized += remaining * (1.0 - cl / entry)
                exit_i, exit_price = j, cl
                exit_reason = "stop" if (not any(hit) and not use_trail) else "trail_stop"
                remaining = 0.0
                break

    # Residual closed at the final bar's close (max-hold / partial).
    if remaining > 1e-9:
        j = exit_i if exit_i is not None else last_bar
        close_px = closes[j]
        if side == "long":
            realized += remaining * (close_px / entry - 1.0)
        else:
            realized += remaining * (1.0 - close_px / entry)
        if exit_i is None:
            exit_i, exit_price = last_bar, close_px

    return exit_i, exit_price, realized, exit_reason


def _simulate_one(
    df: pd.DataFrame, sigs: pd.DataFrame, symbol: str,
    max_hold_override: int | None = None,
    trail_amount: float | None = None, trail_is_pct: bool = False,
) -> list[Trade]:
    """Walk bars forward; open at next bar's open; manage scaling exits."""
    trades: list[Trade] = []
    if sigs.empty:
        return trades

    idx = df.index
    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    pos_of_date = {ts: i for i, ts in enumerate(idx)}

    in_trade_until = -1  # bar index before which we cannot open a new trade

    for _, s in sigs.iterrows():
        sig_i = pos_of_date.get(s["date"])
        if sig_i is None or sig_i + 1 >= len(idx):
            continue
        entry_i = sig_i + 1
        if entry_i <= in_trade_until:
            continue  # already in a trade

        side = s["side"]
        entry = opens[entry_i]                    # execute at next bar's open
        stop = s["stop"]
        targets = [s["t1"], s["t2"], s["t3"]]
        weights = [s["w1"], s["w2"], s["w3"]]
        max_hold = int(max_hold_override) if max_hold_override else int(s["max_hold"])

        # Validate geometry.
        if side == "long" and not (stop < entry):
            continue
        if side == "short" and not (stop > entry):
            continue

        exit_i, exit_price, realized, exit_reason = resolve_exit(
            side, entry, stop, targets, weights,
            highs, lows, closes, entry_i, max_hold, len(idx),
            trail_amount=trail_amount, trail_is_pct=trail_is_pct)

        net = _apply_costs(realized, side)
        trades.append(Trade(
            symbol=symbol, side=side,
            entry_date=idx[entry_i], exit_date=idx[exit_i],
            entry_price=float(entry), exit_price=float(exit_price),
            bars_held=int(exit_i - entry_i),
            ret_pct=float(net), exit_reason=exit_reason,
        ))
        in_trade_until = exit_i

    return trades


def run_backtest(
    strategy: Strategy,
    data_map: dict[str, pd.DataFrame],
    bench_close: pd.Series | None = None,
    risk_pct: float = None,
    start_date=None,
    max_hold_override: int | None = None,
) -> dict:
    """Backtest a strategy across one or many symbols.

    Returns dict with keys: trades (DataFrame), equity (Series), metrics (dict),
    doc_stats (dict). Position sizing risks a fixed fraction of a notional
    equity per trade so the equity curve reflects R-multiples, not raw price %.

    ``start_date`` (optional) restricts *entries* to on/after that date, while
    indicators are still computed on the full history — used by the trailing-
    window performance page (last 30/60/90/... days).
    """
    risk_pct = risk_pct if risk_pct is not None else config.DEFAULT_RISK_PCT
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    all_trades: list[Trade] = []
    for sym, df in data_map.items():
        if df is None or df.empty or len(df) < 60:
            continue
        sigs = strategy.generate_signals(df, bench_close=bench_close)
        if start_ts is not None and not sigs.empty:
            # Normalise both sides to tz-naive dates for a safe comparison.
            sig_dates = pd.to_datetime(sigs["date"])
            if getattr(sig_dates.dt, "tz", None) is not None:
                sig_dates = sig_dates.dt.tz_localize(None)
            cutoff = start_ts.tz_localize(None) if start_ts.tzinfo is not None else start_ts
            sigs = sigs[sig_dates.to_numpy() >= cutoff.to_datetime64()]
        all_trades.extend(_simulate_one(
            df, sigs, sym, max_hold_override=max_hold_override,
            trail_amount=getattr(strategy, "trail_amount", None),
            trail_is_pct=getattr(strategy, "trail_is_pct", False)))

    if not all_trades:
        return {"trades": pd.DataFrame(), "equity": pd.Series(dtype=float),
                "metrics": _empty_metrics(), "doc_stats": strategy.doc_stats}

    tdf = pd.DataFrame([t.__dict__ for t in all_trades]).sort_values("exit_date")
    tdf = tdf.reset_index(drop=True)

    equity = _build_equity(tdf, risk_pct)
    metrics = _compute_metrics(tdf, equity, risk_pct)
    return {"trades": tdf, "equity": equity, "metrics": metrics,
            "doc_stats": strategy.doc_stats}


def _build_equity(tdf: pd.DataFrame, risk_pct: float) -> pd.Series:
    """Compound equity by treating each trade's ret_pct as applied to a fixed
    fraction of current equity (simple, position-sized compounding)."""
    equity = 1.0
    pts = []
    for _, t in tdf.iterrows():
        equity *= (1.0 + risk_pct * 10 * t["ret_pct"])  # 10x leverage-of-risk proxy
        pts.append((t["exit_date"], equity))
    ser = pd.Series({d: v for d, v in pts})
    ser.index = pd.to_datetime(ser.index)
    return ser.sort_index()


def _compute_metrics(tdf: pd.DataFrame, equity: pd.Series, risk_pct: float) -> dict:
    rets = tdf["ret_pct"]
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    n = len(rets)
    win_rate = len(wins) / n if n else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 1e-9 else np.inf
    expectancy = rets.mean() if n else 0.0

    # Equity-based stats.
    if len(equity) > 1:
        total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
        days = max((equity.index[-1] - equity.index[0]).days, 1)
        years = days / 365.25
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if years > 0 else 0.0
        roll_max = equity.cummax()
        drawdown = equity / roll_max - 1.0
        max_dd = drawdown.min()
        eq_ret = equity.pct_change().dropna()
        sharpe = (eq_ret.mean() / eq_ret.std() * np.sqrt(252)) if eq_ret.std() > 1e-9 else 0.0
        downside = eq_ret[eq_ret < 0]
        sortino = (eq_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 1e-9 else 0.0
    else:
        total_return = cagr = max_dd = sharpe = sortino = 0.0

    return {
        "trades": int(n),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "best_trade": float(rets.max()) if n else 0.0,
        "worst_trade": float(rets.min()) if n else 0.0,
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_dd": float(max_dd),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "avg_bars_held": float(tdf["bars_held"].mean()),
    }


def _empty_metrics() -> dict:
    keys = ["trades", "win_rate", "avg_win", "avg_loss", "best_trade",
            "worst_trade", "profit_factor", "expectancy", "total_return",
            "cagr", "max_dd", "sharpe", "sortino", "avg_bars_held"]
    return {k: 0.0 for k in keys}
