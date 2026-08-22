"""Walk-forward validation harness (NEW module).

The point of this file: a backtest number you can *trust*. A strategy tuned on
the same history it's measured on is curve-fit — it will look great in the past
and fail live. Walk-forward removes that bias by only ever measuring the
strategy on data it did NOT see while being chosen.

Two things live here:

1. ``rolling_folds`` — split the timeline into rolling (train, test) windows,
   e.g. 9 months train / 3 months test, stepping forward 3 months each time.

2. ``walk_forward_momentum`` — for the cross-sectional momentum engine, run each
   fold's TEST window and stitch the test-only segments into one continuous
   out-of-sample (OOS) equity curve. Because the momentum strategy is rule-based
   (no per-fold parameter fitting by default), "train" here means the formation
   window / regime model is only ever built from data up to the test window's
   start — the OOS curve is what a live trader would actually have earned.

   If ``param_grid`` is supplied, each fold ALSO picks the best parameter combo
   on its train window and applies it to the test window — this is true
   walk-forward optimisation, and the OOS curve then reflects the honest cost of
   choosing parameters from the past.

The OOS equity curve + its metrics are the numbers to believe. Everything is
compared against the in-sample (full-history) run so the optimism gap is visible.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config, momentum, regime as rg


# --------------------------------------------------------------------------
# Fold construction
# --------------------------------------------------------------------------
def rolling_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int = 9,
    test_months: int = 3,
    step_months: int | None = None,
) -> list[dict]:
    """Return a list of {train_start, train_end, test_start, test_end} folds.

    Default 9/3 rolling, stepping forward by ``test_months`` (so test windows
    tile the timeline with no gaps and no overlap). The regime/formation models
    for a fold use only data up to ``test_start``.
    """
    step_months = step_months or test_months
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    folds = []
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start >= end:
            break
        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": min(test_end, end),
        })
        if test_end >= end:
            break
        train_start = train_start + pd.DateOffset(months=step_months)
    return folds


# --------------------------------------------------------------------------
# Walk-forward for the momentum engine
# --------------------------------------------------------------------------
def walk_forward_momentum(
    data_map: dict[str, pd.DataFrame],
    bench_close: pd.Series | None = None,
    capital: float = None,
    train_months: int = 9,
    test_months: int = 3,
    momentum_params: dict | None = None,
    param_grid: dict | None = None,
    use_regime: bool = False,
    regime_states: int = 3,
    regime_labels: pd.Series | None = None,
    n_years: float = 5.0,
    end_date: pd.Timestamp | None = None,
) -> dict:
    """Run rolling-fold out-of-sample validation of the momentum strategy.

    Returns dict:
      oos_equity    : ₹ Series stitched from every fold's TEST window (the
                      number to trust).
      oos_metrics   : metrics of that OOS curve.
      is_metrics    : in-sample (full-window) metrics for the optimism gap.
      folds         : per-fold table (train/test dates, test CAGR, chosen params).
      per_fold      : list of raw fold result dicts.

    ``momentum_params`` : fixed kwargs for momentum.run_momentum (used when no
                          grid). ``param_grid`` : {param: [values]} to optimise
                          per fold on the train window (max Sharpe), then apply
                          to test. ``use_regime`` gates the book with a
                          point-in-time HMM on ``bench_close``.
    """
    capital = capital if capital is not None else config.DEFAULT_CAPITAL
    momentum_params = dict(momentum_params or {})
    end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=n_years)

    # Point-in-time regime labels once over the whole span (each label uses only
    # past prices, so it's valid to slice per fold). Reuse a precomputed series
    # when the caller supplies one (the ablation ladder computes it once and
    # shares it across all three configs — a big speed-up).
    reg = None
    if use_regime:
        if regime_labels is not None:
            reg = regime_labels
        elif bench_close is not None and len(bench_close):
            reg = rg.regime_series_pit(bench_close, n_states=regime_states)

    folds = rolling_folds(start_date, end_date, train_months, test_months)
    if not folds:
        return {"oos_equity": pd.Series(dtype=float), "oos_metrics": momentum._empty(),
                "is_metrics": momentum._empty(), "folds": pd.DataFrame(), "per_fold": []}

    oos_segments = []
    fold_rows = []
    per_fold = []
    oos_trades = []     # combined out-of-sample trade ledger (per test fold)
    running = capital   # OOS equity compounds across folds

    for i, fd in enumerate(folds, 1):
        # Pick params for this fold.
        if param_grid:
            best = _optimise_on_train(data_map, bench_close, reg, fd,
                                      momentum_params, param_grid, capital)
            params = {**momentum_params, **best}
        else:
            params = momentum_params
            best = {}

        # Run the TEST window, compounding from the running OOS equity.
        res = momentum.run_momentum(
            data_map, capital=running,
            start_date=fd["test_start"], end_date=fd["test_end"],
            regime_series=reg if use_regime else None, **params)
        eq = res["equity"]
        if eq is None or eq.empty:
            fold_rows.append({"fold": i, "train": _rng(fd["train_start"], fd["train_end"]),
                              "test": _rng(fd["test_start"], fd["test_end"]),
                              "test_return_%": 0.0, "params": _pstr(best)})
            continue
        # Compound: chain this fold onto the previous OOS end value.
        seg = eq / float(eq.iloc[0]) * running
        running = float(seg.iloc[-1])
        oos_segments.append(seg)
        fret = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        fold_rows.append({
            "fold": i, "train": _rng(fd["train_start"], fd["train_end"]),
            "test": _rng(fd["test_start"], fd["test_end"]),
            "test_return_%": round(fret * 100, 2),
            "params": _pstr(best)})
        per_fold.append(res)
        # Collect this fold's trades into the OOS ledger, tagged with the fold #.
        ft = res.get("trades")
        if ft is not None and not ft.empty:
            ft = ft.copy()
            ft.insert(0, "fold", i)
            oos_trades.append(ft)

    # Stitch OOS segments (drop duplicate boundary dates).
    if oos_segments:
        oos_equity = pd.concat(oos_segments)
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index()
    else:
        oos_equity = pd.Series(dtype=float)
    oos_metrics = momentum._metrics_from_equity(oos_equity, capital)
    # Combined OOS trade ledger + trade-level metrics (win rate / PF / avg hold).
    oos_trades_df = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    momentum._augment_trade_metrics(oos_metrics, oos_trades_df)

    # In-sample reference: one full-window run with the base params.
    is_res = momentum.run_momentum(
        data_map, capital=capital, start_date=start_date, end_date=end_date,
        regime_series=reg if use_regime else None, **momentum_params)
    is_metrics = is_res["metrics"]

    return {"oos_equity": oos_equity, "oos_metrics": oos_metrics,
            "oos_trades": oos_trades_df,
            "is_metrics": is_metrics, "folds": pd.DataFrame(fold_rows),
            "per_fold": per_fold}


def _optimise_on_train(data_map, bench_close, reg, fd, base_params,
                       param_grid, capital) -> dict:
    """Grid-search ``param_grid`` on the fold's TRAIN window; return the combo
    with the best train Sharpe. This is what makes it walk-forward *optimisation*
    rather than just validation."""
    keys = list(param_grid.keys())
    best_combo, best_sharpe = {}, -np.inf
    for values in itertools.product(*[param_grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        res = momentum.run_momentum(
            data_map, capital=capital,
            start_date=fd["train_start"], end_date=fd["train_end"],
            regime_series=reg, **{**base_params, **combo})
        m = res["metrics"]
        s = m.get("sharpe", 0.0)
        if np.isfinite(s) and s > best_sharpe:
            best_sharpe, best_combo = s, combo
    return best_combo


# --------------------------------------------------------------------------
# Ablation ladder — the headline table the user asked for
# --------------------------------------------------------------------------
def ablation_ladder(
    data_map: dict[str, pd.DataFrame],
    bench_close: pd.Series | None,
    capital: float = None,
    train_months: int = 9,
    test_months: int = 3,
    n_years: float = 5.0,
    momentum_params: dict | None = None,
    regime_states: int = 3,
    end_date: pd.Timestamp | None = None,
) -> dict:
    """Build the step-by-step ladder, every row measured OUT-OF-SAMPLE:

        Step 1  Momentum only
        Step 2  Momentum + HMM regime filter
        Step 3  Momentum + HMM + vol-scaled sizing

    Returns {table: DataFrame, curves: {label: oos_equity Series}}. The caller
    adds the buy&hold index baseline (Step 0) from performance.strategy_vs_indices.
    """
    capital = capital if capital is not None else config.DEFAULT_CAPITAL
    base = dict(momentum_params or {})

    # Compute the point-in-time regime labels ONCE and share across configs 2 & 3
    # (fitting the HMM is the expensive step; there's no reason to redo it).
    reg = None
    if bench_close is not None and len(bench_close):
        reg = rg.regime_series_pit(bench_close, n_states=regime_states)

    configs = [
        ("1 · Momentum only", dict(use_regime=False), {"vol_scaled": False}),
        ("2 · + HMM regime filter", dict(use_regime=True), {"vol_scaled": False}),
        ("3 · + HMM + vol-scaled sizing", dict(use_regime=True), {"vol_scaled": True}),
    ]
    rows, curves, ledgers = [], {}, {}
    for label, wf_kw, mom_kw in configs:
        res = walk_forward_momentum(
            data_map, bench_close=bench_close, capital=capital,
            train_months=train_months, test_months=test_months, n_years=n_years,
            momentum_params={**base, **mom_kw}, regime_states=regime_states,
            regime_labels=reg, end_date=end_date, **wf_kw)
        m = res["oos_metrics"]
        curves[label] = res["oos_equity"]
        ledgers[label] = res.get("oos_trades", pd.DataFrame())
        rows.append({
            "Configuration": label,
            "OOS CAGR %": round(m["cagr"] * 100, 1),
            "OOS total %": round(m["total_return"] * 100, 1),
            "Max drawdown %": round(m["max_dd"] * 100, 1),
            "Sharpe": round(m["sharpe"], 2),
            "Calmar": round(m["cagr"] / abs(m["max_dd"]), 2) if m["max_dd"] < -1e-9 else np.nan,
            "Win rate %": round(m["win_rate"] * 100, 1),
            "Trades": int(m["trades"]),
            "Final ₹": round(m["final_value_rs"], 0),
        })
    return {"table": pd.DataFrame(rows), "curves": curves, "ledgers": ledgers,
            "regime_labels": reg}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _rng(a, b) -> str:
    return f"{pd.Timestamp(a).date()} → {pd.Timestamp(b).date()}"


def _pstr(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else "(fixed)"
