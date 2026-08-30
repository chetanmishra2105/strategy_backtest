"""Momentum Lab — grid-search harness for the cross-sectional momentum engine.

The momentum analogue of ``experiments.py`` (the Supertrend sweep). It sweeps
combinations of universe × momentum-lookback × hold-count (top-N) × rebalance ×
layer-config, running EACH through the SAME walk-forward harness the Momentum Lab
uses (``walkforward.walk_forward_momentum``), so every number is **out-of-sample**
— what a live trader would actually have earned, not a curve fit to the past.

It deliberately does NOT touch ``experiments.py`` or the Supertrend strategy — it
is a separate engine with a parallel API (``DEFAULT_AXES``, ``count_combos``,
``build_grid``, ``run_grid``, ``rank``) so the Experiment Lab UI can drive either
one with the same controls.

Each combo produces one metrics row; combos are ranked by risk-adjusted Calmar
(OOS CAGR ÷ |OOS max drawdown|) with a minimum-trades floor, exactly like the
Supertrend grid.
"""

from __future__ import annotations

import itertools

import pandas as pd

from . import config
from . import data as datamod
from . import walkforward as wf

STRATEGY = "Cross-sectional Momentum"

# --------------------------------------------------------------------------
# Axes — the momentum knobs a systematic fund tunes. Mirrors the shape of the
# Supertrend grid (5 axes) so the UI treats them identically.
# --------------------------------------------------------------------------
LOOKBACK_MONTHS = [3, 6, 9, 12]     # momentum formation window
TOP_N_CHOICES = [8, 12, 20]         # how many names to hold
REBALANCE_MODES = {                 # label -> pandas offset alias
    "Monthly": "ME",
    "Weekly": "W-FRI",
    "Quarterly": "QE",
}

# The layer config — the momentum analogue of the Supertrend "gates" axis. Each
# maps to (use_regime, vol_scaled), matching the Momentum Lab's ablation ladder.
LAYER_CONFIGS: dict[str, tuple[bool, bool]] = {
    "Momentum only": (False, False),
    "+ HMM regime filter": (True, False),
    "+ HMM + vol-scaled sizing": (True, True),
}

UNIVERSE_NAMES = list(config.UNIVERSES.keys())

# Fixed, sane simulation defaults (not swept — kept constant so combos compare).
# These mirror the Momentum Lab's defaults.
FIXED = dict(
    skip=21,                # classic 12-1 momentum (skip most-recent month)
    use_trend_filter=True,  # only hold names above the 200-DMA
    regime_states=3,        # Bear / Neutral / Bull HMM
    train_months=9,         # walk-forward fold: 9mo train / 3mo test
    test_months=3,
    n_years=5,              # history window
)

# Balanced default axis selection. All overridable in the UI.
DEFAULT_AXES = dict(
    universes=["NIFTY Midcap 50", "NIFTY Smallcap 50", "Midcap/Smallcap 150"],
    lookbacks=[6, 12],
    top_ns=[8, 12],
    rebalances=["Monthly", "Quarterly"],
    layers=list(LAYER_CONFIGS.keys()),
)


def count_combos(axes: dict) -> int:
    return (len(axes["universes"]) * len(axes["lookbacks"]) * len(axes["top_ns"])
            * len(axes["rebalances"]) * len(axes["layers"]))


def build_grid(axes: dict) -> list[dict]:
    """Cartesian product of the selected axes -> list of combo dicts."""
    combos = []
    for uni, lb, tn, reb, layer in itertools.product(
            axes["universes"], axes["lookbacks"], axes["top_ns"],
            axes["rebalances"], axes["layers"]):
        combos.append({
            "universe": uni, "lookback_m": lb, "top_n": tn,
            "rebalance": reb, "layer": layer,
        })
    return combos


# --------------------------------------------------------------------------
# Caches so we don't refetch data / refit the HMM per combo.
# --------------------------------------------------------------------------
def _load_universe(uni_name: str, cache: dict) -> tuple[dict, pd.Series | None]:
    """Load (and cache) the {sym: OHLCV} data_map + benchmark close for a
    universe. Uses ``datamod.get_ohlcv`` directly (not the streamlit-cached UI
    helper) so this engine has no Streamlit dependency."""
    if uni_name in cache:
        return cache[uni_name]
    universe = config.UNIVERSES[uni_name]
    dm = {}
    for s in universe:
        df = datamod.get_ohlcv(s, "1d")
        if df is not None and not df.empty:
            dm[s] = df
    b = datamod.get_ohlcv(config.BENCHMARK, "1d")
    bench = b["Close"] if b is not None and not b.empty else None
    cache[uni_name] = (dm, bench)
    return dm, bench


def _run_one(combo: dict, data_cache: dict) -> dict:
    """Run a single momentum combination through the walk-forward harness."""
    uni_name = combo["universe"]
    dm, bench = _load_universe(uni_name, data_cache)
    use_regime, vol_scaled = LAYER_CONFIGS[combo["layer"]]

    lookback = int(combo["lookback_m"] * 21)
    mom_params = dict(
        top_n=combo["top_n"], lookback=lookback, skip=FIXED["skip"],
        rebalance=REBALANCE_MODES[combo["rebalance"]],
        use_trend_filter=FIXED["use_trend_filter"],
        vol_scaled=vol_scaled)

    res = wf.walk_forward_momentum(
        dm, bench_close=bench, capital=config.DEFAULT_CAPITAL,
        train_months=FIXED["train_months"], test_months=FIXED["test_months"],
        n_years=FIXED["n_years"], momentum_params=mom_params,
        use_regime=use_regime, regime_states=FIXED["regime_states"])

    m = res["oos_metrics"]
    cagr = m["cagr"]
    dd = abs(m["max_dd"])
    calmar = (cagr / dd) if dd > 1e-9 else (cagr / 0.01 if cagr else 0.0)
    return {
        "universe": uni_name,
        "lookback_m": combo["lookback_m"],
        "top_n": combo["top_n"],
        "rebalance": combo["rebalance"],
        "layer": combo["layer"],
        "trades": int(m["trades"]),
        "win_%": round(m.get("win_rate", 0.0) * 100, 1),
        "total_%": round(m["total_return"] * 100, 1),
        "CAGR_%": round(cagr * 100, 1),
        "max_dd_%": round(m["max_dd"] * 100, 1),
        "sharpe": round(m.get("sharpe", 0.0), 2),
        "calmar": round(calmar, 2),
    }


def run_grid(grid: list[dict], progress_cb=None) -> pd.DataFrame:
    """Run every combo through walk-forward; return a long-form DataFrame.

    ``progress_cb(done, total)`` is called after each combo for a UI progress bar.
    """
    data_cache: dict = {}
    rows = []
    total = len(grid)
    for i, combo in enumerate(grid, 1):
        try:
            rows.append(_run_one(combo, data_cache))
        except Exception as e:  # keep the grid going; record the failure
            rows.append({
                "universe": combo["universe"], "lookback_m": combo["lookback_m"],
                "top_n": combo["top_n"], "rebalance": combo["rebalance"],
                "layer": combo["layer"], "trades": 0, "win_%": 0.0,
                "total_%": 0.0, "CAGR_%": 0.0, "max_dd_%": 0.0, "sharpe": 0.0,
                "calmar": 0.0, "error": str(e)[:80]})
        if progress_cb:
            progress_cb(i, total)
    return pd.DataFrame(rows)


def rank(df: pd.DataFrame, min_trades: int = 20) -> pd.DataFrame:
    """Rank by risk-adjusted Calmar (OOS CAGR ÷ |max_dd|), floor on trade count.

    Combos with fewer than ``min_trades`` are kept but flagged ``thin=True`` and
    sorted below all non-thin combos so a lucky handful of trades can't win.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["thin"] = out["trades"] < min_trades
    out = out.sort_values(
        by=["thin", "calmar", "CAGR_%"], ascending=[True, False, False]
    ).reset_index(drop=True)
    return out
