"""Strategy library — the three strategies from
``Swing_Trading_Strategies_Indian_Market.docx`` as pluggable classes.

Each strategy exposes ``generate_signals(df)`` returning a DataFrame of
trade *setups* (one row per bar where an entry triggers), with columns:

    side        : 'long' or 'short'
    entry       : intended entry price (executed next bar in the backtester)
    stop        : hard stop-loss price
    t1, t2, t3  : the three scaling targets
    w1, w2, w3  : position fraction taken off at each target
    max_hold    : max bars to hold (swing horizon cap)

The backtester consumes these and simulates execution with costs/slippage and
no look-ahead. Rules are transcribed directly from the document's §6 scanner
specs and §2/3/4 entry-exit tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from . import signals as sig


@dataclass
class Strategy:
    """Base class. Subclasses implement ``_entries`` and ``_levels``.

    ``params`` holds tunable thresholds. Each subclass sets doc-faithful
    defaults; callers may pass ``sensitivity='relaxed'`` (or override individual
    params) to loosen the conjunction so daily yfinance data yields a tradeable
    number of signals. See ``notes`` for why this matters.
    """

    name: str = "base"
    side: str = "long"
    max_hold: int = 10
    rr_min: float = 2.0
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25)
    params: dict = field(default_factory=dict)
    # Optional user-chosen reward:risk. When set (e.g. 1/2/3), the strategy's
    # own targets are replaced by a single target at entry ± rr_target×risk, so
    # you can test "how does a fixed 1:R exit perform?" from the UI.
    rr_target: float | None = None
    # Optional fixed percentage stop. When set (e.g. 0.05 = 5%), the stop is
    # placed pct_stop away from entry and the target at pct_stop×target_mult
    # away — a wider, percentage-based bracket that lets swing trades breathe
    # instead of being shaken out by tight ATR stops. Takes precedence over
    # rr_target and the strategy's own levels.
    pct_stop: float | None = None
    pct_target_mult: float = 2.0
    # Optional trailing stop: follows price once the trade is in profit. A rupee
    # amount, or a fraction of price if trail_is_pct. Ratchets only (never loosens).
    trail_amount: float | None = None
    trail_is_pct: bool = False
    # doc-published reference stats for side-by-side comparison
    doc_stats: dict = field(default_factory=dict)

    def prepare(self, df: pd.DataFrame, bench_close: pd.Series | None = None) -> pd.DataFrame:
        return ind.enrich(df, bench_close=bench_close)

    def _entries(self, d: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _levels(self, d: pd.DataFrame) -> dict[str, pd.Series]:
        raise NotImplementedError

    def generate_signals(
        self, df: pd.DataFrame, bench_close: pd.Series | None = None
    ) -> pd.DataFrame:
        d = self.prepare(df, bench_close=bench_close)
        entries = self._entries(d).fillna(False)
        levels = self._levels(d)

        # Level overrides (priority: pct_stop > rr_target > strategy's own).
        entry_s = levels["entry"]
        if self.pct_stop:
            # Fixed % stop from entry; target at pct_stop × target_mult.
            sl = self.pct_stop
            tp = self.pct_stop * self.pct_target_mult
            if self.side == "long":
                stop_s = entry_s * (1 - sl)
                tgt = entry_s * (1 + tp)
            else:
                stop_s = entry_s * (1 + sl)
                tgt = entry_s * (1 - tp)
            levels = {"entry": entry_s, "stop": stop_s, "t1": tgt, "t2": tgt, "t3": tgt}
        elif self.rr_target:
            # Single target at that R multiple (full exit there). Direction-aware.
            stop_s = levels["stop"]
            risk_s = (entry_s - stop_s).abs()
            if self.side == "long":
                tgt = entry_s + self.rr_target * risk_s
            else:
                tgt = entry_s - self.rr_target * risk_s
            levels = {"entry": entry_s, "stop": stop_s, "t1": tgt, "t2": tgt, "t3": tgt}

        rows = []
        idx = d.index
        for i in np.flatnonzero(entries.to_numpy()):
            ts = idx[i]
            entry = levels["entry"].iloc[i]
            stop = levels["stop"].iloc[i]
            t1 = levels["t1"].iloc[i]
            t2 = levels["t2"].iloc[i]
            t3 = levels["t3"].iloc[i]
            if not np.isfinite([entry, stop, t1]).all():
                continue
            # basic sanity: stop on the correct side of entry
            if self.side == "long" and stop >= entry:
                continue
            if self.side == "short" and stop <= entry:
                continue
            rows.append({
                "date": ts, "side": self.side,
                "entry": float(entry), "stop": float(stop),
                "t1": float(t1), "t2": float(t2), "t3": float(t3),
                "w1": self.weights[0], "w2": self.weights[1], "w3": self.weights[2],
                "max_hold": self.max_hold, "strategy": self.name,
            })
        return pd.DataFrame(rows)


# ==========================================================================
# Strategy 1 — BULL TRAP REVERSAL  (doc §2 / §6.2)  ⭐ priority
# Direction: SHORT (fade a failed breakout). Doc labels "LONG" but the SL is
# above the breakout high and all targets are downside -> mechanically a short.
# ==========================================================================
class BullTrapReversal(Strategy):
    # Doc-faithful thresholds (§6.2). 'relaxed' loosens the volume conjunction
    # and makes RSI-divergence optional so daily data yields tradeable counts.
    STRICT = {"vol_break": 1.5, "vol_rev": 2.0, "require_div": True, "require_macd": True}
    RELAXED = {"vol_break": 1.2, "vol_rev": 1.3, "require_div": False, "require_macd": True}

    def __init__(self, sensitivity: str = "relaxed", **overrides):
        base = dict(self.STRICT if sensitivity == "strict" else self.RELAXED)
        base.update(overrides)
        super().__init__(
            name="Bull Trap Reversal",
            side="short",
            max_hold=10,               # doc: 2-10 days
            rr_min=2.0,
            weights=(0.5, 0.25, 0.25),
            params=base,
            doc_stats={
                "win_rate": 0.681, "cagr": 0.253, "profit_factor": 2.34,
                "max_dd": -0.182, "avg_win": 0.028, "avg_loss": -0.012,
                "trades": 248, "sharpe": 1.87,
            },
        )

    def _entries(self, d: pd.DataFrame) -> pd.Series:
        p = self.params
        # Prior bar = breakout candle; current bar = reversal candle.
        prev_break = sig.breakout_candle(d).shift(1)
        prev_high = d["High"].shift(1)
        prev_high2 = d["High"].shift(2)
        prev_close = d["Close"].shift(1)
        prev_open = d["Open"].shift(1)

        # Reversal candle (current): close below prior close & prior open; high rejected.
        reversal = (d["Close"] < prev_close) & (d["Close"] < prev_open) & (d["High"] <= prev_high)

        # Volume: breakout bar and reversal bar volume thresholds.
        vol_break = (d["Volume"].shift(1) >= p["vol_break"] * d["VOLMA20"].shift(1))
        vol_rev = (d["Volume"] >= p["vol_rev"] * d["VOLMA20"])

        cond = prev_break.fillna(False) & reversal & vol_break & vol_rev

        if p["require_div"]:
            rsi_div = (prev_high > prev_high2) & (d["RSI14"].shift(1) <= d["RSI14"].shift(2))
            cond &= rsi_div.fillna(False)
        if p["require_macd"]:
            cond &= (d["MACD"] < d["MACD_SIGNAL"])

        return cond

    def _levels(self, d: pd.DataFrame) -> dict[str, pd.Series]:
        # Entry 0.5-1% below rejection candle close (use 0.75%).
        entry = d["Close"] * (1 - 0.0075)
        # Hard SL = highest point of breakout attempt, but capped at 1.5*ATR
        # above entry so risk:reward stays sane (raw breakout highs can be
        # absurdly far, wrecking R:R -- the doc's swing-point levels were
        # degenerate on daily data, so we anchor targets to R-multiples).
        raw_stop = pd.concat([d["High"], d["High"].shift(1)], axis=1).max(axis=1)
        capped = entry + 1.5 * d["ATR14"]
        stop = pd.concat([raw_stop, capped], axis=1).min(axis=1)
        risk = (stop - entry).abs()               # per-share risk
        # Downside targets at 2R / 3R / 4R (doc: min 1:2, target 1:3-1:4).
        t1 = entry - 2.0 * risk
        t2 = entry - 3.0 * risk
        t3 = entry - 4.0 * risk
        return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "t3": t3}


# ==========================================================================
# Strategy 2 — ACCUMULATION BREAKOUT  (doc §3 / §6.3)   LONG
# ==========================================================================
class AccumulationBreakout(Strategy):
    STRICT = {"range_pct": 0.015, "low_vol": 0.8, "vol_expand": 2.5,
              "upper_pct": 0.20, "box": 4}
    RELAXED = {"range_pct": 0.05, "low_vol": 1.0, "vol_expand": 1.8,
               "upper_pct": 0.35, "box": 4}

    def __init__(self, sensitivity: str = "relaxed", **overrides):
        base = dict(self.STRICT if sensitivity == "strict" else self.RELAXED)
        base.update(overrides)
        super().__init__(
            name="Accumulation Breakout",
            side="long",
            max_hold=15,               # doc: 4-15 days
            rr_min=2.5,
            weights=(0.40, 0.35, 0.25),
            params=base,
            doc_stats={
                "win_rate": 0.643, "cagr": 0.238, "profit_factor": 2.18,
                "max_dd": -0.221, "avg_win": 0.034, "avg_loss": -0.018,
                "trades": 186, "sharpe": 1.64,
            },
        )

    def _entries(self, d: pd.DataFrame) -> pd.Series:
        p = self.params
        box = int(p["box"])
        box_high = sig.consolidation_high(d, box)
        # Consolidation: prior bars tight range, low volume.
        prior_tight = sig.tight_range(d, box, p["range_pct"]).shift(1).fillna(False)
        low_vol = (d["Volume"].shift(1).rolling(box).mean() <= p["low_vol"] * d["VOLMA20"].shift(1))
        # Breakout: close above box, volume expansion, strong close.
        breakout = d["Close"] > box_high
        vol_expand = d["Volume"] >= p["vol_expand"] * d["VOLMA20"]
        strong_close = sig.close_in_upper_pct(d, p["upper_pct"])
        momentum = (d["RSI14"] > 50) & (d["MACD"] > d["MACD_SIGNAL"])
        return (prior_tight & low_vol.fillna(False) & breakout
                & vol_expand & strong_close.fillna(False) & momentum)

    def _levels(self, d: pd.DataFrame) -> dict[str, pd.Series]:
        entry = d["Close"] * (1 + 0.003)          # 0.3% above breakout close
        # SL = box low, but capped at 1.5*ATR below entry to bound risk.
        raw_stop = sig.consolidation_low(d, int(self.params["box"]))
        capped = entry - 1.5 * d["ATR14"]
        stop = pd.concat([raw_stop, capped], axis=1).max(axis=1)
        risk = (entry - stop).abs()
        # Upside targets at 2.5R / 4R / 5R (doc: min 1:2.5, target 1:4-1:5).
        t1 = entry + 2.5 * risk
        t2 = entry + 4.0 * risk
        t3 = entry + 5.0 * risk
        return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "t3": t3}


# ==========================================================================
# Strategy 3 — AGGRESSIVE VOLUME EXPANSION / CLIMAX REVERSAL (doc §4 / §6.4) LONG
# ==========================================================================
class VolumeClimaxReversal(Strategy):
    STRICT = {"climax_vol": 4.0, "buy_vol": 3.0, "oversold_pct": 0.03,
              "rsi_max": 30, "require_stoch": True}
    RELAXED = {"climax_vol": 2.5, "buy_vol": 1.8, "oversold_pct": 0.02,
               "rsi_max": 40, "require_stoch": False}

    def __init__(self, sensitivity: str = "relaxed", **overrides):
        base = dict(self.STRICT if sensitivity == "strict" else self.RELAXED)
        base.update(overrides)
        super().__init__(
            name="Volume Climax Reversal",
            side="long",
            max_hold=7,                # doc: 1-7 days
            rr_min=3.5,
            weights=(0.30, 0.40, 0.30),
            params=base,
            doc_stats={
                "win_rate": 0.582, "cagr": 0.221, "profit_factor": 2.08,
                "max_dd": -0.253, "avg_win": 0.041, "avg_loss": -0.013,
                "trades": 312, "sharpe": 1.52,
            },
        )

    def _entries(self, d: pd.DataFrame) -> pd.Series:
        p = self.params
        # Prior bar = selling climax (red, high vol, oversold vs 20MA).
        climax_red = sig.is_red(d).shift(1).fillna(False)
        climax_vol = (d["Volume"].shift(1) >= p["climax_vol"] * d["VOLMA20"].shift(1))
        oversold = d["Low"].shift(1) < (d["SMA20"].shift(2) * (1 - p["oversold_pct"]))
        # Current bar = buyer response (green, high vol, close > prior close).
        buy_green = sig.is_green(d)
        buy_vol = d["Volume"] >= p["buy_vol"] * d["VOLMA20"]
        recover = d["Close"] > d["Close"].shift(1)
        strong_close = sig.close_in_upper_half(d, 0.40)  # upper 40% of range
        # Momentum: prior RSI oversold, RSI recovering.
        rsi_ok = (d["RSI14"].shift(1) < p["rsi_max"]) & (d["RSI14"] > d["RSI14"].shift(1))
        cond = (climax_red & climax_vol & oversold.fillna(False)
                & buy_green & buy_vol & recover & strong_close.fillna(False)
                & rsi_ok.fillna(False))
        if p["require_stoch"]:
            stoch_ok = (d["STOCH_K"].shift(1) < 25) & (d["STOCH_K"] > d["STOCH_D"])
            cond &= stoch_ok.fillna(False)
        return cond

    def _levels(self, d: pd.DataFrame) -> dict[str, pd.Series]:
        entry = d["Close"] * (1 + 0.004)
        # SL = low of reversal candle, capped at 1.5*ATR below entry.
        raw_stop = d["Low"]
        capped = entry - 1.5 * d["ATR14"]
        stop = pd.concat([raw_stop, capped], axis=1).max(axis=1)
        risk = (entry - stop).abs()
        # Upside targets at 3.5R / 5R / 6R (doc: min 1:3.5, target 1:5-1:6).
        t1 = entry + 3.5 * risk
        t2 = entry + 5.0 * risk
        t3 = entry + 6.0 * risk
        return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "t3": t3}


# ==========================================================================
# Strategy 4 — WILLIAMS %R(280) DEEP-OVERSOLD  (from the Chartink-style image)
# Buy when Williams %R(280) crosses below -90 (extremely oversold). The image
# uses both weekly and daily; here the strategy runs on ONE timeframe at a time
# (the data it is given) so the UI's Interval dropdown (1d / 1wk) selects which.
# Long mean-reversion. The image's market-cap > 10,000 Cr filter is applied via
# the chosen universe (large/midcaps).
# ==========================================================================
class WilliamsROversold(Strategy):
    # period 280 & level -90 are the image's literal values.
    STRICT = {"period": 280, "level": -90}
    RELAXED = {"period": 180, "level": -85}

    def __init__(self, sensitivity: str = "strict", **overrides):
        # Default STRICT: this is a literal transcription of the image and,
        # unlike the doc strategies, it produces a workable number of signals.
        base = dict(self.STRICT if sensitivity != "relaxed" else self.RELAXED)
        base.update(overrides)
        super().__init__(
            name="Williams %R Oversold",
            side="long",
            max_hold=20,               # deep-oversold bounces can take longer
            rr_min=2.0,
            weights=(0.4, 0.3, 0.3),
            params=base,
            doc_stats={},              # user's own scan — no published claim
        )

    def prepare(self, df, bench_close=None):
        d = super().prepare(df, bench_close=bench_close)
        d["WR"] = ind.williams_r(d, int(self.params["period"]))
        return d

    def _entries(self, d: pd.DataFrame) -> pd.Series:
        lvl = self.params["level"]
        # Crossed below the level on this timeframe: was >= lvl, now < lvl.
        crossed = (d["WR"] < lvl) & (d["WR"].shift(1) >= lvl)
        return crossed.fillna(False)

    def _levels(self, d: pd.DataFrame) -> dict[str, pd.Series]:
        entry = d["Close"] * (1 + 0.002)
        # Stop below the recent low, bounded to 2*ATR.
        raw_stop = d["Low"].rolling(5).min()
        capped = entry - 2.0 * d["ATR14"]
        stop = pd.concat([raw_stop, capped], axis=1).max(axis=1)
        risk = (entry - stop).abs()
        # Mean-reversion targets at 2R / 3R / 4R.
        t1 = entry + 2.0 * risk
        t2 = entry + 3.0 * risk
        t3 = entry + 4.0 * risk
        return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "t3": t3}


# ==========================================================================
# Strategy 5 — SUPERTREND SECTOR MOMENTUM  (PLAN B)   LONG   ⭐
# A long-only swing setup built on the Supertrend indicator plus confirmations:
# "a rising stock, in an uptrend, breaking out on rising volume with buyers
# accumulating." Sector-rotation and fundamentals are handled by the shared
# pipeline (Layers 3-4); this class only supplies the technical entry + levels.
#
# Entry (all must hold):
#   * daily Supertrend GREEN (ST_DIR == +1)
#   * weekly Supertrend GREEN         (strict only; dropped in relaxed)
#   * Close > EMA20 and EMA20 > EMA50 (trend structure)
#   * RSI(14) in [rsi_min, rsi_max]   (momentum up, not over-extended)
#   * ADX(14) > adx_min               (genuinely trending, not chop)
#   * Volume >= vol_mult * VOLMA20    (volume rising on the move)
#   * OBV rising over obv_lb bars     (accumulation)
#   * Close breaks above the recent breakout_lb-day high (breakout)
#
# Levels: entry = close + 0.3%; stop = the Supertrend line (capped to 1.5*ATR),
#         i.e. a Supertrend trailing stop; targets at 2R / 3R / 4R.
# ==========================================================================
class SupertrendSectorMomentum(Strategy):
    # Every confirming filter is an independent toggle (use_*) so the Experiment
    # Lab can grid-search which combination works. The daily Supertrend-green
    # core is ALWAYS on (it defines the strategy); everything else is optional.
    # Defaults below reproduce the original Plan B behaviour exactly:
    #   strict  = weekly filter on, tight RSI/ADX;
    #   relaxed = daily-only, looser thresholds, more signals.
    STRICT = {
        "st_period": 10, "st_mult": 3.0,
        "use_weekly": True, "require_weekly": True,   # require_weekly kept as alias
        "use_ema_structure": True,
        "use_ema200": False,
        "use_rsi": True, "rsi_min": 50, "rsi_max": 70,
        "use_adx": True, "adx_min": 25,
        "use_volume": True, "vol_mult": 1.5,
        "use_obv": True, "obv_lb": 5,
        "use_macd": False,
        "use_breakout": True, "breakout_lb": 20,
    }
    RELAXED = {
        "st_period": 10, "st_mult": 3.0,
        "use_weekly": False, "require_weekly": False,
        "use_ema_structure": True,
        "use_ema200": False,
        "use_rsi": True, "rsi_min": 50, "rsi_max": 80,
        "use_adx": True, "adx_min": 20,
        "use_volume": True, "vol_mult": 1.5,
        "use_obv": True, "obv_lb": 5,
        "use_macd": False,
        "use_breakout": True, "breakout_lb": 10,
    }

    def __init__(self, sensitivity: str = "relaxed", **overrides):
        base = dict(self.STRICT if sensitivity == "strict" else self.RELAXED)
        base.update(overrides)
        # Keep the two weekly flags in sync whichever the caller sets.
        if "use_weekly" in overrides and "require_weekly" not in overrides:
            base["require_weekly"] = base["use_weekly"]
        elif "require_weekly" in overrides and "use_weekly" not in overrides:
            base["use_weekly"] = base["require_weekly"]
        super().__init__(
            name="Supertrend Sector Momentum",
            side="long",
            max_hold=30,               # swing horizon; trail on Supertrend
            rr_min=2.0,
            weights=(0.5, 0.25, 0.25),
            params=base,
            doc_stats={},              # our own design — no published claim
        )

    def _wants_weekly(self) -> bool:
        return bool(self.params.get("use_weekly", self.params.get("require_weekly", False)))

    def prepare(self, df, bench_close=None):
        d = super().prepare(df, bench_close=bench_close)
        # Non-default Supertrend settings: recompute SUPERT/ST_DIR with the
        # chosen period/multiplier (enrich() uses the standard 10/3).
        stp = int(self.params.get("st_period", 10))
        stm = float(self.params.get("st_mult", 3.0))
        if (stp, stm) != (10, 3.0):
            st_line, st_dir = ind.supertrend(d, stp, stm)
            d["SUPERT"], d["ST_DIR"] = st_line, st_dir
        # Weekly Supertrend, computed point-in-time: resample daily -> weekly,
        # run Supertrend on the weekly bars, then forward-fill the weekly
        # direction back onto the daily index (only past weekly closes are used
        # because each weekly bar is stamped at its week-end date and ffill'd).
        if self._wants_weekly():
            try:
                wk = df.resample("W-FRI").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
                if len(wk) > 12:
                    _, wk_dir = ind.supertrend(wk, stp, stm)
                    d["WK_ST_DIR"] = wk_dir.reindex(d.index, method="ffill")
                else:
                    d["WK_ST_DIR"] = 1  # not enough weekly history -> don't block
            except Exception:
                d["WK_ST_DIR"] = 1
        return d

    def _entries(self, d: pd.DataFrame) -> pd.Series:
        p = self.params
        # Core (always on): daily Supertrend green defines the strategy.
        cond = (d["ST_DIR"] == 1)
        # Each confirming filter is applied only when its use_* flag is set, so
        # the Experiment Lab can toggle indicators independently.
        if p.get("use_ema_structure", True):
            cond &= (d["Close"] > d["EMA20"]) & (d["EMA20"] > d["EMA50"])
        if p.get("use_ema200", False):
            cond &= (d["Close"] > d["EMA200"])
        if p.get("use_rsi", True):
            cond &= ((d["RSI14"] >= p["rsi_min"]) & (d["RSI14"] <= p["rsi_max"])).fillna(False)
        if p.get("use_adx", True):
            cond &= (d["ADX14"] > p["adx_min"]).fillna(False)
        if p.get("use_volume", True):
            cond &= (d["Volume"] >= p["vol_mult"] * d["VOLMA20"]).fillna(False)
        if p.get("use_obv", True):
            cond &= (d["OBV"] > d["OBV"].shift(int(p["obv_lb"]))).fillna(False)
        if p.get("use_macd", False):
            cond &= (d["MACD"] > d["MACD_SIGNAL"]).fillna(False)
        if p.get("use_breakout", True):
            prior_high = d["High"].shift(1).rolling(int(p["breakout_lb"])).max()
            cond &= (d["Close"] > prior_high).fillna(False)
        if self._wants_weekly():
            cond &= (d.get("WK_ST_DIR", 1) == 1)
        return cond

    def _levels(self, d: pd.DataFrame) -> dict[str, pd.Series]:
        entry = d["Close"] * (1 + 0.003)          # 0.3% above signal close
        # Stop = the Supertrend line (the trailing-stop anchor), but never wider
        # than 1.5*ATR below entry so risk stays bounded when ST is far away.
        raw_stop = d["SUPERT"]
        capped = entry - 1.5 * d["ATR14"]
        stop = pd.concat([raw_stop, capped], axis=1).max(axis=1)
        # If ST is above entry (rare, transitional bar) fall back to the ATR cap.
        stop = stop.where(stop < entry, capped)
        risk = (entry - stop).abs()
        t1 = entry + 2.0 * risk
        t2 = entry + 3.0 * risk
        t3 = entry + 4.0 * risk
        return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "t3": t3}


# Registry for the UI / backtester.
STRATEGIES = {
    "Bull Trap Reversal": BullTrapReversal,
    "Accumulation Breakout": AccumulationBreakout,
    "Volume Climax Reversal": VolumeClimaxReversal,
    "Williams %R Oversold": WilliamsROversold,
    "Supertrend Sector Momentum": SupertrendSectorMomentum,
}


def enabled_strategies() -> list[str]:
    """Names of strategies enabled in config.STRATEGY_ENABLED, in registry order.

    A strategy is shown only if its flag is True. Unknown/missing keys default
    to enabled so adding a new strategy never silently hides it.
    """
    from . import config
    flags = getattr(config, "STRATEGY_ENABLED", {})
    names = [n for n in STRATEGIES if flags.get(n, True)]
    return names or list(STRATEGIES)  # never return empty (safety)


def get_strategy(name: str, sensitivity: str = "relaxed",
                 rr_target: float | None = None,
                 pct_stop: float | None = None,
                 pct_target_mult: float = 2.0,
                 trail_amount: float | None = None,
                 trail_is_pct: bool = False, **overrides) -> Strategy:
    strat = STRATEGIES[name](sensitivity=sensitivity, **overrides)
    if rr_target:
        strat.rr_target = float(rr_target)
    if pct_stop:
        strat.pct_stop = float(pct_stop)
        strat.pct_target_mult = float(pct_target_mult)
    if trail_amount:
        strat.trail_amount = float(trail_amount)
        strat.trail_is_pct = bool(trail_is_pct)
    return strat
