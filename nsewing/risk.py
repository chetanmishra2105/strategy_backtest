"""Risk management — position sizing, VIX-scaled risk, drawdown triggers.

All rules transcribed from the document's §5 framework.
"""

from __future__ import annotations

from . import config


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> dict:
    """Doc §5.1: Position Size = (Capital x Risk%) / |Entry - SL|.

    Returns shares, rupee risk, and notional exposure.
    """
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return {"shares": 0, "rupee_risk": 0.0, "notional": 0.0, "per_share_risk": 0.0}
    rupee_risk = capital * risk_pct
    shares = int(rupee_risk // per_share_risk)
    return {
        "shares": shares,
        "rupee_risk": round(shares * per_share_risk, 2),
        "notional": round(shares * entry, 2),
        "per_share_risk": round(per_share_risk, 2),
    }


def vix_regime(vix: float | None) -> dict:
    """Doc §5.2: map India VIX to a regime, suggested risk %, and note."""
    if vix is None:
        return {"vix": None, "label": "Unknown", "risk_pct": config.DEFAULT_RISK_PCT,
                "note": "VIX unavailable — using default risk."}
    for lo, hi, label, risk_pct, note in config.VIX_REGIMES:
        if lo <= vix < hi:
            return {"vix": round(vix, 2), "label": label, "risk_pct": risk_pct, "note": note}
    return {"vix": round(vix, 2), "label": "Unknown", "risk_pct": config.DEFAULT_RISK_PCT,
            "note": ""}


def drawdown_alerts(daily=0.0, weekly=0.0, monthly=0.0, quarterly=0.0) -> list[str]:
    """Doc §5.3: return human-readable warnings for breached drawdown triggers.

    Inputs are fractional drawdowns (e.g. -0.04 for -4%).
    """
    t = config.DRAWDOWN_TRIGGERS
    alerts = []
    if abs(daily) >= t["daily"]:
        alerts.append(f"Daily drawdown {daily*100:.1f}% ≥ {t['daily']*100:.1f}% — take the rest of the day off.")
    if abs(weekly) >= t["weekly"]:
        alerts.append(f"Weekly drawdown {weekly*100:.1f}% ≥ {t['weekly']*100:.1f}% — cut position size 50% next week.")
    if abs(monthly) >= t["monthly"]:
        alerts.append(f"Monthly drawdown {monthly*100:.1f}% ≥ {t['monthly']*100:.1f}% — HALT trading until reviewed.")
    if abs(quarterly) >= t["quarterly"]:
        alerts.append(f"Quarterly drawdown {quarterly*100:.1f}% ≥ {t['quarterly']*100:.1f}% — major strategy review.")
    return alerts


def target_for_4pct(entry: float, side: str = "long") -> float:
    """Convenience: the price that yields the user's headline 4% target."""
    return entry * (1.04 if side == "long" else 0.96)
