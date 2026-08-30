"""All-time-best benchmark logger (NEW — decision-support only).

Keeps a leaderboard of the single best combination ever found for each strategy,
ranked by CAGR. Every time someone runs a combination sweep (Experiment Lab),
the winning combo is offered to ``update_best`` — it only overwrites the stored
entry when the new CAGR beats the previous best, so ``benchmark/benchmark.md``
is a running record of champions, not of the latest run.

Two files live in the ``benchmark/`` folder:
  * ``benchmark.json`` — the machine-readable source of truth (one entry per
    strategy). This is what we read/compare/overwrite.
  * ``benchmark.md``   — a human-readable render of the JSON, rewritten whenever
    the JSON changes. This is the file you open to see the leaderboard.

This module does NOT touch the strategies, the backtester, the momentum engine,
or the existing Experiment Lab grid — it only records their results.
"""

from __future__ import annotations

import json
import os

from . import config

# The folder the user created for this. Kept beside the project root.
BENCHMARK_DIR = os.path.join(config.PROJECT_DIR, "benchmark")
JSON_PATH = os.path.join(BENCHMARK_DIR, "benchmark.json")
MD_PATH = os.path.join(BENCHMARK_DIR, "benchmark.md")


def _load() -> dict:
    """Read the current leaderboard (``{strategy: entry}``); {} if none yet."""
    if not os.path.exists(JSON_PATH):
        return {}
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        # Corrupt/unreadable file shouldn't crash a sweep — start fresh.
        return {}


def _save(data: dict) -> None:
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _render_md(data)


def update_best(
    strategy: str,
    cagr: float,
    params: dict,
    metrics: dict | None = None,
    when: str | None = None,
) -> dict:
    """Record ``cagr``/``params`` as the best for ``strategy`` IF it beats the
    stored CAGR (or there's no entry yet). Returns a small status dict:

        {"updated": bool, "previous_cagr": float|None, "entry": <stored entry>}

    ``cagr`` is a fraction (0.23 = 23%). ``params`` is a plain dict of the winning
    combination's knobs. ``metrics`` (optional) carries extras like calmar / max_dd
    / trades that we also show in the .md. ``when`` is an ISO date string supplied
    by the caller (this module never reads the clock so it stays import-safe).
    """
    data = _load()
    prev = data.get(strategy)
    prev_cagr = float(prev["cagr"]) if prev and "cagr" in prev else None

    # Overwrite only when strictly better (or nothing recorded yet).
    if prev_cagr is not None and cagr <= prev_cagr:
        return {"updated": False, "previous_cagr": prev_cagr, "entry": prev}

    entry = {
        "cagr": round(float(cagr), 4),
        "params": {k: _jsonable(v) for k, v in (params or {}).items()},
        "metrics": {k: _jsonable(v) for k, v in (metrics or {}).items()},
        "updated": when or "",
    }
    data[strategy] = entry
    _save(data)
    return {"updated": True, "previous_cagr": prev_cagr, "entry": entry}


def _jsonable(v):
    """Coerce numpy/pandas scalars to plain JSON-safe Python values."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except Exception:
        pass
    return v


def _render_md(data: dict) -> None:
    """Rewrite benchmark.md from the JSON leaderboard (best CAGR first)."""
    lines = [
        "# Benchmark — best combinations",
        "",
        "The single best combination ever recorded for each strategy, ranked by "
        "**CAGR**. Updated automatically whenever a combination sweep finds a new "
        "champion (it only overwrites when the CAGR beats the previous best).",
        "",
    ]
    if not data:
        lines.append("_No runs recorded yet. Run a combination sweep in the Experiment Lab._")
    else:
        ranked = sorted(data.items(), key=lambda kv: kv[1].get("cagr", 0), reverse=True)
        for strat, entry in ranked:
            cagr = entry.get("cagr", 0) * 100
            m = entry.get("metrics", {})
            lines.append(f"## {strat}")
            lines.append("")
            lines.append(f"- **Best CAGR:** {cagr:+.1f}%")
            if "calmar" in m:
                lines.append(f"- **Calmar:** {m['calmar']}")
            if "max_dd_%" in m:
                lines.append(f"- **Max drawdown:** {m['max_dd_%']}%")
            if "sharpe" in m:
                lines.append(f"- **Sharpe:** {m['sharpe']}")
            if "trades" in m:
                lines.append(f"- **Trades:** {m['trades']}")
            params = entry.get("params", {})
            if params:
                pstr = ", ".join(f"`{k}={v}`" for k, v in params.items())
                lines.append(f"- **Parameters:** {pstr}")
            if entry.get("updated"):
                lines.append(f"- **Updated:** {entry['updated']}")
            lines.append("")

    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def get_best(strategy: str) -> dict | None:
    """Return the stored best entry for ``strategy`` (or None)."""
    return _load().get(strategy)
