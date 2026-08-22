"""Silence a benign Windows-only asyncio noise.

On Windows, Streamlit/Tornado run on asyncio's Proactor event loop. When a
browser tab is closed or reloaded, the client drops the socket and Windows
raises ``ConnectionResetError: [WinError 10054]`` deep inside
``_ProactorBasePipeTransport._call_connection_lost`` while it tries to
``shutdown()`` an already-dead socket. It is cosmetic — it does not affect any
computation or result — but it prints an ugly traceback to the console.

This module wraps that one internal method to swallow exactly that error (and
the related "Event loop is closed" RuntimeError) and is a no-op on non-Windows
platforms. Import it once at process start (see app.py)."""

from __future__ import annotations

import functools
import sys


def apply() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
    except Exception:
        return

    orig = _ProactorBasePipeTransport._call_connection_lost

    @functools.wraps(orig)
    def _quiet_call_connection_lost(self, exc):
        try:
            return orig(self, exc)
        except (ConnectionResetError, RuntimeError):
            # 10054 (client dropped the socket) or "Event loop is closed" during
            # teardown — nothing actionable, so don't spam the console.
            pass

    # Guard against double-patching on Streamlit hot-reload.
    if getattr(_ProactorBasePipeTransport._call_connection_lost, "__wrapped__", None) is None:
        _ProactorBasePipeTransport._call_connection_lost = _quiet_call_connection_lost
