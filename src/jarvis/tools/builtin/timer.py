"""Countdown timers ("ten-minute timer for the pasta").

Timers run in-process as daemon threads and are announced when they ring through a handler the
daemon installs (``set_ring_handler``): by default it speaks the timer's label through the active
TTS engine, so a ringing timer interrupts nothing but is heard even mid-conversation.

The tool is LLM-free: the model converts the spoken duration to seconds, the tool only counts.
Timers do not survive a restart (in-memory only), which the tool result states.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

_lock = threading.Lock()
_timers: Dict[str, tuple[float, threading.Timer]] = {}
_ring_handler: Optional[Callable[[str], None]] = None


def set_ring_handler(handler: Optional[Callable[[str], None]]) -> None:
    """Install the callback called with the timer label when a timer rings."""
    global _ring_handler
    _ring_handler = handler


def _ring(label: str) -> None:
    with _lock:
        _timers.pop(label, None)
    debug_log(f"timer rang: {label}", "tools")
    print(f"⏰ Timer finished: {label}", flush=True)
    handler = _ring_handler
    if handler is not None:
        try:
            handler(label)
        except Exception as e:
            debug_log(f"timer ring handler failed: {e}", "tools")


def _fmt(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


def active_timers() -> Dict[str, float]:
    """Label -> seconds remaining."""
    now = time.time()
    with _lock:
        return {k: end - now for k, (end, _) in _timers.items()}


class TimerTool(Tool):
    @property
    def name(self) -> str:
        return "timer"

    @property
    def description(self) -> str:
        return ("Start, list or cancel countdown timers (cooking, reminders in a few minutes). "
                "Convert the spoken duration to seconds yourself (ten minutes = 600). "
                "The timer is announced out loud when it ends.")

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "list", "cancel"]},
                "seconds": {"type": "integer", "minimum": 1, "maximum": 86400,
                            "description": "Duration in seconds (start)"},
                "label": {"type": "string", "description": "Short name in the user's language, e.g. pasta"},
            },
            "required": ["action"],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        a = args or {}
        action = str(a.get("action", "start")).lower()
        label = str(a.get("label") or "timer").strip().lower()
        if action == "list":
            items = active_timers()
            if not items:
                return ToolExecutionResult(True, "No active timers.")
            return ToolExecutionResult(True, "Active timers: " + "; ".join(
                f"{k}: {_fmt(v)} remaining" for k, v in items.items()))
        if action == "cancel":
            with _lock:
                if label not in _timers and len(_timers) == 1:
                    label = next(iter(_timers))
                entry = _timers.pop(label, None)
            if entry is None:
                return ToolExecutionResult(False, None, error_message=f"No active timer named '{label}'.")
            entry[1].cancel()
            return ToolExecutionResult(True, f"Timer '{label}' cancelled.")
        try:
            seconds = int(a["seconds"])
        except (KeyError, TypeError, ValueError):
            return ToolExecutionResult(False, None, error_message="'seconds' is required to start a timer.")
        if not 1 <= seconds <= 86400:
            return ToolExecutionResult(False, None, error_message="Timer duration must be 1 s to 24 h.")
        with _lock:
            if label in _timers:
                _timers.pop(label)[1].cancel()
            t = threading.Timer(seconds, _ring, args=(label,))
            t.daemon = True
            _timers[label] = (time.time() + seconds, t)
            t.start()
        context.user_print(f"⏲️ Timer '{label}' set for {_fmt(seconds)}")
        return ToolExecutionResult(True, f"Timer '{label}' started for {_fmt(seconds)} "
                                         "(in memory: lost if Jarvis restarts).")
