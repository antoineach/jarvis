"""Per-call latency report for LLM backends (``JARVIS_TIMINGS=1``).

A voice turn runs several model calls (intent judge, planner, tool router, reply, digests). Without
timings it is impossible to tell which stage makes a turn slow, so every backend call can print one
line naming the pipeline function that issued it:

    ⏱️ llm.chat      1.84s  select_tools/_select_llm  (812→41 tok)

Off by default; enabled by the environment variable so it works for every entry point (daemon,
desktop app, evals) without threading a setting through each one.
"""

from __future__ import annotations

import functools
import os
import sys
import time
from typing import Any, Callable

_OWN_DIRS = (os.sep + "llm" + os.sep,)


def enabled() -> bool:
    return os.environ.get("JARVIS_TIMINGS", "").strip().lower() in ("1", "true", "yes", "on")


def _caller(depth: int = 2) -> str:
    """The first two pipeline functions above the backend (outermost/innermost)."""
    names = []
    f = sys._getframe(depth)
    while f is not None and len(names) < 2:
        path = f.f_code.co_filename
        if "jarvis" in path and not any(d in path for d in _OWN_DIRS):
            names.append(f.f_code.co_name)
        f = f.f_back
    return "/".join(reversed(names)) or "?"


def _usage(result: Any) -> str:
    if isinstance(result, dict):
        u = result.get("usage") or {}
        p = u.get("prompt_tokens", result.get("prompt_eval_count"))
        c = u.get("completion_tokens", result.get("eval_count"))
        if p is not None or c is not None:
            return f"  ({p}→{c} tok)"
    return ""


def timed(kind: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            who = _caller()
            t0 = time.perf_counter()
            ok = False
            try:
                result = fn(*args, **kwargs)
                ok = result is not None
                return result
            finally:
                dt = time.perf_counter() - t0
                tail = _usage(locals().get("result")) if ok else "  (failed)"
                print(f"  ⏱️ llm.{kind:<9} {dt:5.2f}s  {who}{tail}", flush=True)
        return wrapper
    return deco


def stage(name: str, t0: float) -> None:
    """Report a non-LLM stage (speech recognition, first audio, whole turn)."""
    if enabled():
        print(f"  ⏱️ {name:<13} {time.perf_counter() - t0:5.2f}s", flush=True)
