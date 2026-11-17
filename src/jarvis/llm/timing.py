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
from typing import Any, Callable, Optional

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
        t = result.get("timings") or {}           # llama-server: tokens reused from its prompt cache
        cache = f", {t['cache_n']} cached" if isinstance(t, dict) and "cache_n" in t else ""
        if p is not None or c is not None:
            return f"  ({p}→{c} tok{cache})"
    return ""


_last_prompt: dict = {}


def prefix_report(key: str, tools: Any, messages: list, render: Optional[Callable] = None) -> None:
    """Print where this prompt first differs from the previous one sent by the same caller.

    The server can only reuse the identical leading part of a prompt from its cache; everything
    after the first difference is recomputed. Message-level, so it names the culprit (which
    message, what text) without needing the server's chat template.
    """
    if not enabled():
        return
    import json
    rendered = None
    if render is not None:                   # the exact text the server tokenizes (its chat template)
        try:
            rendered = render()
        except Exception:
            rendered = None
    if isinstance(rendered, str) and rendered:
        parts = [("rendered prompt", rendered)]
        prev = _last_prompt.get(key)
        _last_prompt[key] = parts
        if prev is not None and prev[0][0] == "rendered prompt":
            a, b = prev[0][1], rendered
            k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            sa = a[max(0, k - 40):k + 160].replace("\n", "⏎")
            sb = b[max(0, k - 40):k + 160].replace("\n", "⏎")
            print(f"  🧩 prompt cache: identical to previous call for the first {100 * k / max(len(b), 1):.0f}% "
                  f"({k}/{len(b)} chars)\n     before: …{sa}…\n     now:    …{sb}…", flush=True)
            try:                                       # full texts, for a closer look
                from pathlib import Path
                d = Path.home() / ".local" / "share" / "jarvis" / "logs"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"prompt-{key.replace(':', '_')}-before.txt").write_text(a, encoding="utf-8")
                (d / f"prompt-{key.replace(':', '_')}-now.txt").write_text(b, encoding="utf-8")
            except Exception:
                pass
        return
    parts = [("tools", json.dumps(tools or [], sort_keys=True, ensure_ascii=False))]
    for i, m in enumerate(messages or []):
        content = m.get("content") if isinstance(m, dict) else m
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
        parts.append((f"message {i} ({m.get('role', '?') if isinstance(m, dict) else '?'})", content))
    prev = _last_prompt.get(key)
    _last_prompt[key] = parts
    if prev is None:
        return
    same = 0
    for (label, a), (_, b) in zip(prev, parts):
        if a == b:
            same += len(a)
            continue
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        total = sum(len(x) for _, x in parts)
        snippet = b[max(0, k - 60):k + 60].replace("\n", " ")
        print(f"  🧩 prompt cache: same as previous call up to {label}, char {k} "
              f"(~{100 * (same + k) / max(total, 1):.0f}% of prompt): …{snippet}…", flush=True)
        return



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
                if kind != "embed":
                    try:
                        from .local_server import _started, report_server_log
                        if _started:
                            report_server_log("chat")
                    except Exception:
                        pass
        return wrapper
    return deco


_speech_end = {"t": 0.0}


def mark_speech_end(t: float) -> None:
    """Wall-clock time the user stopped speaking (endpoint time minus the endpoint silence)."""
    _speech_end["t"] = t


def first_audio() -> None:
    """Print the latency the user perceives: end of their speech -> first audio out."""
    t = _speech_end["t"]
    if enabled() and t:
        print(f"  ⏱️ heard → voice {time.time() - t:5.2f}s  (end of your speech to first audio)", flush=True)
        _speech_end["t"] = 0.0


def stage(name: str, t0: float) -> None:
    """Report a non-LLM stage (speech recognition, first audio, whole turn)."""
    if enabled():
        print(f"  ⏱️ {name:<13} {time.perf_counter() - t0:5.2f}s", flush=True)
