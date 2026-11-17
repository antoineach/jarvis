"""Start llama.cpp servers with Jarvis, so one launch command brings up the whole assistant.

Jarvis talks to an already-running LLM server (Ollama or any OpenAI-compatible one). With
llama.cpp that means starting ``llama-server`` by hand first; forgetting it makes every model
probe fail. When ``llama_server_path`` and ``llama_server_model`` are set, the daemon starts:

- the chat server on the port of ``llm_base_url`` (GPU, ``llama_server_args``);
- optionally an embedding server on the port of ``embedding_base_url`` for
  ``llama_server_embedding_model`` (CPU, small batch-friendly context), since one llama-server
  process serves a single model.

A server already answering on its port is reused, never restarted. Servers Jarvis started are
stopped when Jarvis exits. Only loopback URLs are managed.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from ..debug import debug_log

# Three slots sharing one unified 16k KV cache (q8_0 keeps it the size of an 8k f16 cache; a turn
# uses ~10k: reply ~6.5k, judge ~1.3k, planner ~2k): slot 0 keeps the reply prompt between turns,
# slot 1 the intent judge, slot 2 the other short calls (planner, digests), so none evicts another. With llama-server's default (4 slots, 8k shared) every short call evicted the reply
# prompt: measured 0 cached tokens on every reply, ~3 s of re-reading per turn.
DEFAULT_ARGS = ["-ngl", "99", "-c", "16384", "-np", "3", "--kv-unified", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--swa-full", "--cache-ram", "0", "-ot", "per_layer_token_embd\\.weight=CPU", "--jinja"]
# --swa-full: Gemma-family models (Gemma 4, Nemotron-3.5-Bolt-E4B) use sliding-window attention
# layers whose default cache only holds the last window, so llama.cpp cannot roll back to a prefix
# that diverges mid-prompt and re-reads everything (measured: 0 cached tokens while 70% of the reply
# prompt was identical turn to turn). The full SWA cache is a few hundred MB at 16k for an E4B.
# --cache-ram 0: recent llama-server builds move a released slot's KV to a host-RAM prompt cache
# and empty the slot (measured: n_tokens = 6129 at release, 0 at the next select of the same slot,
# full re-read every turn). Without the host cache the KV stays in the pinned slot and the common
# prefix is reused in place.
# -ot per_layer_token_embd=CPU: Gemma 4 / Gemma 3n "E" models carry large per-layer embedding
# tables that are only looked up, never multiplied; keeping them in RAM frees VRAM (Unsloth lists
# the E4B Q4_K_M at ~4 GB VRAM plus system RAM). No effect on models without that tensor.
REPLY_SLOT, JUDGE_SLOT, FAST_SLOT = 0, 1, 2
_pinning = False
_slots = 1
EMBED_ARGS = ["--embeddings", "-ngl", "0", "-c", "2048", "-b", "2048", "-ub", "2048"]
READY_TIMEOUT_S = 300.0
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_started: List[subprocess.Popen] = []


def _root(base_url: str) -> str:
    u = urlparse(base_url)
    return f"{u.scheme or 'http'}://{u.hostname}:{u.port or 80}"


def _health(base_url: str, timeout: float = 1.5) -> Optional[int]:
    """HTTP status of /health (200 ready, 503 loading), None when nothing listens."""
    try:
        with urllib.request.urlopen(_root(base_url) + "/health", timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def _log_path(name: str) -> Path:
    p = Path.home() / ".local" / "share" / "jarvis" / f"{name}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _tail(path: Path, n: int = 12) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _start(name: str, exe: str, model: str, base_url: str, args: List[str], cpu_only: bool = False) -> bool:
    u = urlparse(base_url)
    if u.hostname not in _LOCAL_HOSTS:
        debug_log(f"{name}: {base_url} is not local, not starting a server", "llm")
        return True
    if _health(base_url) is not None:
        print(f"  ✅ {name}: server already running on port {u.port}", flush=True)
        return True
    for label, path in (("llama_server_path", exe), (f"model for {name}", model)):
        if not Path(path).exists():
            print(f"  ❌ {name}: {label} not found: {path}", flush=True)
            return False
    cmd = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(u.port or 8080),
           "--alias", Path(model).stem, *args]
    log = _log_path(f"llama_server_{name}")
    print(f"  🚀 Starting llama-server for {name} ({Path(model).name}, port {u.port})...", flush=True)
    debug_log(f"{name}: {' '.join(cmd)}", "llm")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    env = None
    if cpu_only:
        # -ngl 0 alone still creates a CUDA context and compute buffers (several hundred MB of VRAM
        # on a CUDA build): hide the GPU entirely so the chat model and the voice keep it.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, creationflags=flags, env=env)
    _started.append(proc)
    t0 = time.time()
    while time.time() - t0 < READY_TIMEOUT_S:
        if proc.poll() is not None:
            print(f"  ❌ llama-server ({name}) exited with code {proc.returncode}. Log {log}:\n{_tail(log)}",
                  flush=True)
            return False
        if _health(base_url) == 200:
            print(f"  ✅ {name}: llama-server ready in {time.time() - t0:.0f}s", flush=True)
            return True
        time.sleep(0.5)
    print(f"  ❌ llama-server ({name}) not ready after {READY_TIMEOUT_S:.0f}s. Log: {log}", flush=True)
    return False


def _parallel(args: List[str]) -> int:
    for flag in ("-np", "--parallel"):
        if flag in args:
            try:
                return int(args[args.index(flag) + 1])
            except (IndexError, ValueError):
                return 1
    return 1


def slot_for(kind: str) -> Optional[int]:
    """llama-server slot for a request, when Jarvis started the server with >= 2 slots.

    ``kind``: "reply" (chat with tool schemas: only the reply loop sends them), "chat" (chat
    without tools: the intent judge) or "direct" (planner, digests, extraction, summaries).
    Each kind keeps its own stable prompt prefix cached. With 2 slots the short calls share one.
    """
    if not _pinning:
        return None
    if kind == "reply":
        return REPLY_SLOT
    if _slots >= 3:
        return JUDGE_SLOT if kind == "chat" else FAST_SLOT
    return JUDGE_SLOT


_CACHE_LOG = re.compile(r"select:|cache|checkpoint|swa|prompt eval time|n_past|restored|re-process|"
                        r"update_slots|release:|offload|CPU_Mapped|CUDA0 model buffer", re.I)
_log_pos: dict = {}


def report_server_log(name: str = "chat", max_lines: int = 8) -> None:
    """Echo the llama-server log lines about slots and prompt caching written since the last call.

    Diagnostics for JARVIS_TIMINGS: shows the server's own reason for a cache miss (slot chosen,
    tokens reused, SWA/checkpoint messages) next to Jarvis's timing lines, no log digging needed.
    """
    path = _log_path(f"llama_server_{name}")
    try:
        size = path.stat().st_size
        start = _log_pos.get(name, 0)
        if size < start:
            start = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(start)
            new = f.read()
            _log_pos[name] = f.tell()
    except OSError:
        return
    lines = [ln.strip() for ln in new.splitlines() if _CACHE_LOG.search(ln)]
    for ln in lines[-max_lines:]:
        print(f"     🦙 {ln[-170:]}", flush=True)


def report_vram() -> None:
    """One line of GPU memory use (nvidia-smi), when available."""
    import shutil
    smi = shutil.which("nvidia-smi")
    if not smi:
        return
    try:
        out = subprocess.run([smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        used, total = (int(x) for x in out[0].split(","))
        warn = "  ⚠️ nearly full: Windows may spill to system RAM and slow everything" if used > 0.93 * total else ""
        print(f"  📊 VRAM {used} / {total} MiB{warn}", flush=True)
    except Exception:
        pass


def stop_started_servers() -> None:
    while _started:
        p = _started.pop()
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def ensure_local_servers(cfg) -> None:
    """Start the configured llama.cpp servers that are not already running (blocking until ready)."""
    exe = str(getattr(cfg, "llama_server_path", "") or "").strip()
    model = str(getattr(cfg, "llama_server_model", "") or "").strip()
    if not (exe and model):
        return
    exe, model = os.path.expanduser(exe), os.path.expanduser(model)
    atexit.register(stop_started_servers)
    global _pinning, _slots
    args = list(getattr(cfg, "llama_server_args", None) or DEFAULT_ARGS)
    if _start("chat", exe, model, cfg.llm_base_url, args) and _parallel(args) >= 2:
        _pinning, _slots = True, _parallel(args)
    emb = str(getattr(cfg, "llama_server_embedding_model", "") or "").strip()
    emb_url = str(getattr(cfg, "embedding_base_url", "") or "").strip()
    if emb and emb_url and _root(emb_url) != _root(cfg.llm_base_url):
        _start("embeddings", exe, os.path.expanduser(emb), emb_url, EMBED_ARGS, cpu_only=True)
