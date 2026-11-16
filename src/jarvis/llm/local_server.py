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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from ..debug import debug_log

DEFAULT_ARGS = ["-ngl", "99", "-c", "8192", "--jinja"]
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


def _start(name: str, exe: str, model: str, base_url: str, args: List[str]) -> bool:
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
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, creationflags=flags)
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
    args = list(getattr(cfg, "llama_server_args", None) or DEFAULT_ARGS)
    _start("chat", exe, model, cfg.llm_base_url, args)
    emb = str(getattr(cfg, "llama_server_embedding_model", "") or "").strip()
    emb_url = str(getattr(cfg, "embedding_base_url", "") or "").strip()
    if emb and emb_url and _root(emb_url) != _root(cfg.llm_base_url):
        _start("embeddings", exe, os.path.expanduser(emb), emb_url, EMBED_ARGS)
