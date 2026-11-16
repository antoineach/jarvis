"""Copy the console output of each daemon session to a file, for sharing a run without copy-paste.

``~/.local/share/jarvis/logs/last.log`` is the current session (also kept as ``session-<time>.log``,
newest 10 retained). Disabled with ``JARVIS_SESSION_LOG=0``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

KEEP = 10


class _Tee:
    def __init__(self, stream, files):
        self._stream, self._files = stream, files

    def write(self, s):
        n = self._stream.write(s)
        for f in self._files:
            try:
                f.write(s)
                f.flush()
            except Exception:
                pass
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start() -> Path | None:
    if os.environ.get("JARVIS_SESSION_LOG", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        d = Path.home() / ".local" / "share" / "jarvis" / "logs"
        d.mkdir(parents=True, exist_ok=True)
        session = d / time.strftime("session-%Y%m%d-%H%M%S.log")
        files = [open(session, "w", encoding="utf-8", errors="replace"),
                 open(d / "last.log", "w", encoding="utf-8", errors="replace")]
        for old in sorted(d.glob("session-*.log"))[:-KEEP]:
            old.unlink(missing_ok=True)
        sys.stdout = _Tee(sys.stdout, files)
        sys.stderr = _Tee(sys.stderr, files)
        return d / "last.log"
    except Exception:
        return None
