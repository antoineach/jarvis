import sys

import pytest

from jarvis import session_log

pytestmark = [pytest.mark.unit]


def test_console_is_copied_to_last_log(tmp_path, monkeypatch):
    monkeypatch.setattr(session_log.Path, "home", lambda: tmp_path)
    out, err = sys.stdout, sys.stderr
    try:
        path = session_log.start()
        print("📝 Heard: Quelle heure est-il ?")
    finally:
        sys.stdout, sys.stderr = out, err
    assert "Quelle heure" in path.read_text(encoding="utf-8")


def test_can_be_disabled(monkeypatch):
    monkeypatch.setenv("JARVIS_SESSION_LOG", "0")
    assert session_log.start() is None
