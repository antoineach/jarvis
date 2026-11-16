"""Omni-FR setup: forced Whisper language, latency report, French preset script."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jarvis.llm import timing

pytestmark = [pytest.mark.unit]
ROOT = Path(__file__).resolve().parents[1]


def _load(tmp_path, monkeypatch, cfg):
    from jarvis.config import load_settings
    f = tmp_path / "config.json"
    f.write_text(json.dumps(cfg))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(f))
    return load_settings()


def test_whisper_language_defaults_to_autodetect(tmp_path, monkeypatch):
    assert _load(tmp_path, monkeypatch, {}).whisper_language is None


def test_whisper_language_is_normalised(tmp_path, monkeypatch):
    assert _load(tmp_path, monkeypatch, {"whisper_language": "FR-fr"}).whisper_language == "fr"


def test_timed_is_silent_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv("JARVIS_TIMINGS", raising=False)
    assert timing.timed("chat")(lambda: {"x": 1})() == {"x": 1}
    assert capsys.readouterr().out == ""


def test_timed_reports_caller_and_tokens(monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_TIMINGS", "1")
    fake = timing.timed("chat")(lambda: {"usage": {"prompt_tokens": 812, "completion_tokens": 41}})
    fake()
    out = capsys.readouterr().out
    assert "llm.chat" in out and "812→41 tok" in out


def test_timed_marks_failed_calls(monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_TIMINGS", "1")
    timing.timed("chat")(lambda: None)()
    assert "(failed)" in capsys.readouterr().out


@pytest.mark.usefixtures("no_network")
def test_preset_script_merges_and_keeps_user_keys(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"llm_base_url": "http://127.0.0.1:8080/v1", "mcps": {"ha": {}},
                               "tts_qwen3_python": "/env/python"}))
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"RIFF")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "configure_omni_fr.py"), "--voice", str(voice)],
                       env={"JARVIS_CONFIG_PATH": str(cfg), "PATH": ""}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(cfg.read_text())
    assert out["llm_base_url"] == "http://127.0.0.1:8080/v1" and out["mcps"] == {"ha": {}}
    assert out["tts_engine"] == "qwen3" and out["tts_qwen3_mode"] == "clone"
    assert out["tts_qwen3_audio_prompt"] == str(voice.resolve())
    assert out["whisper_language"] == "fr" and out["planner_enabled"] is False
    assert out["tool_selection_strategy"] == "llm"          # MCP tools present: keep the router
    assert (tmp_path / "config.json.bak").exists()
