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
    assert out["whisper_backend"] == "parakeet" and out["stt_quantization"] == "int8"
    assert (tmp_path / "config.json.bak").exists()


class _FakeAsr:
    def __init__(self):
        self.calls = []

    def recognize(self, audio, sample_rate=16000, **kw):
        self.calls.append((len(audio), sample_rate, kw))
        return "  Quelle heure est-il ?  "


def _nemo(model, language=None):
    from jarvis.listening.nemo_stt import NemoSTT
    fake = _FakeAsr()
    stt = NemoSTT(model, device="cpu", language=language, loader=lambda name, **kw: fake)
    return stt, fake


def test_parakeet_transcribes_without_language_prompt():
    import numpy as np
    stt, fake = _nemo("nemo-parakeet-tdt-0.6b-v3", language="fr")
    assert stt.transcribe(np.zeros(16000, np.float32)) == "Quelle heure est-il ?"
    assert fake.calls == [(16000, 16000, {})] and stt.device == "cpu"


def test_canary_gets_the_configured_language():
    import numpy as np
    stt, fake = _nemo("nemo-canary-1b-v2", language="fr")
    stt.transcribe(np.zeros(8000, np.float32))
    assert fake.calls[0][2] == {"language": "fr"}


def test_empty_audio_is_not_sent_to_the_model():
    import numpy as np
    stt, fake = _nemo("nemo-parakeet-tdt-0.6b-v3")
    assert stt.transcribe(np.zeros(0, np.float32)) == "" and fake.calls == []


def test_parakeet_backend_is_accepted_by_config(tmp_path, monkeypatch):
    s = _load(tmp_path, monkeypatch, {"whisper_backend": "parakeet", "stt_model": "nemo-canary-1b-v2"})
    assert s.whisper_backend == "parakeet" and s.stt_model == "nemo-canary-1b-v2"


def test_listener_routes_transcription_to_parakeet():
    import threading
    import types

    import numpy as np
    from jarvis.listening.listener import VoiceListener

    stt, fake = _nemo("nemo-parakeet-tdt-0.6b-v3")
    me = types.SimpleNamespace(_whisper_backend="parakeet", _nemo_stt=stt, _samplerate=16000,
                               transcribe_lock=threading.Lock(),
                               cfg=types.SimpleNamespace(whisper_language="fr"))
    text, lang, events = VoiceListener._transcribe_audio(me, np.zeros(16000, np.float32))
    assert (text, lang, events) == ("Quelle heure est-il ?", "fr", ())
