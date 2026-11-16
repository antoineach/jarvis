"""Qwen3-TTS engine: streamed playback, adaptive buffering, interruption and configuration."""

import threading
import time

import numpy as np
import pytest

from jarvis.output import qwen3_tts
from jarvis.output.qwen3_tts import OUT_SR, Qwen3TTS, _resample
from jarvis.output.tts import SpeechRequest, create_tts_engine

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


class FakeOut:
    instances = []

    def __init__(self, mode, latency):
        self.samples, self.closed, self.aborted = [], False, False
        FakeOut.instances.append(self)

    def write(self, x):
        self.samples.append(np.asarray(x))
        time.sleep(len(x) / OUT_SR * 0.05)      # play 20x faster than real time

    def close(self, abort=False):
        self.closed, self.aborted = True, abort


def _engine(monkeypatch, chunks, delay=0.0, sr=24000):
    FakeOut.instances.clear()
    monkeypatch.setattr(qwen3_tts, "open_output", FakeOut)
    eng = Qwen3TTS(python="/unused", mode="expressive")

    def fake_stream(text):
        def gen():
            for c in chunks:
                time.sleep(delay)
                yield c
        return sr, gen()
    monkeypatch.setattr(eng, "_stream", fake_stream)
    monkeypatch.setattr(eng, "_begin_playback", lambda request, duration: True)
    return eng


def _request(text="Bonjour, je suis Jarvis."):
    return SpeechRequest(text, None, None, None, lambda r: None)


def test_streamed_chunks_are_all_played_in_order_at_output_rate(monkeypatch):
    chunks = [np.full(2400, i + 1, np.float32) for i in range(6)]
    eng = _engine(monkeypatch, chunks, delay=0.01)
    assert eng._speak_once(_request()) is True
    out = FakeOut.instances[0]
    audio = np.concatenate(out.samples)
    played = audio[audio != 0]                        # silence padding, if any, is not content
    assert len(played) == 6 * 2400 * 2               # 24 kHz resampled to 48 kHz
    assert list(dict.fromkeys(played.tolist())) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert out.closed and not out.aborted


def test_slow_generation_is_padded_with_silence_not_left_to_underrun(monkeypatch):
    # 0.1 s of audio every 0.2 s: generation at half real time
    chunks = [np.ones(2400, np.float32) for _ in range(12)]
    eng = _engine(monkeypatch, chunks, delay=0.2)
    monkeypatch.setattr(qwen3_tts, "_estimate_tts_duration", lambda text, wpm: 0.5)
    assert eng._speak_once(_request()) is True
    audio = np.concatenate(FakeOut.instances[0].samples)
    assert np.count_nonzero(audio) == 12 * 2400 * 2
    assert np.count_nonzero(audio == 0) > 0          # explicit silence written while waiting


def test_interrupt_stops_playback_and_aborts_device(monkeypatch):
    chunks = [np.ones(24000, np.float32) for _ in range(40)]
    eng = _engine(monkeypatch, chunks, delay=0.02)
    threading.Timer(0.5, eng._should_interrupt.set).start()
    assert eng._speak_once(_request("Une longue phrase. " * 20)) is False
    out = FakeOut.instances[0]
    assert out.aborted
    assert sum(len(s) for s in out.samples) < 40 * 24000 * 2


def test_missing_interpreter_reports_a_clear_error():
    eng = Qwen3TTS(python="/does/not/exist", mode="clone", audio_prompt_path="/nope.wav")
    ok, error = eng._prepare_speech()
    assert not ok and "tts_qwen3_python" in error


def test_factory_builds_qwen3_engine():
    eng = create_tts_engine(engine="qwen3", qwen3={"python": "/x", "mode": "expressive", "speaker": "ryan"})
    assert isinstance(eng, Qwen3TTS) and eng.speaker == "ryan"


def test_resample_doubles_24k_to_48k():
    assert len(_resample(np.ones(240, np.float32), 24000)) == 480


def test_config_keeps_qwen3_engine(tmp_path, monkeypatch):
    import json

    from jarvis.config import load_settings
    cfg_file = tmp_path / "jarvis" / "config.json"
    cfg_file.parent.mkdir()
    cfg_file.write_text(json.dumps({"tts_engine": "qwen3", "tts_qwen3_language": "French"}))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    s = load_settings()
    assert s.tts_engine == "qwen3" and s.tts_qwen3_language == "French"
