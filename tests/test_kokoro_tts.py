"""Kokoro engine: config, server command, missing files, sentence splitting."""

import importlib.util
import json
from pathlib import Path

import pytest

from jarvis.output.kokoro_tts import SERVER_SCRIPT, KokoroTTS
from jarvis.output.tts import create_tts_engine

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def test_factory_builds_kokoro():
    eng = create_tts_engine(engine="kokoro", kokoro={"python": "/x", "voice": "ff_siwis"})
    assert isinstance(eng, KokoroTTS) and eng.voice == "ff_siwis"


def test_missing_model_files_point_to_the_setup(tmp_path):
    py = tmp_path / "python"
    py.write_text("")
    ok, err = KokoroTTS(python=str(py), model=str(tmp_path / "nope.onnx"), voices=None)._prepare_speech()
    assert not ok and "tts_kokoro_model" in err and "--tts kokoro" in err


def test_server_command():
    k = KokoroTTS(python="/py", model="/m.onnx", voices="/v.bin", lang="fr-fr", device="cpu")
    cmd = k._server_cmd("/py", 9999)
    assert cmd[:2] == ["/py", str(SERVER_SCRIPT)] and "--voices" in cmd and cmd[cmd.index("--device") + 1] == "cpu"


def test_config_accepts_kokoro(tmp_path, monkeypatch):
    from jarvis.config import load_settings
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"tts_engine": "kokoro", "tts_kokoro_device": "gpu?"}))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(f))
    s = load_settings()
    assert s.tts_engine == "kokoro" and s.tts_kokoro_device == "auto" and s.tts_kokoro_voice == "ff_siwis"


def test_server_sentence_splitting_merges_short_pieces():
    spec = importlib.util.spec_from_file_location("kserver", SERVER_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.sentences("Oui. Il fait quinze degrés à Pessac. Pluie ce soir.") == [
        "Oui. Il fait quinze degrés à Pessac.", "Pluie ce soir."]
