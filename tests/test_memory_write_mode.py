"""memory_write / memory_enrichment_source "none": short-term memory only."""

import json
import types

import pytest

pytestmark = [pytest.mark.unit]


def _settings(tmp_path, monkeypatch, cfg):
    from jarvis.config import load_settings
    f = tmp_path / "config.json"
    f.write_text(json.dumps(cfg))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(f))
    return load_settings()


def test_defaults_keep_long_term_memory(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch, {})
    assert s.memory_write == "all" and s.memory_enrichment_source == "all"


def test_short_term_only_is_parsed(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch, {"memory_write": "off", "memory_enrichment_source": "none"})
    assert s.memory_write == "off" and s.memory_enrichment_source == "none"


def test_unknown_values_fall_back(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch, {"memory_write": "maybe", "memory_enrichment_source": "x"})
    assert s.memory_write == "all" and s.memory_enrichment_source == "all"


def test_memory_write_off_saves_nothing():
    from jarvis.memory.conversation import update_diary_from_dialogue_memory

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"dialogue memory touched: {name}")
    cfg = types.SimpleNamespace(memory_write="off")
    assert update_diary_from_dialogue_memory(Boom(), Boom(), cfg, force=True) is None
