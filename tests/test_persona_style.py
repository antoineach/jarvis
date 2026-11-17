"""persona_style: the default butler is unchanged; home_agent is a concise spoken household agent."""

import json

import pytest

from jarvis.system_prompt import build_system_prompt

pytestmark = [pytest.mark.unit]


def test_default_is_the_butler():
    assert "British butler named Friday" in build_system_prompt("Friday")


def test_home_agent_persona_keeps_the_shared_rules():
    p = build_system_prompt("Jarvis", "home_agent")
    assert "runs this home" in p and "spoken aloud" in p
    assert "butler" not in p and "sarcastic:" not in p
    assert "[Context: ...]" in p                     # shared functional rules still present


def test_unknown_style_falls_back_to_butler():
    assert "British butler" in build_system_prompt("Jarvis", "pirate")


def test_config_parses_persona_style(tmp_path, monkeypatch):
    from jarvis.config import load_settings
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"persona_style": "home_agent"}))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(f))
    assert load_settings().persona_style == "home_agent"
    f.write_text(json.dumps({"persona_style": "nope"}))
    assert load_settings().persona_style == "butler"
