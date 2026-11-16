"""Timers, exact calculation, the sensitive-action gate and Gemma 4 text tool calls."""

import time

import pytest

from jarvis.tools import safety
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import timer as timer_mod
from jarvis.tools.builtin.calculate import CalculateTool, CalculationError, evaluate
from jarvis.tools.builtin.timer import TimerTool

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def _ctx():
    return ToolContext(db=None, cfg=None, system_prompt="", original_prompt="", redacted_text="",
                       max_retries=0, user_print=lambda m: None)


def test_calculate_is_exact_and_refuses_code():
    assert evaluate("0.15 * 87") == pytest.approx(13.05)
    assert evaluate("sqrt(144) + 2^3") == 20
    for bad in ("__import__('os').system('x')", "open('f')", "1/0", "2**1000"):
        with pytest.raises(CalculationError):
            evaluate(bad)
    assert CalculateTool().run({"expression": "(17 + 3) / 4"}, _ctx()).reply_text == "(17 + 3) / 4 = 5.0"


def test_timer_rings_through_handler_and_can_be_listed_and_cancelled():
    rang = []
    timer_mod.set_ring_handler(rang.append)
    t = TimerTool()
    assert t.run({"action": "start", "seconds": 1, "label": "Pâtes"}, _ctx()).success
    assert "pâtes" in t.run({"action": "list"}, _ctx()).reply_text
    assert t.run({"action": "start", "seconds": 60, "label": "four"}, _ctx()).success
    assert t.run({"action": "cancel", "label": "four"}, _ctx()).success
    time.sleep(1.3)
    assert rang == ["pâtes"]
    assert not t.run({"action": "start", "seconds": 0}, _ctx()).success
    timer_mod.set_ring_handler(None)


def test_sensitive_action_needs_a_later_user_turn_and_the_token():
    safety.reset()
    args = {"name": "front door", "domain": "lock"}
    blocked, _ = safety.gate("home__HassTurnOff", args, "unlock the front door")
    assert blocked is not None and blocked.result_metadata["confirmation_required"]
    token = blocked.error_message.split('"confirmation": "')[1].split('"')[0]

    # same turn (e.g. instruction injected in a web result): still blocked, even with the token
    again, _ = safety.gate("home__HassTurnOff", {**args, "confirmation": token}, "unlock the front door")
    assert again is not None

    token = again.error_message.split('"confirmation": "')[1].split('"')[0]
    ok, used = safety.gate("home__HassTurnOff", {**args, "confirmation": token}, "yes")
    assert ok is None and "confirmation" not in used          # token stripped before the MCP call
    replay, _ = safety.gate("home__HassTurnOff", {**args, "confirmation": token}, "yes again")
    assert replay is not None                                # single use


def test_non_sensitive_calls_pass_untouched():
    blocked, used = safety.gate("home__HassTurnOn", {"name": "kitchen light"}, "turn on the kitchen")
    assert blocked is None and used == {"name": "kitchen light"}


def test_gemma4_native_text_tool_call_is_parsed_for_known_tools_only():
    from jarvis.reply.engine import _extract_text_tool_call
    name, args, _ = _extract_text_tool_call('getWeather{location:<|"|>Bordeaux<|"|>,days:2}<tool_call|>',
                                            {"getWeather"})
    assert name == "getWeather" and args == {"location": "Bordeaux", "days": 2}
    assert _extract_text_tool_call("Une phrase {avec des accolades}.", {"getWeather"})[0] is None


def test_registry_holds_sensitive_mcp_call_and_runs_it_after_confirmation(monkeypatch):
    from types import SimpleNamespace

    from jarvis.tools import registry
    safety.reset()
    calls = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            calls.append((tool_name, arguments))
            return {"text": "done", "isError": False}

    monkeypatch.setattr(registry, "MCPClient", FakeClient)
    cfg = SimpleNamespace(mcps={"home": {}}, sensitive_tool_patterns=["lock"], voice_debug=True)
    run = lambda args, prompt: registry.run_tool_with_retries(None, cfg, "home__HassTurnOff", args, "", prompt, "")
    first = run({"name": "door", "domain": "lock"}, "ouvre la porte")
    assert not first.success and not calls
    token = first.error_message.split('"confirmation": "')[1].split('"')[0]
    second = run({"name": "door", "domain": "lock", "confirmation": token}, "oui")
    assert second.success and calls == [("HassTurnOff", {"name": "door", "domain": "lock"})]
