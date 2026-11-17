"""speculative_planning: a plan computed during intent judging is reused, never recomputed."""

import types

import pytest

from jarvis.reply import planner

pytestmark = [pytest.mark.unit]


@pytest.fixture
def cfg(monkeypatch):
    planner._spec.clear()
    monkeypatch.setattr(planner, "resolve_model", lambda cfg, tier: "m")
    return types.SimpleNamespace(planner_enabled=True, planner_timeout_sec=3.0)


def test_prefetched_plan_is_reused_for_identical_inputs(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(planner, "_call_planner_llm",
                        lambda *a: calls.append(a) or 'getWeather\nReply to the user.')
    tools = [("getWeather", "Current weather")]
    planner.prefetch_plan(cfg, "Quelle météo ce soir ?", "", tools)
    steps = planner.plan_query(cfg, "Quelle météo ce soir ?", "", tools)
    assert len(calls) == 1 and steps and steps[-1].lower().startswith("reply")


def test_different_inputs_recompute(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(planner, "_call_planner_llm", lambda *a: calls.append(a) or 'Reply to the user.')
    planner.prefetch_plan(cfg, "Quelle météo ce soir ?", "", [])
    planner.plan_query(cfg, "Et demain, quelle météo ?", "", [])
    assert len(calls) == 2


def test_prefetched_plan_is_used_once(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(planner, "_call_planner_llm", lambda *a: calls.append(a) or 'Reply to the user.')
    planner.prefetch_plan(cfg, "Quelle heure est-il ?", "", [])
    planner.plan_query(cfg, "Quelle heure est-il ?", "", [])
    planner.plan_query(cfg, "Quelle heure est-il ?", "", [])
    assert len(calls) == 2


def test_failed_prefetch_falls_back_to_a_live_call(cfg, monkeypatch):
    state = {"n": 0}

    def llm(*a):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("server busy")
        return 'Reply to the user.'
    monkeypatch.setattr(planner, "_call_planner_llm", llm)
    planner.prefetch_plan(cfg, "Quelle heure est-il ?", "", [])
    assert planner.plan_query(cfg, "Quelle heure est-il ?", "", []) == ["Reply to the user."]
    assert state["n"] == 2
