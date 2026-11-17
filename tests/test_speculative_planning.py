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


def test_single_misnamed_argument_is_remapped_without_the_resolver():
    schemas = {"webSearch": {"type": "object", "properties": {"search_query": {"type": "string"}},
                             "required": ["search_query"]}}
    got = planner._parse_plan_step_concrete("webSearch query='suite de Fibonacci'", ["webSearch"], schemas)
    assert got == ("webSearch", {"search_query": "suite de Fibonacci"})


def test_ambiguous_remap_still_goes_to_the_resolver():
    schemas = {"t": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}}
    assert planner._parse_plan_step_concrete("t q='x'", ["t"], schemas) is None


@pytest.mark.parametrize("text,expected", [
    ("jarvis, combien font six fois deux ?", True),
    ("salut jarvis, comment ça va ?", True),
    ("hé djarvis, quelle heure est-il ?", True),
    ("j'ai demandé tout à l'heure à jarvis la météo", False),
    ("qu'est-ce qu'on mange ce soir ?", False),
])
def test_leading_wake_word_skips_the_judge(text, expected):
    from jarvis.listening.listener import VoiceListener
    me = types.SimpleNamespace(cfg=types.SimpleNamespace(wake_word_skips_judge=True, wake_word="jarvis",
                                                         wake_aliases=["djarvis"], wake_fuzzy_ratio=0.78))
    assert VoiceListener._leading_wake_word(me, text, False) is expected


def test_judge_kept_while_speaking_or_when_disabled():
    from jarvis.listening.listener import VoiceListener
    on = types.SimpleNamespace(cfg=types.SimpleNamespace(wake_word_skips_judge=True, wake_word="jarvis",
                                                         wake_aliases=[], wake_fuzzy_ratio=0.78))
    off = types.SimpleNamespace(cfg=types.SimpleNamespace(wake_word_skips_judge=False))
    assert VoiceListener._leading_wake_word(on, "jarvis, stop", True) is False
    assert VoiceListener._leading_wake_word(off, "jarvis, quelle heure ?", False) is False


def test_judge_warms_both_mode_prompts(monkeypatch):
    from jarvis.listening import intent_judge as ij
    sent = []

    class B:
        def chat(self, model, messages, **kw):
            sent.append((messages[0]["content"], kw["extra_options"]["max_tokens"]))
            return {"message": {"content": "{}"}}
    monkeypatch.setattr(ij, "get_llm_backend", lambda cfg: B())
    j = ij.IntentJudge.__new__(ij.IntentJudge)
    j.config = types.SimpleNamespace(cfg=None, model="m", timeout_sec=6.0)
    j._build_system_prompt = lambda hot: f"judge prompt hot={hot}"
    j.warm_prompts()
    assert sent == [("judge prompt hot=False", 1), ("judge prompt hot=True", 1)]


def test_early_plan_matches_despite_a_different_transcript_context(cfg, monkeypatch):
    """The early plan cannot know the final transcript timestamps: matched on query text."""
    calls = []
    monkeypatch.setattr(planner, "_call_planner_llm", lambda *a: calls.append(a) or "Reply to the user.")
    planner.prefetch_plan(cfg, "Salut Jarvis.", "", [], transcript_context='{"end": 1.0}', loose=True)
    assert planner.plan_query(cfg, "Salut Jarvis.", "", [], transcript_context='{"end": 1.4}') == \
        ["Reply to the user."]
    assert len(calls) == 1


def test_early_plan_for_other_words_is_not_used(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(planner, "_call_planner_llm", lambda *a: calls.append(a) or "Reply to the user.")
    planner.prefetch_plan(cfg, "Salut Jarvis", "", [], loose=True)        # user then kept talking
    planner.plan_query(cfg, "Salut Jarvis, quelle heure est-il ?", "", [])
    assert len(calls) == 2


def test_early_plan_transcribes_the_audio_so_far_and_prefetches(monkeypatch):
    import threading as th

    import numpy as np
    from jarvis.listening.listener import VoiceListener
    started = []
    me = types.SimpleNamespace(
        _early_plan_busy=False, tts=None, _utterance_frames=[np.zeros(320, np.float32)] * 40,
        _samplerate=16000, _stream_samplerate=16000,
        state_manager=types.SimpleNamespace(is_hot_window_active=lambda: False),
        _transcribe_audio=lambda audio: ("Jarvis, quelle heure est-il ?", "fr", ()),
        _leading_wake_word=lambda text, speaking: True, _canonical_query=lambda text: text,
        _start_plan_prefetch=lambda q, lang, loose=False: started.append((q, lang, loose)))
    VoiceListener._start_early_plan(me, 20)
    for t in th.enumerate():
        if t.name == "early-plan":
            t.join(2)
    assert started == [("Jarvis, quelle heure est-il ?", "fr", True)] and me._early_plan_busy is False
