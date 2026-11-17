"""Fixes from a field session (RTX 2070, French): misheard wake name in the query, DuckDuckGo
bot challenge with no fallback, a slow digest of an empty search, a claimed-but-not-done action."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jarvis.listening.wake_detection import canonicalise_wake_name
from jarvis.tools.builtin import web_search
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("heard,expected", [
    ("Quel est le prix d'une manette de Xbox One Sarvis",
     "Quel est le prix d'une manette de Xbox One Jarvis"),
    ("Djarvis, allume la lumière", "Jarvis, allume la lumière"),
    ("jarvis ça va", "jarvis ça va"),               # already right: untouched
    ("Le service est ouvert", "Le service est ouvert"),
    ("Il a mis le mars au frigo", "Il a mis le mars au frigo"),
])
def test_misheard_wake_name_is_restored(heard, expected):
    assert canonicalise_wake_name(heard, "jarvis") == expected


def _run_blocked_search(monkeypatch, rows, metasearch=True):
    instant = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    monkeypatch.setattr(web_search.requests, "get", lambda url, **kw: instant)
    monkeypatch.setattr(web_search, "download_search_html",
                        lambda *a, **kw: (202, b'<div class="anomaly-modal"></div>'))
    monkeypatch.setattr(web_search, "_metasearch", lambda *a, **kw: rows)
    monkeypatch.setattr(web_search, "_cascade_fetch", lambda *a, **kw: None)
    cfg = SimpleNamespace(web_search_enabled=True, voice_debug=False, brave_search_api_key="",
                          wikipedia_fallback_enabled=False, web_metasearch_enabled=metasearch)
    return web_search.WebSearchTool().run({"search_query": "prix manette Xbox One"},
                                          SimpleNamespace(cfg=cfg, language="fr", user_print=Mock()))


def test_bot_challenge_falls_back_to_metasearch_snippets(monkeypatch):
    rows = [("Manette Xbox sans fil", "https://shop.example/manette", "Prix : 59,99 €, livraison gratuite")]
    result = _run_blocked_search(monkeypatch, rows)
    assert result.success
    assert "59,99" in result.reply_text                     # the snippet is the content
    assert "<<<BEGIN UNTRUSTED WEB EXTRACT>>>" in result.reply_text
    assert "blocked" not in result.reply_text.lower()
    assert result.skip_digest is False


def test_nothing_retrieved_skips_the_digest(monkeypatch):
    result = _run_blocked_search(monkeypatch, [])
    assert "blocked" in result.reply_text.lower()
    assert result.skip_digest is True


def test_metasearch_off_when_not_enabled(monkeypatch):
    called = []
    monkeypatch.setattr(web_search, "_metasearch", lambda *a, **kw: called.append(1) or [])
    instant = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    monkeypatch.setattr(web_search.requests, "get", lambda url, **kw: instant)
    monkeypatch.setattr(web_search, "download_search_html", lambda *a, **kw: (202, b"anomaly.js"))
    cfg = SimpleNamespace(web_search_enabled=True, voice_debug=False, brave_search_api_key="",
                          wikipedia_fallback_enabled=False)
    web_search.WebSearchTool().run({"search_query": "x"},
                                   SimpleNamespace(cfg=cfg, language="fr", user_print=Mock()))
    assert called == []


def test_engine_does_not_digest_an_empty_result(monkeypatch):
    from jarvis.reply import engine
    monkeypatch.setattr(engine, "_maybe_digest_tool_result",
                        lambda *a, **kw: pytest.fail("digest must be skipped"))
    result = ToolExecutionResult(success=True, reply_text="Search was blocked.", skip_digest=True)
    content, _ = engine._tool_result_content(SimpleNamespace(), "q", "webSearch", result)
    assert content == "Search was blocked."


def test_home_agent_never_claims_an_action_without_its_tool():
    from jarvis.system_prompt import PERSONA_STYLES
    assert "only when its tool ran in this turn" in PERSONA_STYLES["home_agent"]
