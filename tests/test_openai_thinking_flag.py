"""thinking=False reaches local reasoning models (llama-server / vLLM chat_template_kwargs)."""

import pytest

from jarvis.llm.openai_compatible import OpenAICompatibleBackend

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("url,expected", [
    ("http://127.0.0.1:8080/v1", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("http://localhost:8088/v1", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("https://api.openai.com/v1", {}),
])
def test_thinking_switch_only_for_local_servers(url, expected):
    assert OpenAICompatibleBackend(url)._thinking_fields(False) == expected


def test_direct_sends_the_switch(monkeypatch):
    sent = {}

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    def post(url, json=None, **kw):
        sent.update(json)
        return R()
    from jarvis.llm import openai_compatible
    monkeypatch.setattr(openai_compatible.http, "post", post)
    assert OpenAICompatibleBackend("http://127.0.0.1:8080/v1").direct("m", "sys", "hi") == "ok"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
