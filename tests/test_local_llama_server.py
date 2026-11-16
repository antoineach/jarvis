"""Jarvis starts the configured llama.cpp servers (fake llama-server here) and stops them on exit."""

import socket
import sys
import textwrap
import types

import pytest

from jarvis.llm import local_server

pytestmark = [pytest.mark.unit]

FAKE = textwrap.dedent('''\
    #!{py}
    import sys, time, http.server
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    if "--fail" in args:
        print("error: model is corrupted"); sys.exit(3)
    time.sleep(0.5)                                   # loading
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        def log_message(self, *a): pass
    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(local_server.Path, "home", lambda: tmp_path)
    exe = tmp_path / "llama-server"
    exe.write_text(FAKE.format(py=sys.executable))
    exe.chmod(0o755)
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"GGUF")
    yield exe, model
    local_server.stop_started_servers()


def _cfg(exe, model, port, **kw):
    return types.SimpleNamespace(llama_server_path=str(exe), llama_server_model=str(model),
                                 llm_base_url=f"http://127.0.0.1:{port}/v1", **kw)


@pytest.mark.skipif(sys.platform == "win32", reason="shebang fake executable")
def test_starts_chat_and_embedding_servers_then_stops_them(fake):
    exe, model = fake
    p1, p2 = _port(), _port()
    cfg = _cfg(exe, model, p1, llama_server_args=["-ngl", "0"], llama_server_embedding_model=str(model),
               embedding_base_url=f"http://127.0.0.1:{p2}/v1")
    local_server.ensure_local_servers(cfg)
    assert local_server._health(cfg.llm_base_url) == 200
    assert local_server._health(cfg.embedding_base_url) == 200
    procs = list(local_server._started)
    assert len(procs) == 2
    local_server.stop_started_servers()
    assert all(p.poll() is not None for p in procs)


@pytest.mark.skipif(sys.platform == "win32", reason="shebang fake executable")
def test_running_server_is_reused_not_restarted(fake, capsys):
    exe, model = fake
    p = _port()
    local_server.ensure_local_servers(_cfg(exe, model, p))
    first = list(local_server._started)
    local_server.ensure_local_servers(_cfg(exe, model, p))
    assert local_server._started == first and "already running" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform == "win32", reason="shebang fake executable")
def test_crash_reports_log_tail(fake, capsys):
    exe, model = fake
    local_server.ensure_local_servers(_cfg(exe, model, _port(), llama_server_args=["--fail"]))
    out = capsys.readouterr().out
    assert "exited with code 3" in out and "model is corrupted" in out


def test_nothing_happens_without_configuration():
    local_server.ensure_local_servers(types.SimpleNamespace(llama_server_path=None, llama_server_model=None,
                                                            llm_base_url="http://127.0.0.1:1/v1"))
    assert local_server._started == []


def test_remote_server_is_never_started(fake):
    exe, model = fake
    cfg = types.SimpleNamespace(llama_server_path=str(exe), llama_server_model=str(model),
                                llm_base_url="http://192.168.1.20:8080/v1")
    local_server.ensure_local_servers(cfg)
    assert local_server._started == []


def test_missing_model_is_reported(fake, capsys):
    exe, _ = fake
    local_server.ensure_local_servers(_cfg(exe, "/nope.gguf", _port()))
    assert "not found" in capsys.readouterr().out and local_server._started == []


def test_embedding_server_gets_no_gpu(fake, monkeypatch):
    exe, model = fake
    seen = {}
    real = local_server.subprocess.Popen

    def spy(cmd, **kw):
        seen[cmd[cmd.index("--port") + 1]] = (kw.get("env") or {}).get("CUDA_VISIBLE_DEVICES")
        return real(cmd, **kw)
    monkeypatch.setattr(local_server.subprocess, "Popen", spy)
    p1, p2 = _port(), _port()
    local_server.ensure_local_servers(_cfg(exe, model, p1, llama_server_embedding_model=str(model),
                                           embedding_base_url=f"http://127.0.0.1:{p2}/v1"))
    assert seen[str(p2)] == "-1" and seen[str(p1)] is None
