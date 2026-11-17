"""Local Kokoro-82M server, run inside its own Python environment.

Engine: kokoro-onnx (onnxruntime; espeak-ng phonemes bundled by espeakng-loader). Its own venv
because kokoro-onnx needs numpy 2 while Jarvis pins numpy < 2; the venv may also carry
onnxruntime-gpu without clashing with piper's CPU onnxruntime. Same protocol as
qwen3_tts_server.py, so Jarvis reuses its streaming player (adaptive buffer, barge-in):

    <venv>/python kokoro_tts_server.py --port 8096 --model kokoro-v1.0.onnx --voices voices-v1.0.bin
    POST /tts {"text": "..."}
      -> raw float32 mono PCM stream (X-Sample-Rate header), one chunk per sentence

The text is cut into sentences so the first one plays while the next are synthesised. This file
imports nothing from Jarvis.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_SENTENCE = re.compile(r"(?<=[.!?…;:])\s+")


def sentences(text: str, min_chars: int = 12) -> list[str]:
    """Sentences, very short ones merged into the next (Kokoro is weak below ~10 tokens)."""
    out, carry = [], ""
    for part in _SENTENCE.split(text.strip()):
        carry = f"{carry} {part}".strip() if carry else part.strip()
        if len(carry) >= min_chars:
            out.append(carry)
            carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--model", required=True, help="kokoro-v1.0.onnx (GPU) or kokoro-v1.0.int8.onnx (CPU)")
    ap.add_argument("--voices", required=True, help="voices-v1.0.bin")
    ap.add_argument("--voice", default="ff_siwis", help="French: ff_siwis (the only French voice)")
    ap.add_argument("--lang", default="fr-fr", help="espeak-ng language code")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if a.device != "cpu" and "CUDAExecutionProvider" in available:
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()          # CUDA/cuDNN from the nvidia-* wheels on Windows
            except Exception:
                pass
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif a.device == "cuda":
        print("CUDA requested but onnxruntime-gpu is missing: using CPU", flush=True)
    session = ort.InferenceSession(a.model, providers=providers)
    k = Kokoro.from_session(session, a.voices)
    print(f"providers: {session.get_providers()}", flush=True)
    t0 = time.time()
    k.create("Bonjour.", voice=a.voice, lang=a.lang, speed=a.speed)     # warm-up
    print(f"warm-up {time.time() - t0:.2f}s", flush=True)
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def _chunk(self, audio) -> None:
            data = np.asarray(audio, dtype=np.float32).reshape(-1).tobytes()
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            parts = sentences(str(req.get("text", ""))) or ["."]
            with lock:
                try:
                    first, sr = k.create(parts[0], voice=a.voice, lang=a.lang, speed=a.speed)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}".encode()
                    self.send_response(500)
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Sample-Rate", str(sr))
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    self._chunk(first)
                    for p in parts[1:]:
                        audio, _ = k.create(p, voice=a.voice, lang=a.lang, speed=a.speed)
                        self._chunk(audio)
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass             # barge-in: the client closed the connection

    print(f"ready on port {a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
