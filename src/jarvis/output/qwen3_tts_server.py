"""Local streaming Qwen3-TTS server, run inside its own Python environment.

Engine: faster-qwen3-tts (CUDA graphs + streaming generation). The first audio chunk is ready
after a few hundred milliseconds instead of waiting for the whole sentence. This file imports
nothing from Jarvis: it runs with the interpreter configured in ``tts_qwen3_python`` because
qwen-tts pins transformers versions that conflict with the rest of the stack.

    <venv>/bin/python qwen3_tts_server.py --port 8095 --mode clone --voice ref.wav
    POST /tts {"text": "...", "style": "calm"}
      -> raw float32 mono PCM stream (X-Sample-Rate header), chunk by chunk
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STYLES = {"calm": "Speak in a calm, soft and reassuring tone.",
          "happy": "Speak in a cheerful, happy and lively tone.",
          "worried": "Speak in a worried, urgent and tense tone.",
          "ironic": "Speak in a dry, ironic, slightly amused tone, like a witty British butler.",
          "neutral": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--mode", choices=["expressive", "clone"], default="clone")
    ap.add_argument("--size", default="0.6B")
    ap.add_argument("--voice", default="", help="reference WAV (clone mode)")
    ap.add_argument("--voice-text", default="", help="transcript of the reference (better cloning, slower)")
    ap.add_argument("--speaker", default="", help="built-in voice (expressive mode)")
    ap.add_argument("--language", default="Auto")
    ap.add_argument("--instruct", default="", help="default speaking style instruction")
    ap.add_argument("--backend", choices=["torch", "ggml"], default="torch",
                    help="ggml: qwentts.cpp C++ engine (pip install 'faster-qwen3-tts[ggml]'), experimental")
    ap.add_argument("--chunk", type=int, default=8,
                    help="codec steps per chunk (12 Hz: 8 = 667 ms). Larger = better throughput, later first audio")
    a = ap.parse_args()

    import threading

    import numpy as np
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    kind = "Base" if a.mode == "clone" else "CustomVoice"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # Turing has no bf16
    if a.backend == "ggml":
        m = FasterQwen3TTS.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{a.size}-{kind}", backend="ggml")
    else:
        m = FasterQwen3TTS.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{a.size}-{kind}", device="cuda", dtype=dtype)
    print(f"backend: {a.backend}", flush=True)
    speaker = a.speaker
    if a.mode == "expressive":
        inner = getattr(m, "model", None)
        names = []
        for attr in ("get_supported_speakers", "get_supported_spks"):
            if inner is not None and hasattr(inner, attr):
                names = list(getattr(inner, attr)())
                break
        speaker = speaker or ("ryan" if not names or "ryan" in names else names[0])
        print(f"voices: {', '.join(names) or '?'}; selected: {speaker}", flush=True)
    m.warmup(prefill_len=100)          # capture CUDA graphs once
    lock = threading.Lock()            # one stream at a time on the GPU

    def stream(text: str, style: str):
        instr = STYLES.get(style, "") or a.instruct or None
        if a.mode == "clone":
            return m.generate_voice_clone_streaming(text=text, language=a.language, ref_audio=a.voice,
                                                    ref_text=a.voice_text, xvec_only=not a.voice_text,
                                                    chunk_size=a.chunk)
        return m.generate_custom_voice_streaming(text=text, speaker=speaker, language=a.language, instruct=instr,
                                                 chunk_size=a.chunk)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                gen = stream(req["text"], req.get("style", "neutral"))
                try:
                    first = next(gen)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}".encode()
                    self.send_response(500)
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Sample-Rate", str(first[1]))
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for item in _chain(first, gen):
                        data = np.asarray(item[0], dtype=np.float32).reshape(-1).tobytes()
                        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass             # barge-in: the client closed the connection

    print(f"ready on port {a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


def _chain(first, gen):
    yield first
    yield from gen


if __name__ == "__main__":
    main()
