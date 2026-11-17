"""Kokoro-82M speech output (``tts_engine: "kokoro"``): very fast, one fixed voice per language.

Trade-off against Qwen3-TTS, from the Kokoro voice notes: no cloning and no emotion control, and
French has a single voice (ff_siwis, graded B-, under 11 h of training audio) that is weaker on
very short utterances. In exchange: 82M parameters, no autoregressive codec, several times faster
than real time on a GPU (~0.3 GB) and around real time or better on a desktop CPU.

The model runs in its own environment (``tts_kokoro_python``: kokoro-onnx needs numpy 2) through
``kokoro_tts_server.py``, which speaks the same streaming protocol as the Qwen3 server, so the
whole playback path (adaptive buffer, silence padding, barge-in) is shared with Qwen3TTS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .qwen3_tts import Qwen3TTS

SERVER_SCRIPT = Path(__file__).with_name("kokoro_tts_server.py")


class KokoroTTS(Qwen3TTS):
    _label = "Kokoro"
    _python_setting = "tts_kokoro_python"
    _start_note = ""

    def __init__(self, enabled: bool = True, rate: Optional[int] = None, python: Optional[str] = None,
                 model: Optional[str] = None, voices: Optional[str] = None, voice: str = "ff_siwis",
                 lang: str = "fr-fr", speed: float = 1.0, device: str = "auto",
                 output: str = "auto", output_latency: float = 0.25) -> None:
        super().__init__(enabled=enabled, rate=rate, python=python, mode="kokoro", output=output,
                         output_latency=output_latency, chunk=5)    # shortest speed window (0.5 s)
        self.model, self.voices, self.voice = model, voices, voice
        self.lang, self.speed, self.device = lang, speed, device
        self._log_path = self._log_path.with_name("kokoro_tts.log")

    def _input_error(self) -> Optional[str]:
        for label, path in (("tts_kokoro_model", self.model), ("tts_kokoro_voices", self.voices)):
            if not (path and Path(path).exists()):
                return f"{label} not found (got {path!r}): run scripts/configure_omni_fr.py --tts kokoro"
        return None

    def _server_cmd(self, py: str, port: int) -> list:
        return [py, str(SERVER_SCRIPT), "--port", str(port), "--model", str(self.model),
                "--voices", str(self.voices), "--voice", self.voice, "--lang", self.lang,
                "--speed", str(self.speed), "--device", self.device]
