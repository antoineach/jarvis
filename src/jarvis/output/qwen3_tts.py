"""Qwen3-TTS speech output: voice cloning, streaming playback, local GPU.

The model runs in a separate process (``qwen3_tts_server.py``) using the Python interpreter set
in ``tts_qwen3_python``. qwen-tts / faster-qwen3-tts pin transformers versions that conflict with
the rest of Jarvis, so they live in their own environment; this module only speaks HTTP to it.

Playback starts before the whole sentence is synthesised. Because a mid-range GPU (e.g. RTX 2070)
generates audio at roughly real-time speed with no margin, an adaptive buffer measures generation
speed over the first ~0.8 s and waits just long enough to finish without gaps. If generation still
falls behind, silence is fed to the device instead of letting it underrun (which crackles).
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from ..debug import debug_log
from ..llm import timing as _llm_timing
from .tts import SpeechRequest, _QueuedSpeech, _call_speech_callback, _estimate_tts_duration

OUT_SR = 48000          # native rate of PulseAudio/WSLg and most devices: we resample ourselves
SERVER_SCRIPT = Path(__file__).with_name("qwen3_tts_server.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _resample(x: np.ndarray, sr: int, to: int = OUT_SR) -> np.ndarray:
    if sr == to or len(x) == 0:
        return x.astype(np.float32, copy=False)
    n = int(round(len(x) * to / sr))
    return np.interp(np.arange(n) * (sr / to), np.arange(len(x)), x).astype(np.float32)


class _SoundDeviceOut:
    def __init__(self, latency: float):
        import sounddevice as sd
        from ..utils.audio_lock import portaudio_lock
        self._lock = portaudio_lock
        with self._lock:
            self._s = sd.OutputStream(samplerate=OUT_SR, channels=1, dtype="float32", latency=latency)
            self._s.start()

    def write(self, x: np.ndarray) -> None:
        self._s.write(x.reshape(-1, 1))

    def close(self, abort: bool = False) -> None:
        with self._lock:
            (self._s.abort if abort else self._s.stop)()
            self._s.close()


class _PacatOut:
    """Straight to PulseAudio, bypassing the ALSA plugin that crackles under WSLg."""

    def __init__(self, latency: float):
        self._p = subprocess.Popen(["pacat", "--playback", "--raw", "--format=float32le", f"--rate={OUT_SR}",
                                    "--channels=1", f"--latency-msec={int(latency * 1000)}", "--client-name=jarvis"],
                                   stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, x: np.ndarray) -> None:
        try:
            self._p.stdin.write(np.asarray(x, np.float32).tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def close(self, abort: bool = False) -> None:
        if abort:
            self._p.kill()
            return
        try:
            self._p.stdin.close()
        except Exception:
            pass
        self._p.wait()


def open_output(mode: str, latency: float):
    """``auto`` uses pacat under WSL when available, PortAudio otherwise."""
    if mode == "pulse" or (mode == "auto" and _is_wsl() and shutil.which("pacat")):
        return _PacatOut(latency)
    return _SoundDeviceOut(latency)


class Qwen3TTS(_QueuedSpeech):
    """Streaming Qwen3-TTS engine (``tts_engine: "qwen3"``)."""

    def __init__(self, enabled: bool = True, rate: Optional[int] = None, python: Optional[str] = None,
                 mode: str = "clone", audio_prompt_path: Optional[str] = None, speaker: Optional[str] = None,
                 size: str = "0.6B", language: str = "Auto", instruct: str = "", backend: str = "torch",
                 output: str = "auto", output_latency: float = 0.25, chunk: int = 8) -> None:
        self.enabled, self.rate = enabled, rate
        self.python = python
        self.mode, self.audio_prompt_path, self.speaker = mode, audio_prompt_path, speaker
        self.size, self.language, self.instruct, self.backend = size, language, instruct, backend
        self.output, self.output_latency = output, output_latency
        self.chunk = int(chunk)
        super().__init__()
        self._proc: Optional[subprocess.Popen] = None
        self._port = 0
        self._init_error: Optional[str] = None
        self._init_lock = threading.Lock()
        self._log_path = Path.home() / ".local" / "share" / "jarvis" / "qwen3_tts.log"

    # -- server lifecycle -------------------------------------------------------

    def _prepare_speech(self) -> tuple[bool, Optional[str]]:
        with self._init_lock:
            if self._proc is not None and self._proc.poll() is None:
                return True, None
            return self._start_server(), self._init_error

    # Hooks overridden by engines that reuse this streaming client with another server script.
    _label = "Qwen3-TTS"
    _python_setting = "tts_qwen3_python"
    _start_note = " (first start captures CUDA graphs)"

    def _input_error(self) -> Optional[str]:
        if self.mode == "clone" and not (self.audio_prompt_path and Path(self.audio_prompt_path).exists()):
            return f"clone mode needs tts_qwen3_audio_prompt (got {self.audio_prompt_path!r})"
        return None

    def _server_cmd(self, py: str, port: int) -> list:
        cmd = [py, str(SERVER_SCRIPT), "--port", str(port), "--mode", self.mode, "--size", self.size,
               "--language", self.language, "--backend", self.backend, "--chunk", str(self.chunk)]
        if self.mode == "clone":
            cmd += ["--voice", str(self.audio_prompt_path)]
        if self.speaker:
            cmd += ["--speaker", self.speaker]
        if self.instruct:
            cmd += ["--instruct", self.instruct]
        return cmd

    def _start_server(self) -> bool:
        py = self.python or ""
        if not py or not Path(py).exists():
            self._init_error = (f"{self._python_setting} must point to the Python interpreter of the "
                                f"{self._label} environment (got {py!r})")
            return False
        err = self._input_error()
        if err:
            self._init_error = err
            return False
        self._port = _free_port()
        cmd = self._server_cmd(py, self._port)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🔧 [TTS] Starting {self._label} server{self._start_note}...", flush=True)
        self._proc = subprocess.Popen(cmd, stdout=open(self._log_path, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self._proc.poll() is not None:
                tail = "\n".join(self._log_path.read_text(errors="replace").splitlines()[-15:])
                self._init_error = f"{self._label} server exited:\n{tail}"
                return False
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self._port}/", timeout=2)
                debug_log(f"{self._label} server ready on port {self._port} after {time.time() - t0:.1f}s", "tts")
                print(f"✅ [TTS] {self._label} ready", flush=True)
                return True
            except Exception:
                time.sleep(1)
        self._init_error = f"{self._label} server did not become ready within 600 s"
        return False

    def stop(self) -> None:
        super().stop()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # -- synthesis --------------------------------------------------------------

    def _stream(self, text: str):
        req = urllib.request.Request(f"http://127.0.0.1:{self._port}/tts",
                                     data=json.dumps({"text": text, "style": "neutral"}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Qwen3-TTS: {e.read().decode('utf-8', 'replace')[:300]}") from None
        sr = int(r.headers["X-Sample-Rate"])

        def chunks():
            buf = b""
            with r:
                while True:
                    data = r.read1(65536)
                    if not data:
                        break
                    buf += data
                    n = len(buf) // 4 * 4
                    if n:
                        yield np.frombuffer(buf[:n], dtype=np.float32)
                        buf = buf[n:]
        return sr, chunks()

    def interrupt(self) -> None:
        self._interrupt_active_request()

    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text

    def _speak_once(self, request: SpeechRequest) -> bool:
        with self._lifecycle_lock:
            if self._stop.is_set() or request.abandoned:
                return False
            self._should_interrupt.clear()
        t0 = time.time()
        try:
            sr, it = self._stream(request.text)
        except Exception as e:
            debug_log(f"Qwen3-TTS request failed: {e}", "tts")
            print(f"  ⚠️ [TTS] {e}", flush=True)
            return False
        q: queue.Queue = queue.Queue()
        stats = {"gen": 0, "done": False, "t_first": None}

        def producer():
            try:
                for c in it:
                    if self._should_interrupt.is_set() or self._stop.is_set():
                        break
                    if stats["t_first"] is None:
                        stats["t_first"] = time.time()
                    stats["gen"] += len(c)
                    q.put(c)
            except Exception as e:
                debug_log(f"Qwen3-TTS stream error: {e}", "tts")
            finally:
                stats["done"] = True
                q.put(None)
        threading.Thread(target=producer, daemon=True).start()

        first = q.get()
        if first is None:
            return False
        pending, first_s = [first], len(first) / sr
        est_total = _estimate_tts_duration(request.text, self.rate or 160)   # cloned voices speak ~160 WPM
        # Adaptive buffer: wait until the audio already generated covers the remaining generation time.
        while not stats["done"] and not self._should_interrupt.is_set():
            el = time.time() - stats["t_first"]
            gen_s = stats["gen"] / sr
            # Speed is measured over 1.2 chunks (0.5 s minimum): 0.8 s with the default 667 ms
            # chunks, as before; smaller chunks give a valid estimate, and playback, sooner.
            if el >= max(0.5, 1.2 * self.chunk / 12.0):
                speed = 0.8 * (gen_s - first_s) / el          # pessimistic: generation slows down later
                need = 0.5 if speed >= 1.4 else max(0.8, (est_total - gen_s) * (1 - speed) / max(speed, 0.3) + 0.6)
                if gen_s >= need:
                    break
            try:
                c = q.get(timeout=0.05)
            except queue.Empty:
                continue
            if c is None:
                # End of stream reached while buffering (fast engines finish before playback starts):
                # hand the marker to the playback loop, which would otherwise wait for it forever
                # and write silence (measured with Kokoro: the reply after the first one never spoke).
                pending.append(None)
                break
            pending.append(c)
        debug_log(f"Qwen3-TTS first audio after {time.time() - t0:.2f}s, buffered {stats['gen'] / sr:.1f}s", "tts")
        if _llm_timing.enabled():
            print(f"  ⏱️ tts start     {time.time() - t0:5.2f}s  (buffered {stats['gen'] / sr:.1f}s)", flush=True)

        if request.duration_callback is not None:
            _call_speech_callback(request.duration_callback, (est_total,), "duration")
        if not self._begin_playback(request, est_total):
            return False
        out = open_output(self.output, self.output_latency)
        interrupted, total = False, 0
        try:
            while True:
                if self._should_interrupt.is_set() or self._stop.is_set():
                    interrupted = True
                    break
                if pending:
                    c = pending.pop(0)
                else:
                    try:
                        c = q.get(timeout=0.03)
                    except queue.Empty:
                        out.write(np.zeros(int(0.05 * OUT_SR), np.float32))   # clean pause, no underrun
                        continue
                if c is None:
                    break
                out.write(_resample(c, sr))
                total += len(c)
        finally:
            out.close(abort=interrupted)
            self._is_speaking.clear()
            if interrupted and hasattr(it, "close"):
                try:
                    it.close()
                except Exception:
                    pass
        debug_log(f"Qwen3-TTS played {total / sr:.1f}s, done {time.time() - t0:.1f}s after request", "tts")
        return not interrupted
