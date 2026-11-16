"""NVIDIA Parakeet / Canary speech recognition through onnx-asr (``whisper_backend: "parakeet"``).

Why: on French, Parakeet-TDT-0.6B-v3 has a third lower WER than Whisper large-v3 on short,
varied-microphone speech (CoVoST2 fr 6.05 vs 11.02, arXiv 2509.14128 table 13), is ~20x faster,
and its transducer decoder is far less prone to inventing text on noise than Whisper's
autoregressive decoder. Canary-1B-v2 (``stt_model: "nemo-canary-1b-v2"``) is slightly better on
read speech but autoregressive and twice the size.

onnx-asr runs the ONNX export on onnxruntime (CUDA when ``onnxruntime-gpu`` is installed, CPU
otherwise): no NeMo, no PyTorch, works on Windows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..debug import debug_log

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"
_LANGUAGE_MODELS = ("canary",)       # models that need the spoken language as a prompt token


def _providers(device: str) -> list[str]:
    import onnxruntime as ort
    available = ort.get_available_providers()
    if device != "cpu" and "CUDAExecutionProvider" in available:
        if hasattr(ort, "preload_dlls"):          # CUDA/cuDNN from the nvidia-* pip wheels (Windows)
            try:
                ort.preload_dlls()
            except Exception as e:
                debug_log(f"onnxruntime preload_dlls failed: {e}", "voice")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        print("  ⚠️  CUDA requested for speech recognition but onnxruntime-gpu is not installed: "
              "using CPU (pip install onnxruntime-gpu)", flush=True)
    return ["CPUExecutionProvider"]


class NemoSTT:
    """Small wrapper with the one call the listener needs: 16 kHz float32 audio -> text."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 quantization: Optional[str] = None, language: Optional[str] = None, loader=None):
        if loader is None:
            import onnx_asr
            loader = onnx_asr.load_model
        self.model_name = model or DEFAULT_MODEL
        self.language = language
        self.providers = _providers(device)
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
        self._model = loader(self.model_name, quantization=quantization, providers=self.providers)
        self._needs_language = any(k in self.model_name for k in _LANGUAGE_MODELS)
        debug_log(f"NeMo STT loaded: {self.model_name} on {self.device} (quantization={quantization})", "voice")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        kwargs = {}
        if self._needs_language:
            kwargs["language"] = self.language or "en"
        text = self._model.recognize(audio, sample_rate=sample_rate, **kwargs)
        return (text or "").strip()
