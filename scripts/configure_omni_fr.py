#!/usr/bin/env python3
"""Write the Omni-FR preset into the Jarvis config (French voice assistant on one 8 GB GPU).

Merges into the existing config file; keys you already set for the LLM server, MCP servers or
anything not listed below are left alone. Prints every key it changes. Re-run it any time.

    python scripts/configure_omni_fr.py --voice C:\\Users\\me\\Music\\voice.wav
    python scripts/configure_omni_fr.py --speaker vivian        # built-in voice, no cloning
    python scripts/configure_omni_fr.py --dry-run               # show changes, write nothing

Only clone a voice you have the owner's consent for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jarvis.config import get_config_path  # noqa: E402

TTS_PYTHON_CANDIDATES = [
    r"C:\qwen3-tts-env\Scripts\python.exe",
    "~/qwen3-tts-env/Scripts/python.exe",
    "~/qwen3-tts-env/bin/python",
    "~/.omnifr/venv-voix/bin/python",
]

# French mis-transcriptions of "Jarvis" seen with Whisper set to French.
FR_WAKE_ALIASES = ["djarvis", "jarvisse", "jarvice", "charvis", "jar vis", "jarvi"]
FR_STOP = ["stop", "arrête", "arrête-toi", "tais-toi", "silence", "chut", "ça suffit"]


def find_tts_python() -> str | None:
    for c in TTS_PYTHON_CANDIDATES:
        p = Path(os.path.expanduser(c))
        if p.exists():
            return str(p)
    return None


def preset(args, current: dict) -> dict:
    p: dict = {
        # Speech recognition: French forced (no per-utterance detection, no misdetected
        # short commands), large-v3-turbo: better French WER than medium, 4-layer decoder.
        "whisper_language": "fr",
        "whisper_model": args.whisper_model,
        "whisper_device": "cuda",
        "whisper_compute_type": "int8_float16",
        # Speech output: Qwen3-TTS streamed, French.
        "tts_engine": "qwen3",
        "tts_qwen3_language": "French",
        "tts_qwen3_size": "0.6B",
        # Fewer model calls per turn. The task-list planner costs one full call and only helps
        # multi-step requests. The LLM tool router costs another call; with only the built-in
        # tools, sending all of them is cheaper than choosing. "keyword" is NOT an option for
        # French: it matches query words against English tool descriptions.
        "planner_enabled": False,
        "tool_selection_strategy": "llm" if current.get("mcps") else "all",
        # French household wording.
        "timer_announcement": "{label} : c'est terminé !",
        "stop_commands": FR_STOP,
        "wake_aliases": sorted(set(current.get("wake_aliases", [])) | set(FR_WAKE_ALIASES)),
        "location_auto_detect": True,
    }
    py = args.tts_python or current.get("tts_qwen3_python") or find_tts_python()
    if py:
        p["tts_qwen3_python"] = py
    if args.voice:
        voice = Path(os.path.expanduser(args.voice)).resolve()
        if not voice.exists():
            sys.exit(f"voice file not found: {voice}")
        p.update({"tts_qwen3_mode": "clone", "tts_qwen3_audio_prompt": str(voice)})
    elif args.speaker:
        p.update({"tts_qwen3_mode": "expressive", "tts_qwen3_speaker": args.speaker})
    elif not current.get("tts_qwen3_audio_prompt"):
        p.update({"tts_qwen3_mode": "expressive", "tts_qwen3_speaker": "vivian"})
    if args.llm_url:
        p.update({"llm_provider": "openai_compatible", "llm_base_url": args.llm_url})
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", help="audio file of the voice to clone (wav/mp3/flac, 5-15 s, one speaker)")
    ap.add_argument("--speaker", help="built-in Qwen3 voice instead of cloning (e.g. vivian, serena, ryan)")
    ap.add_argument("--tts-python", help="python of the Qwen3-TTS environment (auto-detected)")
    ap.add_argument("--llm-url", help="OpenAI-compatible server, e.g. http://127.0.0.1:8080/v1 (llama-server)")
    ap.add_argument("--whisper-model", default="large-v3-turbo",
                    help="large-v3-turbo (default), medium, small (less VRAM)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(get_config_path())
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    new = preset(args, current)
    changed = {k: v for k, v in new.items() if current.get(k) != v}

    print(f"Config: {path}")
    for k, v in changed.items():
        print(f"  {k}: {json.dumps(current.get(k), ensure_ascii=False)} -> {json.dumps(v, ensure_ascii=False)}")
    if not changed:
        print("  already up to date")
    if "tts_qwen3_python" not in new:
        print("  ! Qwen3-TTS environment not found: pass --tts-python <path to its python>")
    if not (current.get("llm_base_url") or args.llm_url):
        print("  ! no llm_base_url: pass --llm-url http://127.0.0.1:8080/v1 if you use llama-server")
    if args.dry_run or not changed:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.with_suffix(".json.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps({**current, **new}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved (previous version: config.json.bak). Restart Jarvis.")


if __name__ == "__main__":
    main()
