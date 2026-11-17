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
import glob
import json
import os
import shutil
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


LLAMA_GLOBS = ["~/llama*/**/llama-server*", "~/Downloads/llama*/**/llama-server*", "C:/llama*/**/llama-server*",
               "C:/modeles/**/llama-server*", "C:/models/**/llama-server*",
               "~/.omnifr/**/llama-server*", "~/Downloads/**/llama-server.exe"]
GGUF_DIRS = ["~/models", "~/Downloads", "~/.omnifr", "~/.lmstudio/models", "C:/models", "C:/modeles",
             "~/.local/share/jarvis/models", "~/.cache/huggingface/hub"]
LLM_REPO = "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF"
EMBED_REPOS = ["ggml-org/embeddinggemma-300M-GGUF", "nomic-ai/nomic-embed-text-v1.5-GGUF"]


def _exists(p: str) -> bool:
    return Path(os.path.expanduser(p)).is_file()


def find_llama_server() -> str | None:
    hit = shutil.which("llama-server")
    if hit:
        return hit
    for g in LLAMA_GLOBS:
        for f in sorted(glob.glob(os.path.expanduser(g), recursive=True)):
            if Path(f).is_file() and Path(f).suffix.lower() in ("", ".exe"):
                return f
    return None


def list_ggufs() -> list[str]:
    out = []
    for d in GGUF_DIRS:
        out += glob.glob(os.path.join(os.path.expanduser(d), "**", "*.gguf"), recursive=True)
    return sorted(set(out))


def _is_aux(f: str) -> bool:
    name = Path(f).name.lower()
    return "mmproj" in name or "embed" in name or ("-of-" in name and "-00001-of-" not in name)


def resolve_gguf(path: str, kind: str) -> str:
    """A .gguf file, or a folder holding exactly one candidate (vision projectors and later shards skipped)."""
    p = Path(os.path.expanduser(path))
    if p.is_file():
        return str(p.resolve())
    if not p.is_dir():
        sys.exit(f"{kind} not found: {p}")
    files = sorted(glob.glob(str(p / "**" / "*.gguf"), recursive=True))
    cands = [f for f in files if (("embed" in Path(f).name.lower()) if kind == "embedding model" else not _is_aux(f))]
    if len(cands) == 1:
        print(f"  {kind}: {cands[0]}")
        return str(Path(cands[0]).resolve())
    if not cands:
        sys.exit(f"no {kind} .gguf in {p}" + (f" (found: {', '.join(Path(f).name for f in files)})" if files else ""))
    sys.exit(f"several {kind} candidates in {p}, pass the file itself:\n  " + "\n  ".join(cands))


def download_embedding_model() -> str | None:
    """Small multilingual embedding GGUF (Q8_0), first repo that can be downloaded."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("  ! huggingface_hub missing: pip install huggingface_hub")
        return None
    dest = Path.home() / ".local" / "share" / "jarvis" / "models"
    for repo in EMBED_REPOS:
        try:
            files = [f for f in list_repo_files(repo) if f.endswith(".gguf")]
            pick = next((f for f in files if "q8_0" in f.lower()), files[0] if files else None)
            if pick:
                print(f"  downloading {repo}/{pick} ...")
                return hf_hub_download(repo, pick, local_dir=str(dest))
        except Exception as e:                       # gated repo, network, ...
            print(f"  ! {repo}: {type(e).__name__}: {str(e)[:120]}")
    return None


def download_dflash(dest: Path) -> str | None:
    """The DFlash drafter of LLM_REPO (Q8_0, ~0.4 GB: the 2070 has no native BF16)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("  ! huggingface_hub missing: pip install huggingface_hub")
        return None
    files = [f for f in list_repo_files(LLM_REPO) if f.endswith(".gguf") and "dflash" in f.lower()]
    pick = next((f for f in files if "q8_0" in f.lower()), files[0] if files else None)
    if not pick:
        print(f"  ! no DFlash file in {LLM_REPO}")
        return None
    target = dest / Path(pick).name
    if target.exists():
        return str(target)
    print(f"  downloading {LLM_REPO}/{pick} to {dest} ...")
    return hf_hub_download(LLM_REPO, pick, local_dir=str(dest))


def download_llm(quant: str, dest: Path) -> str | None:
    """The chat GGUF of LLM_REPO matching ``quant`` (e.g. UD-Q4_K_XL), into ``dest``."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("  ! huggingface_hub missing: pip install huggingface_hub")
        return None
    files = [f for f in list_repo_files(LLM_REPO) if f.endswith(".gguf") and quant.lower() in f.lower()
             and "mmproj" not in f.lower() and "dflash" not in f.lower()]
    if not files:
        print(f"  ! no {quant} file in {LLM_REPO}")
        return None
    target = dest / Path(files[0]).name
    if target.exists():
        print(f"  {target.name} already downloaded")
        return str(target)
    print(f"  downloading {LLM_REPO}/{files[0]} to {dest} (several GB)...")
    return hf_hub_download(LLM_REPO, files[0], local_dir=str(dest))


def find_tts_python() -> str | None:
    for c in TTS_PYTHON_CANDIDATES:
        p = Path(os.path.expanduser(c))
        if p.exists():
            return str(p)
    return None


def preset(args, current: dict) -> dict:
    p: dict = {
        # Speech recognition, French forced. Parakeet v3 (default): CoVoST2 fr WER 6.05 vs 11.02 for
        # Whisper large-v3 (arXiv 2509.14128), int8 on CPU leaves the GPU to the LLM and the voice.
        "whisper_language": "fr",
        # Speech output: Qwen3-TTS streamed, French.
        "tts_engine": "qwen3",
        "tts_qwen3_language": "French",
        "tts_qwen3_size": "0.6B",
        # 8 codec steps (667 ms). Measured on an RTX 2070: 4 made generation slower than real time
        # (per-chunk overhead) and doubled the wait before speaking (1.6 s -> 3.4 s).
        "tts_qwen3_chunk": 8,
        # The planner stays on: with a small model it is what makes the reply call tools instead
        # of inventing an answer (measured: weather made up with it off), and it skips the memory
        # search when not needed. The LLM tool router costs a call; with only the built-in tools,
        # sending all of them is cheaper. "keyword" is NOT an option for French: it matches
        # query words against English tool descriptions.
        "planner_enabled": True,
        "tool_selection_strategy": "llm" if current.get("mcps") else "all",
        # French household wording.
        "timer_announcement": "{label} : c'est terminé !",
        "stop_commands": FR_STOP,
        "wake_aliases": sorted(set(current.get("wake_aliases", [])) | set(FR_WAKE_ALIASES)),
        "location_auto_detect": True,
        # End of speech after 600 ms of silence instead of 800: answers 0.2 s sooner, at the cost of
        # cutting a long mid-sentence pause.
        "endpoint_silence_ms": 600,
        # Short-term memory only: the conversation of the session, nothing filed or recalled
        # across sessions (a 4B model mis-files facts and recalls its own wrong answers).
        # Concise household voice agent instead of the dry-witted butler (spoken, tools first).
        "persona_style": "home_agent",
        # Plan while the intent judge decides (parallel LLM calls on separate llama-server slots).
        "speculative_planning": True,
        "memory_write": "off",
        "memory_enrichment_source": "none",
    }
    if args.stt == "whisper":
        p.update({"whisper_backend": "auto", "whisper_model": args.whisper_model,
                  "whisper_device": "cuda", "whisper_compute_type": "int8_float16"})
    else:
        p.update({"whisper_backend": "parakeet", "whisper_device": args.stt_device,
                  "stt_model": "nemo-canary-1b-v2" if args.stt == "canary" else "nemo-parakeet-tdt-0.6b-v3",
                  "stt_quantization": "int8" if args.stt_device == "cpu" else None})
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

    # llama.cpp started by Jarvis: chat server on 8080 (GPU), embedding server on 8081 (CPU).
    llm_model = args.llm_model or current.get("llama_server_model")
    if args.download_llm:
        here = Path(llm_model).expanduser() if llm_model else Path("C:/modeles")
        dest = here if here.is_dir() else (here.parent if llm_model else here)
        dest.mkdir(parents=True, exist_ok=True)
        llm_model = download_llm(args.download_llm, dest) or llm_model
    if llm_model:
        llm_model = resolve_gguf(llm_model, "chat model")
        exe = args.llama_server or current.get("llama_server_path") or find_llama_server()
        if not exe or not _exists(exe):
            sys.exit("llama-server not found: pass --llama-server <path to llama-server(.exe)>")
        llm_url = args.llm_url or current.get("llm_base_url") or "http://127.0.0.1:8080/v1"
        if "11434" in llm_url:                       # an Ollama URL left over from the defaults
            llm_url = "http://127.0.0.1:8080/v1"
        p.update({"llama_server_path": str(Path(exe).resolve()),
                  "llama_server_model": str(Path(llm_model).expanduser().resolve()),
                  "llm_provider": "openai_compatible", "llm_base_url": llm_url,
                  "llm_chat_model": Path(llm_model).stem, "fast_model": ""})
        if args.dflash:
            drafter = download_dflash(Path(p["llama_server_model"]).parent)
            if drafter:
                p["llama_server_draft_model"] = str(Path(drafter).resolve())
        elif args.no_dflash:
            p["llama_server_draft_model"] = None
        emb = args.embed_model or current.get("llama_server_embedding_model")
        if not emb and args.download_embed:
            emb = download_embedding_model()
        if emb:
            emb = resolve_gguf(emb, "embedding model")
            p.update({"llama_server_embedding_model": str(Path(emb).expanduser().resolve()),
                      "embedding_provider": "openai_compatible",
                      "embedding_base_url": "http://127.0.0.1:8081/v1",
                      "embedding_model": Path(emb).stem})
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", help="audio file of the voice to clone (wav/mp3/flac, 5-15 s, one speaker)")
    ap.add_argument("--speaker", help="built-in Qwen3 voice instead of cloning (e.g. vivian, serena, ryan)")
    ap.add_argument("--tts-python", help="python of the Qwen3-TTS environment (auto-detected)")
    ap.add_argument("--llm-url", help="OpenAI-compatible server, e.g. http://127.0.0.1:8080/v1 (llama-server)")
    ap.add_argument("--llm-model", help="chat model .gguf: Jarvis then starts llama-server itself")
    ap.add_argument("--download-llm", metavar="QUANT", nargs="?", const="UD-Q4_K_XL",
                    help="download Nemotron-3.5-Bolt-E4B at this quant (default UD-Q4_K_XL, 5.1 GB) next to "
                         "the current model and use it")
    ap.add_argument("--dflash", action="store_true",
                    help="download the DFlash drafter (Q8_0, ~0.4 GB) and enable speculative decoding")
    ap.add_argument("--no-dflash", action="store_true", help="disable speculative decoding")
    ap.add_argument("--llama-server", help="path to llama-server(.exe) (auto-detected)")
    ap.add_argument("--embed-model", help="embedding model .gguf (memory search), served on port 8081")
    ap.add_argument("--download-embed", action="store_true",
                    help="download a small multilingual embedding model (EmbeddingGemma 300M Q8_0)")
    ap.add_argument("--stt", choices=["parakeet", "canary", "whisper"], default="parakeet",
                    help="speech recognition: parakeet (default), canary (1B), whisper")
    ap.add_argument("--stt-device", choices=["cpu", "auto"], default="cpu",
                    help="cpu (default, int8, keeps VRAM free) or auto (GPU if onnxruntime-gpu is installed)")
    ap.add_argument("--whisper-model", default="large-v3-turbo", help="with --stt whisper")
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
    if not new.get("llama_server_model"):
        print("  ! llama-server is not started by Jarvis: pass --llm-model <chat model .gguf>")
        found = list_ggufs()
        if found:
            print("    .gguf files found:")
            for f in found[:15]:
                print(f"      {f}")
    elif not new.get("llama_server_embedding_model"):
        print("  ! no embedding model (memory search will not work): add --download-embed")
    if args.dry_run or not changed:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.with_suffix(".json.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps({**current, **new}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved (previous version: config.json.bak). Restart Jarvis.")


if __name__ == "__main__":
    main()
