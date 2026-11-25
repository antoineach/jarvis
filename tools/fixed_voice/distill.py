"""Offline, single-speaker Qwen3-TTS to Piper waveform distillation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


LANGUAGES = {
    "English": "en-us", "French": "fr-fr", "German": "de",
    "Spanish": "es", "Italian": "it", "Portuguese": "pt",
    "Russian": "ru", "Chinese": "cmn", "Japanese": "ja", "Korean": "ko",
}


def generate(args):
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    lines = list(dict.fromkeys(t.strip() for t in args.sentences.read_text("utf8").splitlines() if t.strip()))
    if not lines or any("|" in t for t in lines):
        raise ValueError("Supply one non-empty sentence per line, without pipe characters")
    transcript = args.transcript.read_text("utf8").strip()
    if not transcript:
        raise ValueError("Reference transcript is empty")
    prompt_info = {
        "language": args.language,
        "reference": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "transcript": hashlib.sha256(transcript.encode()).hexdigest(),
        "teacher": str(args.teacher.resolve()),
    }
    args.dataset.mkdir(parents=True, exist_ok=True)
    identity = args.dataset / "voice.json"
    if identity.exists():
        if json.loads(identity.read_text()) != prompt_info:
            raise ValueError("Choose a new dataset directory for this voice or language")
    elif any(args.dataset.iterdir()):
        raise ValueError("Choose an empty dataset directory")
    else:
        identity.write_text(json.dumps(prompt_info))
    wav_dir = args.dataset / "wav"
    wav_dir.mkdir(exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(str(args.teacher.resolve()), device_map=args.device,
                                          dtype=torch.float32 if args.device == "cpu" else torch.float16)
    prompt = model.create_voice_clone_prompt(ref_audio=str(args.reference), ref_text=transcript)
    rows = []
    for line in lines:
        key = hashlib.sha256(line.encode()).hexdigest()
        wav = wav_dir / (key + ".wav")
        if not wav.exists():
            waves, rate = model.generate_voice_clone(text=line, language=args.language, voice_clone_prompt=prompt)
            samples = np.asarray(waves[0]).squeeze()
            if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all() or not np.any(samples):
                raise ValueError(f"Invalid teacher audio for {key}")
            tmp = wav.with_suffix(".partial.wav")
            sf.write(tmp, samples, rate)
            tmp.replace(wav)
        rows.append(f"{key}|{line}\n")
    tmp = args.dataset / "metadata.tmp"
    tmp.write_text("".join(rows), encoding="utf8")
    tmp.replace(args.dataset / "metadata.csv")


def piper(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


def train(args):
    language = json.loads((args.dataset / "voice.json").read_text())["language"]
    args.work.mkdir(parents=True, exist_ok=False)
    piper("piper_train.preprocess", "--language", LANGUAGES[language], "--input-dir", args.dataset,
          "--output-dir", args.work, "--dataset-format", "ljspeech", "--single-speaker", "--sample-rate", 22050)
    piper("piper_train", "--dataset-dir", args.work, "--default_root_dir", args.work,
          "--accelerator", "gpu", "--devices", 1, "--batch-size", args.batch_size,
          "--max_epochs", args.epochs, "--checkpoint-epochs", 1, "--num-test-examples", 0,
          "--validation-split", 0.05, "--precision", 32)


def export(args):
    import onnx

    config = args.work / "config.json"
    if json.loads(config.read_text())["num_speakers"] != 1:
        raise ValueError("Expected a single-speaker Piper config")
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as scratch:
        model = Path(scratch) / "voice.onnx"
        piper("piper_train.export_onnx", args.checkpoint, model)
        graph = onnx.load(str(model))
        onnx.checker.check_model(graph)
        count = sum(math.prod(t.dims) for t in graph.graph.initializer)
        if not 0 < count < 100_000_000:
            raise ValueError(f"Export outside parameter budget: {count:,}")
        shutil.copyfile(config, args.output / "voice.onnx.json")
        model.replace(args.output / model.name)
        print(f"🔊 Exported Piper voice with {count:,} ONNX initializer elements")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    for name in ("teacher", "reference", "transcript", "sentences", "dataset"):
        gen.add_argument("--" + name, type=Path, required=True)
    gen.add_argument("--language", choices=LANGUAGES, required=True)
    gen.add_argument("--device", default="cuda:0")
    fit = commands.add_parser("train")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--work", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=2000)
    fit.add_argument("--batch-size", type=int, default=16)
    out = commands.add_parser("export")
    for name in ("work", "checkpoint", "output"):
        out.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"generate": generate, "train": train, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
