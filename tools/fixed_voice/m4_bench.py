"""Measure local CPU synthesis and save listening samples on Apple silicon."""
import argparse
import json
import math
import platform
import resource
import statistics
import sys
import time
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("sentences", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    lines = [s.strip() for s in args.sentences.read_text("utf8").splitlines() if s.strip()]
    if not lines or args.repeats < 1:
        parser.error("Supply sentences and at least one repetition")
    from piper import PiperVoice

    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    voice = PiperVoice.load(str(args.model), use_cuda=False)
    load_ms = 1000 * (time.perf_counter() - begin)
    rows = []
    for repeat in range(args.repeats + 1):
        phrases = lines[:1] if repeat == 0 else lines
        for index, sentence in enumerate(phrases):
            start = time.perf_counter()
            pcm = bytearray()
            first = None
            for chunk in voice.synthesize(sentence):
                pcm.extend(chunk.audio_int16_bytes)
                if first is None and pcm:
                    first = time.perf_counter()
            end = time.perf_counter()
            if first is None:
                raise RuntimeError("No audio for " + sentence)
            seconds = len(pcm) / (chunk.sample_rate * chunk.sample_width * chunk.sample_channels)
            rows.append(dict(repeat=repeat, text=sentence, first_pcm_ms=1000 * (first-start),
                             rtf=(end-start)/seconds, audio_s=seconds))
            if repeat == 1:
                with wave.open(str(args.output / f"{index:03}.wav"), "wb") as wav:
                    wav.setparams((chunk.sample_channels, chunk.sample_width, chunk.sample_rate, 0, "NONE", ""))
                    wav.writeframes(pcm)
    warm = [r for r in rows if r["repeat"]]
    firsts = sorted(r["first_pcm_ms"] for r in warm)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(platform=platform.platform(), machine=platform.machine(), load_ms=load_ms,
                  model_bytes=args.model.stat().st_size, cold_first_pcm_ms=rows[0]["first_pcm_ms"],
                  warm_first_pcm_p50_ms=statistics.median(firsts),
                  warm_first_pcm_p95_ms=firsts[math.ceil(.95*len(firsts))-1],
                  warm_median_rtf=statistics.median(r["rtf"] for r in warm),
                  peak_process_mib=rss/(1024**2 if sys.platform=="darwin" else 1024))
    (args.output/"metrics.json").write_text(json.dumps(dict(summary=result, trials=rows), indent=2))
    print("🔊", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
