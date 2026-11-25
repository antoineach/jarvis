"""Select intelligible teacher recordings for a Piper fine-tune."""
import argparse
import json
from pathlib import Path
import unicodedata


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text.casefold()) if c.isalnum())


def cer(reference, prediction):
    source, target = normalize(reference), normalize(prediction)
    row = list(range(len(target) + 1))
    for i, char in enumerate(source, 1):
        new = [i]
        for j, other in enumerate(target, 1):
            new.append(min(row[j] + 1, new[-1] + 1, row[j - 1] + (char != other)))
        row = new
    return row[-1] / max(1, len(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--max-cer", type=float, default=0.12)
    args = parser.parse_args()
    import onnx_asr
    import soundfile as sf

    asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path=str(args.asr), quantization="int8",
                              providers=["CPUExecutionProvider"])
    heldout = {normalize(t) for t in args.heldout.read_text("utf8").splitlines()}
    kept = []
    total_seconds = 0
    manifest = args.dataset / "metadata.csv"
    import hashlib
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with (args.dataset / "review.jsonl").open("w", encoding="utf8") as report:
        for line in manifest.read_text("utf8").splitlines():
            key, text = line.split("|", 1)
            wav = args.dataset / "wav" / (key + ".wav")
            seconds = sf.info(wav).duration
            prediction = ""
            reason = "heldout" if normalize(text) in heldout else ""
            if not reason and not 0.25 <= seconds <= 20:
                reason = "duration"
            if not reason:
                prediction = asr.recognize(str(wav))
                if cer(text, prediction) > args.max_cer:
                    reason = "transcript"
            if not reason:
                kept.append(f"{wav.name}|{text}\n")
                total_seconds += seconds
            report.write(json.dumps(dict(key=key, text=text, prediction=prediction, reason=reason,
                                         seconds=seconds), ensure_ascii=False) + "\n")
    tmp = args.dataset / "accepted.tmp"
    tmp.write_text("".join(kept), encoding="utf8")
    tmp.replace(args.dataset / "accepted.csv")
    (args.dataset / "accepted.source").write_text(digest)
    print(f"🔎 {len(kept)} accepted clips, {total_seconds / 3600:.2f} hours")


if __name__ == "__main__":
    main()
