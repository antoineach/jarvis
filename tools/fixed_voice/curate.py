"""Select intelligible teacher recordings for a Piper fine-tune."""
import argparse
import json
import re
from pathlib import Path
import unicodedata


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text.casefold()) if c.isalnum())


def french_text(text):
    from num2words import num2words

    def clock(match):
        hour = int(match[1])
        minute = int(match[2] or match[3] or 0)
        unit = "heure" if hour < 2 else "heures"
        hours = f"{hour} {unit}" if hour != 1 else "une heure"
        return f"{hours} {minute}" if minute else hours

    text = re.sub(r"(?<!\w)([01]?\d|2[0-3])\s*(?:h\s*([0-5]\d)?|:([0-5]\d))(?!\w)",
                  clock, text, flags=re.I)
    return re.sub(r"(?<!\w)[0-9]{1,15}(?!\w)", lambda m: num2words(int(m[0]), lang="fr"), text)


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
    parser.add_argument("--approved-short", type=Path,
                        help="One reviewed clip key per line, for borderline short phrases")
    args = parser.parse_args()
    approved = set(args.approved_short.read_text().splitlines()) if args.approved_short else set()
    language = json.loads((args.dataset / "voice.json").read_text())["language"]
    spoken = french_text if language == "French" else str
    import onnx_asr
    import soundfile as sf

    asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path=str(args.asr), quantization="int8",
                              providers=["CPUExecutionProvider"])
    heldout = {normalize(spoken(t)) for t in args.heldout.read_text("utf8").splitlines()}
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
            score = None
            manually_kept = False
            reason = "heldout" if normalize(spoken(text)) in heldout else ""
            if not reason and not 0.25 <= seconds <= 20:
                reason = "duration"
            if not reason:
                prediction = asr.recognize(str(wav))
                target = spoken(text)
                score = cer(target, spoken(prediction))
                if score > args.max_cer:
                    size = len(normalize(target))
                    borderline = 1 <= size <= 8 and normalize(prediction) and round(score * size) == 1
                    manually_kept = bool(borderline and key in approved)
                    if not manually_kept:
                        reason = "review_short" if borderline else "transcript"
            if not reason:
                kept.append(f"{wav.name}|{text}\n")
                total_seconds += seconds
            report.write(json.dumps(dict(key=key, text=text, prediction=prediction, reason=reason,
                                         seconds=seconds, cer=score, approved_short=manually_kept), ensure_ascii=False) + "\n")
    tmp = args.dataset / "accepted.tmp"
    tmp.write_text("".join(kept), encoding="utf8")
    tmp.replace(args.dataset / "accepted.csv")
    (args.dataset / "accepted.source").write_text(digest)
    print(f"🔎 {len(kept)} accepted clips, {total_seconds / 3600:.2f} hours")


if __name__ == "__main__":
    main()
