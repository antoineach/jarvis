"""Prepare anonymised, paired listening trials for the same held-out phrases."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path


def samples(folder):
    metrics = folder / "metrics.json"
    if metrics.is_file():
        trials = json.loads(metrics.read_text("utf8"))["trials"]
        texts = [row["text"] for row in trials if row["repeat"] == 1]
        pairs = [(text, folder / f"{index:03}.wav") for index, text in enumerate(texts)]
    else:
        pairs = []
        for row in (folder / "metadata.csv").read_text("utf8").splitlines():
            key, text = row.split("|", 1)
            if key != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError(f"Invalid teacher manifest key in {folder}")
            pairs.append((text, folder / "wav" / (key + ".wav")))
    result = dict(pairs)
    if not result or len(result) != len(pairs):
        raise ValueError(f"Empty or duplicated prompts in {folder}")
    for wav in result.values():
        if not wav.is_file():
            raise FileNotFoundError(wav)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    folders = [p.resolve() for p in args.results]
    if len(folders) < 2 or len(set(folders)) != len(folders):
        parser.error("Supply two or more distinct candidate directories")
    candidates = {folder: samples(folder) for folder in folders}
    phrases = list(candidates[folders[0]])
    for folder, clips in candidates.items():
        if clips.keys() != set(phrases):
            parser.error(f"Held-out prompts differ in {folder}: {clips.keys() ^ set(phrases)}")
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.SystemRandom()
    with (args.output/"key.csv").open("w", newline="") as key, (args.output/"scores.csv").open("w", newline="") as scores:
        answer, sheet = csv.writer(key), csv.writer(scores)
        answer.writerow(["wav", "candidate"])
        sheet.writerow(["wav", "text", "incorrect_words", "similarity_1_to_5", "naturalness_1_to_5"])
        for index, phrase in enumerate(phrases):
            order = rng.sample(folders, len(folders))
            for position, folder in enumerate(order, 1):
                filename = f"{index:03}-{position}.wav"
                shutil.copyfile(candidates[folder][phrase], args.output / filename)
                answer.writerow([filename, str(folder)])
                sheet.writerow([filename, phrase, "", "", ""])


if __name__ == "__main__":
    main()
