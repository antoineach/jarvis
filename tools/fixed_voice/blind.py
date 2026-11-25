"""Prepare anonymised, paired listening trials for the same held-out phrases."""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    if len(set(p.resolve() for p in args.results)) < 2:
        parser.error("Supply at least two distinct benchmark directories")
    phrases = None
    for folder in args.results:
        trials = json.loads((folder / "metrics.json").read_text())["trials"]
        current = [r["text"] for r in trials if r["repeat"] == 1]
        if not current or (phrases is not None and phrases != current):
            parser.error("All candidates must use the same non-empty held-out set in the same order")
        phrases = current
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.SystemRandom()
    with (args.output/"key.csv").open("w", newline="") as key, (args.output/"scores.csv").open("w", newline="") as scores:
        answer, sheet = csv.writer(key), csv.writer(scores)
        answer.writerow(["wav", "candidate"])
        sheet.writerow(["wav", "text", "incorrect_words", "similarity_1_to_5", "naturalness_1_to_5"])
        for index, phrase in enumerate(phrases):
            order = rng.sample(args.results, len(args.results))
            for position, folder in enumerate(order, 1):
                filename = f"{index:03}-{position}.wav"
                shutil.copyfile(folder / f"{index:03}.wav", args.output / filename)
                answer.writerow([filename, str(folder)])
                sheet.writerow([filename, phrase, "", "", ""])


if __name__ == "__main__":
    main()
