"""Sample a French Tatoeba TSV export; keep sentence IDs for provenance."""
import argparse
import bz2
import random
from pathlib import Path

from curate import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv_bz2", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--domain", type=Path, default=Path(__file__).with_name("pilot-fr.txt"))
    parser.add_argument("--heldout", type=Path, default=Path(__file__).with_name("heldout-fr.txt"))
    parser.add_argument("--count", type=int, default=3200)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    excluded = {normalize(t) for t in args.heldout.read_text("utf8").splitlines()}
    rows = []
    for line in args.domain.read_text("utf8").splitlines():
        key = normalize(line)
        if key and key not in excluded and "|" not in line:
            rows.append(("local", line.strip()))
            excluded.add(key)
    pool = []
    with bz2.open(args.tsv_bz2, "rt", encoding="utf8") as source:
        for line in source:
            ident, lang, sentence = line.rstrip("\n").split("\t", 2)
            key = normalize(sentence)
            if lang == "fra" and 4 <= len(sentence.split()) <= 28 and "|" not in sentence and key not in excluded:
                excluded.add(key)
                pool.append((ident, sentence))
    if len(pool) < args.count:
        parser.error("Too few eligible French sentences")
    rows.extend(random.Random(42).sample(pool, args.count))
    args.output.write_text("".join(t + "\n" for _, t in rows), encoding="utf8")
    args.output.with_suffix(".sources.tsv").write_text("".join(f"{i}\t{t}\n" for i, t in rows), encoding="utf8")


if __name__ == "__main__":
    main()
