#!/usr/bin/env python3
"""Why does llama-server re-read the whole prompt every turn? Find where two turns diverge.

llama-server only skips the part of a prompt identical to one it has cached. This script takes the
first model call of the two most recent replies (prompt dumps written with JARVIS_DUMP_PROMPTS=1),
asks the running llama-server to render them with the model's own chat template
(/apply-template), and prints how many leading tokens the two turns share and the text around the
first difference. Everything after that point is recomputed on every turn.

    set JARVIS_DUMP_PROMPTS=1   (PowerShell: $env:JARVIS_DUMP_PROMPTS=1), run Jarvis, ask 2 questions
    python scripts/prompt_cache_probe.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jarvis.config import load_settings  # noqa: E402
from jarvis.llm.backend import strip_nonstandard_message_fields  # noqa: E402


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    cfg = load_settings()
    u = urlparse(cfg.llm_base_url)
    root = f"{u.scheme}://{u.hostname}:{u.port}"
    first = {}                                     # first model call of each reply (t01, or t02 after a tool)
    for d in sorted((Path.home() / ".local" / "share" / "jarvis" / "prompts").glob("turn-*.json")):
        session = d.name.split("-")[2]
        first.setdefault(session, d)
    dumps = sorted(first.values())
    if len(dumps) < 2:
        sys.exit("need two prompt dumps: run Jarvis with JARVIS_DUMP_PROMPTS=1 and ask two questions")
    prompts = []
    for d in dumps[-2:]:
        p = json.loads(d.read_text(encoding="utf-8"))
        body = {"messages": strip_nonstandard_message_fields(p["messages"])}
        if p.get("tools_schema") and not p.get("use_text_tools"):
            body["tools"] = p["tools_schema"]
        prompts.append(post(root + "/apply-template", body)["prompt"])
        print(f"{d.name}: query {p['query']!r}")
    a, b = prompts
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    tok = lambda s: len(post(root + "/tokenize", {"content": s})["tokens"])
    shared, total = tok(a[:i]), tok(b)
    print(f"\nshared prefix: {shared} of {total} tokens ({100 * shared / max(total, 1):.0f}%) "
          f"-> {total - shared} tokens recomputed each turn")
    print("\n--- turn A around the first difference ---\n" + a[max(0, i - 400):i + 300])
    print("\n--- turn B around the first difference ---\n" + b[max(0, i - 400):i + 300])


if __name__ == "__main__":
    main()
