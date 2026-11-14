"""Sample gold sentences for `nlp-compare` into one JSONL file.

Two sources, both read from the local HuggingFace cache (download them with
`hf download <repo> --repo-type dataset` first):

* `dragonscale-ai/kniv-corpus-en`, `corpus/gold/dev.parquet`: NER, POS,
  dependencies and a dialogue act per sentence, six domains. kniv v1 was
  trained on this corpus's train split, so its dev split is held out but in
  v1's own domain and annotation style. The labels are LLM-produced.
* `DFKI-SLT/conll04`, test split: person / location / organisation spans on
  news text that neither generation was trained on.

Usage:
    uv run --no-project --with pyarrow python -I nlp_eval_prep.py OUT.jsonl \
        [--per-domain 300] [--seed 7]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import pyarrow.parquet as pq

HUB = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))


def snapshot_file(repo: str, rel: str) -> str:
    hits = glob.glob(f"{HUB}/datasets--{repo.replace('/', '--')}/snapshots/*/{rel}")
    if not hits:
        raise SystemExit(f"{repo}:{rel} is not in {HUB}; download it first")
    return hits[0]


def bio_spans(tags: list[str]) -> list[dict]:
    spans, cur = [], None
    for i, tag in enumerate(tags + ["O"]):
        if cur and not (tag.startswith("I-") and tag[2:] == cur["type"]):
            cur["end"] = i
            spans.append(cur)
            cur = None
        if tag.startswith("B-") or (tag.startswith("I-") and cur is None):
            cur = {"start": i, "type": tag[2:].split(" ")[0]}
    return spans


def kniv_dev(per_domain: int, rng: random.Random) -> list[dict]:
    table = pq.read_table(snapshot_file("dragonscale-ai/kniv-corpus-en", "corpus/gold/dev.parquet"))
    rows = table.to_pylist()
    by_domain: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_domain.setdefault(r["sent_id"].rsplit("-", 1)[0], []).append(i)
    out = []
    for domain in sorted(by_domain):
        for i in sorted(rng.sample(by_domain[domain], min(per_domain, len(by_domain[domain])))):
            r = rows[i]
            out.append(
                {
                    "source": "kniv-dev",
                    "domain": domain,
                    "id": r["sent_id"],
                    "text": r["text"],
                    "tokens": r["tokens"],
                    "pos": r["pos_tags"],
                    # 1-based, 0 = root, as in the corpus.
                    "heads": r["heads"],
                    "deprels": r["deprels"],
                    "entities": bio_spans(r["ner_tags"]),
                    "act": r["cls"],
                }
            )
    return out


def conll04() -> list[dict]:
    path = snapshot_file("DFKI-SLT/conll04", "data/test-00000-of-00001.parquet")
    names = {"Peop": "PERSON", "Loc": "LOC", "Org": "ORG", "Other": "OTHER"}
    out = []
    for r in pq.read_table(path).to_pylist():
        out.append(
            {
                "source": "conll04",
                "domain": "news",
                "id": f"conll04-{r['orig_id']}",
                "text": " ".join(r["tokens"]),
                "tokens": r["tokens"],
                "entities": [
                    {"start": e["start"], "end": e["end"], "type": names[e["type"]]}
                    for e in r["entities"]
                ],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-domain", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rows = kniv_dev(args.per_domain, random.Random(args.seed)) + conll04()
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} sentences to {args.out}")


if __name__ == "__main__":
    main()
