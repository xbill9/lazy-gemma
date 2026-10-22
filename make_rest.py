#!/usr/bin/env python3
"""make_rest.py -- the GSM8K test problems data/heldout.jsonl did not draw, as a second held-out set.

  make_rest.py

  data/heldout-rest.jsonl  every test problem not in data/heldout.jsonl, in source order

Fetches the test split from the URL in data/manifest.json and stops unless its sha256 matches, so the ids
are the ones make_splits.py gave. Leaves train.jsonl and heldout.jsonl untouched.
"""

import hashlib
import json
import os

import make_splits

DATA = os.path.join(make_splits.HERE, "data")


def main():
    with open(os.path.join(DATA, "manifest.json")) as f:
        manifest = json.load(f)
    src = manifest["test_source"]
    raw = make_splits.fetch(src["url"])
    assert hashlib.sha256(raw).hexdigest() == src["sha256"], "test split changed since make_splits.py"
    test = make_splits.rows(raw, "test")
    with open(os.path.join(DATA, "heldout.jsonl")) as f:
        drawn = {json.loads(line)["id"] for line in f}
    with open(os.path.join(DATA, "train.jsonl")) as f:
        train_q = {json.loads(line)["question"] for line in f}
    rest = [r for r in test if r["id"] not in drawn]
    assert len(rest) == manifest["test_total"] - manifest["heldout"], len(rest)
    assert not {r["question"] for r in rest} & train_q, "train/held-out overlap"
    make_splits.write(os.path.join(DATA, "heldout-rest.jsonl"), rest)
    print(json.dumps({"heldout_rest": len(rest)}))


if __name__ == "__main__":
    main()
