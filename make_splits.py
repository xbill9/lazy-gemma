#!/usr/bin/env python3
"""make_splits.py -- fixed, disjoint train and held-out GSM8K sets, written before any training.

  make_splits.py [--heldout N] [--seed S]

  data/heldout.jsonl  N problems sampled from the GSM8K test split by seed S
  data/train.jsonl    the whole GSM8K train split
  data/manifest.json  source URLs, their sha256, seed, counts

Records are {"id", "question", "answer"}, ids "test-<i>" / "train-<i>" by position in the source
file, so the two sets cannot share an id. Answers are the text after "####", commas dropped, as in
gemma4-dev/jev-tpu-v5e1/build_gen_set.py.
"""

import argparse
import hashlib
import json
import os
import random
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def answer(solution):
    return solution.split("####")[-1].strip().replace(",", "")


def rows(raw, prefix):
    lines = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return [{"id": f"{prefix}-{i}", "question": r["question"], "answer": answer(r["answer"])} for i, r in enumerate(lines)]


def write(path, recs):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    manifest = {"seed": a.seed}
    sets = {}
    for split, prefix in (("train", "train"), ("test", "test")):
        url = BASE + split + ".jsonl"
        raw = fetch(url)
        manifest[f"{split}_source"] = {"url": url, "sha256": hashlib.sha256(raw).hexdigest()}
        sets[split] = rows(raw, prefix)

    heldout = random.Random(a.seed).sample(sets["test"], a.heldout)
    assert not {r["question"] for r in heldout} & {r["question"] for r in sets["train"]}, "train/held-out overlap"

    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, "train.jsonl"), sets["train"])
    write(os.path.join(out, "heldout.jsonl"), heldout)
    manifest.update(train=len(sets["train"]), heldout=len(heldout), test_total=len(sets["test"]))
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
