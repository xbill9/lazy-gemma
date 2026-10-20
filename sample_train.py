#!/usr/bin/env python3
"""sample_train.py -- K sampled answers per training problem, token ids kept, graded in code.

  sample_train.py <upstream> <out.jsonl> [--problems N] [--samples K] [--seed S] [--concurrency C]
                  [--temperature T] [--top-p P] [--top-k K] [--max-tokens M]

Draws N problems from data/train.jsonl by seed S. Each sample goes through /apply-template (thinking
on) and /completion with return_tokens, so the row holds the exact prompt and generated token ids the
trainer will see. The answer is the text after the thinking block's closing <channel|>, graded with
lazy_eval.right. Rows: id, sample, prompt_ids, gen_ids, n_gen, finish, right, extracted.
"""

import argparse
import json
import random
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import lazy_eval

END_THINK = "<channel|>"


def post(upstream, path, body, timeout=1800):
    req = urllib.request.Request(upstream.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def answer_text(raw):
    """Text after the thinking block; None if the block never closed."""
    return raw.split(END_THINK, 1)[1] if END_THINK in raw else None


def one(a, rec, k):
    msgs = [{"role": "user", "content": lazy_eval.PROMPT.format(question=rec["question"])}]
    prompt = post(a.upstream, "/apply-template", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": True}})["prompt"]
    ids = post(a.upstream, "/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    r = post(a.upstream, "/completion", {
        "prompt": ids, "n_predict": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
        "min_p": 0.0, "seed": a.seed * 1000 + k, "cache_prompt": False, "return_tokens": True,
    })
    text = answer_text(r["content"])
    return {"id": rec["id"], "sample": k, "answer": rec["answer"], "prompt_ids": ids, "gen_ids": r["tokens"],
            "n_gen": len(r["tokens"]), "finish": "length" if r.get("stop_type") == "limit" else r.get("stop_type"),
            "extracted": lazy_eval.extract(text) if text is not None else None,
            "right": text is not None and lazy_eval.right(text, rec["answer"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("upstream")
    p.add_argument("out")
    p.add_argument("--problems", type=int, default=1000)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=4096)
    a = p.parse_args()

    recs = lazy_eval.load("data/train.jsonl")
    recs = random.Random(a.seed).sample(recs, a.problems)
    lock = threading.Lock()
    n = right = 0
    with open(a.out, "w") as f:

        def job(rk):
            nonlocal n, right
            try:
                row = one(a, *rk)
            except (OSError, ValueError) as e:  # recorded, never selected
                row = {"id": rk[0]["id"], "sample": rk[1], "error": f"{type(e).__name__}: {e}"[:300], "right": False}
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
                right += row["right"]

        with ThreadPoolExecutor(a.concurrency) as ex:
            list(ex.map(job, [(rec, k) for rec in recs for k in range(a.samples)]))
    print(json.dumps({"samples": n, "right": right, "problems": len(recs)}))


if __name__ == "__main__":
    main()
