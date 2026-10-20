#!/usr/bin/env python3
"""lazy_eval.py -- accuracy and output tokens per answer on GSM8K, against a llama-server, scored in code.

  lazy_eval.py run <upstream> <data.jsonl> <out.jsonl> [--limit N] [--samples K] [--temperature T]
                   [--max-tokens M] [--budget B] [--no-think] [--concurrency C]
                   [--prompt-budget N]
  lazy_eval.py score <out.jsonl>

Thinking is on per request (chat_template_kwargs enable_thinking), since the rig servers default to
--reasoning off. Only `content` is graded; the thinking goes to `reasoning_content`. completion_tokens
is the server's count with the model's own tokenizer and covers thinking plus answer. --budget sends
llama.cpp's per-request thinking_budget_tokens, the no-training way to think less. --prompt-budget
states a budget in the prompt instead (PROMPT_BUDGET), for the model trained to follow it.

GSM8K prompt and grading are copied from gemma4-dev/jev-tpu-v5e1/gen_eval.py: right when the number
after "Answer:" (else the last number) equals the reference.
"""

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = (
    "Solve the following math problem. Think step by step, then give the final answer "
    "on its own last line as 'Answer: <number>'.\n\n{question}"
)
PROMPT_BUDGET = (
    "Solve the following math problem. Think for at most {budget} tokens, then give the final answer "
    "on its own last line as 'Answer: <number>'.\n\n{question}"
)


def prompt(question, budget=None):
    return PROMPT.format(question=question) if budget is None else PROMPT_BUDGET.format(budget=budget, question=question)


def number(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract(text):
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text or "")
    if m:
        return number(m[-1])
    m = re.findall(r"-?[\d,]*\.?\d+", text or "")
    return number(m[-1]) if m else None


def right(text, answer):
    got = extract(text)
    return got is not None and abs(got - float(answer)) < 1e-6


def post(upstream, body, timeout=1800):
    req = urllib.request.Request(
        upstream.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def one(a, rec, k):
    body = {
        "temperature": a.temperature,
        "max_tokens": a.max_tokens,
        "messages": [{"role": "user", "content": prompt(rec["question"], a.prompt_budget)}],
        "chat_template_kwargs": {"enable_thinking": not a.no_think},
        "cache_prompt": False,
    }
    if a.temperature > 0:
        body["seed"] = k
    if a.budget is not None:
        body["thinking_budget_tokens"] = a.budget
    out = {"id": rec["id"], "sample": k, "answer": rec["answer"]}
    t0 = time.time()
    try:
        r = post(a.upstream, body)
    except (OSError, ValueError) as e:  # recorded, scored wrong, counted separately
        out.update(error=f"{type(e).__name__}: {e}"[:300], right=False, seconds=round(time.time() - t0, 2))
        return out
    msg = r["choices"][0]["message"]
    out.update(
        text=msg.get("content"),
        reasoning=msg.get("reasoning_content"),
        finish=r["choices"][0].get("finish_reason"),
        completion_tokens=r["usage"]["completion_tokens"],
        prompt_tokens=r["usage"]["prompt_tokens"],
        seconds=round(time.time() - t0, 2),
        extracted=extract(msg.get("content")),
        right=right(msg.get("content"), rec["answer"]),
    )
    return out


def summary(rows):
    ok = [r for r in rows if "error" not in r]
    toks = [r["completion_tokens"] for r in ok]
    n_right = sum(r["right"] for r in rows)
    return {
        "n": len(rows),
        "right": n_right,
        "accuracy": round(n_right / len(rows), 4) if rows else None,
        "errors": len(rows) - len(ok),
        "truncated": sum(r.get("finish") == "length" for r in ok),
        "mean_tokens": round(statistics.mean(toks), 1) if toks else None,
        "median_tokens": statistics.median(toks) if toks else None,
        "tokens_per_correct": round(sum(toks) / n_right, 1) if n_right else None,
        "seconds_per_correct": round(sum(r["seconds"] for r in rows) / n_right, 2) if n_right else None,
    }


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def run(a):
    recs = load(a.data)
    if a.limit:
        recs = recs[: a.limit]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows = []
    lock = threading.Lock()
    with open(a.out, "w") as f:  # written as it goes so a crash keeps the rows

        def job(rec, k):
            row = one(a, rec, k)
            with lock:
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()

        with ThreadPoolExecutor(a.concurrency) as ex:
            list(ex.map(lambda rk: job(*rk), [(rec, k) for rec in recs for k in range(a.samples)]))
    print(json.dumps(summary(rows)))


def score(a):
    print(json.dumps(summary(load(a.out))))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("upstream")
    r.add_argument("data")
    r.add_argument("out")
    r.add_argument("--limit", type=int)
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--budget", type=int)
    r.add_argument("--no-think", action="store_true")
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--prompt-budget", type=int, help="state a thinking budget in the prompt (PROMPT_BUDGET)")
    s = sub.add_parser("score")
    s.add_argument("out")
    a = p.parse_args()
    run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    sys.exit(main())
