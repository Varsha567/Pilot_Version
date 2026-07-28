"""
One-time setup: downloads the official 164-problem HumanEval dataset and
picks the same 20 stratified-by-difficulty problems used in this pilot.
Run this once: python setup_data.py
"""
import json
import gzip
import urllib.request

URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

print("Downloading HumanEval dataset...")
urllib.request.urlretrieve(URL, "HumanEval.jsonl.gz")

with gzip.open("HumanEval.jsonl.gz", "rt") as f:
    rows = [json.loads(line) for line in f]

print(f"Loaded {len(rows)} problems")

rows_sorted = sorted(rows, key=lambda r: len(r["canonical_solution"]))
n = len(rows_sorted)
picks = [rows_sorted[int(i * (n - 1) / 19)] for i in range(20)]

seen = set()
with open("humaneval_subset.jsonl", "w") as f:
    for p in picks:
        if p["task_id"] in seen:
            continue
        seen.add(p["task_id"])
        f.write(json.dumps(p) + "\n")

print(f"Wrote {len(seen)} problems to humaneval_subset.jsonl")