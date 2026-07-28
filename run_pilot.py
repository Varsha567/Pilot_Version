"""
Mini pilot harness.

Usage:
  # dry-run with mocked models + subprocess sandbox (no keys, no Docker needed)
  MOCK_MODELS=1 SANDBOX_MODE=subprocess python3 run_pilot.py

  # the real thing (needs Docker running + API keys exported)
  GROQ_API_KEY=... TOGETHER_API_KEY=... SANDBOX_MODE=docker python3 run_pilot.py

Writes results.csv with one row per (problem, model) pair.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import csv
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_completion, MODELS
from extract import extract_code
from sandbox import run_code, SANDBOX_MODE

PROBLEMS_PATH = os.path.join(os.path.dirname(__file__), "humaneval_subset.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.csv")


def load_problems():
    with open(PROBLEMS_PATH) as f:
        return [json.loads(line) for line in f]


def build_test_script(problem: dict, candidate_code: str) -> str:
    """
    Assembles: candidate function + HumanEval's own test block + a call to
    check(). This mirrors the standard HumanEval execution contract.
    """
    return (
        candidate_code
        + "\n"
        + problem["test"]
        + f"\ncheck({problem['entry_point']})\n"
    )


def run_one(problem: dict, tier: str) -> dict:
    row = {
        "task_id": problem["task_id"],
        "tier": tier,
        "model": MODELS[tier]["model"],
        "sandbox_mode": SANDBOX_MODE,
        "passed": False,
        "timed_out": False,
        "runtime_s": None,
        "error": "",
    }
    try:
        raw = get_completion(tier, problem)
        code = extract_code(raw, problem["entry_point"])
        script = build_test_script(problem, code)
        result = run_code(script)
        row["passed"] = result["passed"]
        row["timed_out"] = result["timed_out"]
        row["runtime_s"] = result["runtime_s"]
        if not result["passed"]:
            # keep it short — full stderr goes nowhere useful in a CSV cell
            row["error"] = (result["stderr"] or "")[-300:].replace("\n", " | ")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return row


def main():
    problems = load_problems()
    tiers = ["cheap", "expensive"]
    rows = []

    print(f"Sandbox mode: {SANDBOX_MODE}")
    print(f"Models: {MODELS}")
    print(f"Running {len(problems)} problems x {len(tiers)} models = {len(problems)*len(tiers)} runs\n")

    for problem in problems:
        for tier in tiers:
            t0 = time.time()
            row = run_one(problem, tier)
            rows.append(row)
            time.sleep(2)
            status = "PASS" if row["passed"] else ("TIMEOUT" if row["timed_out"] else "FAIL")
            print(f"  {problem['task_id']:>16}  {tier:>10}  {status:8}  ({time.time()-t0:.1f}s)")

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {RESULTS_PATH}")

    # quick pass-rate summary by tier
    for tier in tiers:
        tier_rows = [r for r in rows if r["tier"] == tier]
        n_pass = sum(r["passed"] for r in tier_rows)
        print(f"  {tier:>10}: {n_pass}/{len(tier_rows)} passed")


if __name__ == "__main__":
    main()