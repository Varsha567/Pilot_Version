"""
Thin clients for the two providers. Both Groq and Together expose an
OpenAI-compatible /chat/completions endpoint, so one function covers both —
only the base_url, api_key, and model name differ.

Set real API keys as env vars before running for real:
  export GROQ_API_KEY=...
  export TOGETHER_API_KEY=...

MOCK MODE: if MOCK_MODELS=1 is set, no network calls are made — a canned
generator returns a plausible-looking (but not necessarily correct)
completion. Used to test the harness end-to-end without API keys / network.
"""

import os
import json
import time
import requests

MOCK = os.environ.get("MOCK_MODELS") == "1"

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
    },
}

# Both tiers on Groq now (free tier, no Together billing needed).
# gpt-oss-20b = small/fast, gpt-oss-120b = bigger/smarter — same family,
# so it's a fair "cheap vs expensive" comparison within one provider.
MODELS = {
    "cheap": {"provider": "groq", "model": "openai/gpt-oss-20b"},
    "expensive": {"provider": "groq", "model": "openai/gpt-oss-120b"},
}


def _build_prompt(problem: dict) -> str:
    return (
        "Complete the following Python function. Return ONLY the full "
        "function implementation (including the signature), no prose, "
        "no markdown fences.\n\n" + problem["prompt"]
    )


def _mock_completion(problem: dict, tier: str) -> str:
    # Cheap deterministic stand-in: returns the real canonical solution most
    # of the time and a deliberately broken one occasionally, so the pass/fail
    # matrix has visible variance when testing the harness. "expensive" gets a
    # higher hit rate than "cheap" so the two columns actually diverge.
    import random
    random.seed(problem["task_id"] + tier)
    threshold = 0.85 if tier == "expensive" else 0.65
    if random.random() < threshold:
        return problem["prompt"] + problem["canonical_solution"]
    else:
        return problem["prompt"] + "    return None  # mock: intentionally wrong\n"


def get_completion(tier: str, problem: dict, timeout_s: int = 30) -> str:
    """tier is 'cheap' or 'expensive'. Returns raw model text (not yet parsed)."""
    if MOCK:
        return _mock_completion(problem, tier)

    cfg = MODELS[tier]
    provider = PROVIDERS[cfg["provider"]]
    api_key = os.environ.get(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Missing {provider['api_key_env']} — set it before running for real, "
            f"or export MOCK_MODELS=1 to test without API access."
        )

    max_retries = 4
    for attempt in range(max_retries):
        resp = requests.post(
            provider["base_url"],
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": cfg["model"],
                "messages": [{"role": "user", "content": _build_prompt(problem)}],
                "temperature": 0,
                "max_tokens": 1024,
            },
            timeout=timeout_s,
        )
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)  # 5s, 10s, 15s, 20s
            print(f"    rate limited, waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError("Gave up after repeated 429 rate-limit responses")