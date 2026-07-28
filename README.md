# HumanEval mini pilot

Validates the full loop end-to-end before scaling: 20 problems x 2 models,
sandboxed execution, CSV log, hand-eyeball the matrix.

## Layout
pilot/
setup_data.py run once to generate humaneval_subset.jsonl
humaneval_subset.jsonl 20 problems, stratified by solution complexity
models.py Groq + Together clients, plus MOCK_MODELS=1 for offline testing
extract.py pulls code out of raw model text (strips fences)
sandbox/
Dockerfile minimal unprivileged container, no network
sandbox_docker.py real sandbox: one-shot docker run --rm
sandbox_subprocess.py dev-mode fallback, no Docker needed — testing only
init.py picks docker vs subprocess via SANDBOX_MODE env
run_pilot.py harness: loop problems x models, log results.csv
results.csv output (generated)


## Setup (Windows / PowerShell)

1. Install Python 3 (check "Add to PATH" during install) and Docker Desktop.
2. Get API keys from https://console.groq.com and https://api.together.xyz
3. In the pilot folder:
```powershell
pip install requests
python setup_data.py
```

## Dry run first (no Docker, no API keys)

```powershell
$env:MOCK_MODELS="1"
$env:SANDBOX_MODE="subprocess"
python run_pilot.py
```
Should print 40 PASS/FAIL lines and write results.csv. Confirms your setup is correct before spending API credits.

## Running for real

Make sure Docker Desktop is open and running, then:
```powershell
$env:GROQ_API_KEY="your_key_here"
$env:TOGETHER_API_KEY="your_key_here"
$env:SANDBOX_MODE="docker"
Remove-Item Env:\MOCK_MODELS
python run_pilot.py
```

First run builds the sandbox Docker image once, then loops through 40 (problem, model) pairs.

## View results

Open `results.csv` in Excel, or pivot it in PowerShell:
```powershell
python -c "import csv; rows=list(csv.DictReader(open('results.csv'))); tasks=sorted(set(r['task_id'] for r in rows), key=lambda x:int(x.split('/')[1])); by={(r['task_id'],r['tier']):r['passed'] for r in rows}; [print(t, by[(t,'cheap')], by[(t,'expensive')]) for t in tasks]"
```

## Known gaps (intentional, per pilot scope)

- Sandbox is **not hardened**: no seccomp/apparmor, no read-only rootfs, no capability dropping. Fine for grading your own model output in a closed pilot; not fine for arbitrary untrusted code from strangers.
- No Postgres — CSV only, as scoped.
- No retries/backoff on API calls — a transient failure shows up as a FAIL row with an error message.
- No concurrency — sequential loop, fine at this scale.
- Code extraction is regex-based fence-stripping — check the `error` column if pass rates look surprisingly low.

## Next steps if the pilot looks good

- Harden the Docker sandbox before running anyone else's code through it.
- Scale to the full 164 problems.
- Add Postgres once you're tracking runs over time / across more models.
- Parallelize the harness loop.