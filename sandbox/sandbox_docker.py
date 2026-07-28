"""
Real sandbox: runs a Python snippet inside a throwaway Docker container.

Minimal by design (per pilot scope):
  - one-shot container, auto-removed (--rm)
  - no network (--network none)
  - CPU/memory capped, wall-clock timeout enforced from the host side
  - stdout/stderr captured, exit code used for pass/fail

NOT done yet (do this before the sandbox handles untrusted code from anyone
other than model output you're grading in a closed pilot):
  - seccomp/apparmor profiles
  - read-only root filesystem + tmpfs scratch
  - dropped capabilities (--cap-drop ALL)
  - non-default user namespace remapping
"""
import subprocess
import tempfile
import os
import time

IMAGE_NAME = "humaneval-sandbox"
DEFAULT_TIMEOUT_S = 10


def build_image_if_needed():
    check = subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME],
        capture_output=True,
    )
    if check.returncode != 0:
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, here],
            check=True,
        )


def run_code(code: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    """
    Runs `code` in an isolated container.
    Returns: {"passed": bool, "stdout": str, "stderr": str, "timed_out": bool, "runtime_s": float}
    """
    build_image_if_needed()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "256m",
        "--cpus", "1",
        "-v", f"{script_path}:/home/runner/prog.py:ro",
        IMAGE_NAME,
        "/home/runner/prog.py",
    ]

    start = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + "\n[TIMEOUT]"
        returncode = -1
    finally:
        os.unlink(script_path)

    runtime_s = time.time() - start
    return {
        "passed": (returncode == 0) and not timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "runtime_s": round(runtime_s, 2),
    }