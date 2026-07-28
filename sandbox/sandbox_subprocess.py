"""
Dev-mode fallback sandbox: same function signature/return shape as
sandbox_docker.run_code, but runs the code as a plain subprocess with a
resource limit + timeout instead of inside a container.

This is NOT isolation. It exists only so the harness can be built and
tested somewhere without a Docker daemon. Do not use this mode to grade
untrusted model output for real — swap in sandbox_docker before running
the actual pilot.

NOTE for Windows: the `resource` module used below is Unix-only and will
fail to import on native Windows Python. If you're running natively on
Windows (not inside WSL) and want to use this fallback mode, see the
Windows note in the README. It's not needed at all if you're using
SANDBOX_MODE=docker, which is the real path anyway.
"""
import subprocess
import tempfile
import os
import time
import sys

DEFAULT_TIMEOUT_S = 10
MEMORY_LIMIT_BYTES = 256 * 1024 * 1024

try:
    import resource
    HAVE_RESOURCE = True
except ImportError:
    HAVE_RESOURCE = False


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def run_code(code: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name

    start = time.time()
    timed_out = False
    try:
        kwargs = {}
        if HAVE_RESOURCE:
            kwargs["preexec_fn"] = _limit_resources
        proc = subprocess.run(
            [sys.executable, "-I", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            **kwargs,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or ""
        stderr = ((e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr) or "") + "\n[TIMEOUT]"
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