import os

SANDBOX_MODE = os.environ.get("SANDBOX_MODE", "docker")  # "docker" or "subprocess"

if SANDBOX_MODE == "docker":
    from .sandbox_docker import run_code
else:
    from .sandbox_subprocess import run_code

__all__ = ["run_code", "SANDBOX_MODE"]