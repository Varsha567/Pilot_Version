import re


def extract_code(raw: str, entry_point: str) -> str:
    """
    Best-effort extraction of a Python function body/definition from raw
    model text. Handles the common cases: fenced code blocks, or plain text
    that's already just code.
    """
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    code = fence.group(1) if fence else raw

    # If the entry point function isn't defined anywhere, we still return
    # what we have — it'll just fail the sandbox run, which is correct signal.
    return code.strip() + "\n"