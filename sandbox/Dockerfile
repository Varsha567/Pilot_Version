FROM python:3.11-slim

# Minimal, unprivileged runner. NOT hardened (per pilot scope) —
# no seccomp/apparmor tuning, no network egress rules beyond Docker defaults,
# no read-only rootfs. Good enough to get pass/fail signal, not to run untrusted
# code from strangers. Harden before this becomes anything more than a pilot.

RUN useradd -m runner
USER runner
WORKDIR /home/runner

# code is mounted at runtime, not baked in, so we don't rebuild the image per problem
ENTRYPOINT ["python3", "-I"]