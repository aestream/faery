#!/usr/bin/env bash
# Run a command inside the flake's dev shell (the copy baked into the image;
# rebuild the image after changing flake.nix). The shell hook re-runs
# `maturin develop --release`, so a mounted working copy is rebuilt
# (incrementally) before the command starts.
set -euo pipefail
cd /faery

status=0
nix develop /opt/faery-flake --command "$@" || status=$?

# Files written into a bind-mounted working copy are owned by root; hand them
# back to the host user when HOST_UID is given (see docs/dev.md).
if [[ -n "${HOST_UID:-}" ]]; then
    find /faery -xdev \
        \( -path /faery/.venv -o -path /faery/target -o -path /faery/src/mp4/x264-build \) -prune \
        -o -user 0 -exec chown "${HOST_UID}:${HOST_GID:-$HOST_UID}" {} + 2>/dev/null || true
fi
exit "$status"
