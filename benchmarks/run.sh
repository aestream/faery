#!/usr/bin/env bash
# Run the GPU handoff benchmarks in the faery Docker image and save the results.
#
#   benchmarks/run.sh <label> [extra pytest args...]
#
# The working copy is mounted into the container and rebuilt (incrementally)
# before the run, so this always measures the current source tree. Results go
# to benchmarks/results/<machine>/NNNN_<label>.json, tagged with the commit and
# a dirty flag; compare two runs with benchmarks/compare.py.
set -euo pipefail

label="${1:?usage: benchmarks/run.sh <label> [pytest args...]}"
shift
# BENCH_REPO measures another checkout (e.g. a worktree at an older commit).
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="${BENCH_REPO:-$here}"

# Concurrent builds skew results by 2x or more; refuse to run on a busy machine.
max_load="${BENCH_MAX_LOAD:-4}"
load="$(cut -d' ' -f1 /proc/loadavg)"
if awk -v l="$load" -v m="$max_load" 'BEGIN { exit !(l > m) }'; then
    echo "load average is $load (> $max_load); wait for other jobs or set BENCH_MAX_LOAD" >&2
    exit 1
fi

# One build cache per checkout: every checkout is mounted at /faery, and cargo's
# mtime-based freshness check would otherwise reuse another checkout's build.
cache="$(printf '%s' "$repo" | sha1sum | cut -c1-8)"

dirty=0
[[ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]] && dirty=1

# Pin to a fixed set of cores so runs are comparable (BENCH_CPUS to override).
docker run --rm --gpus all --cpuset-cpus="${BENCH_CPUS:-2-5}" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -e BENCH_COMMIT="$(git -C "$repo" rev-parse HEAD)" \
    -e BENCH_BRANCH="$(git -C "$repo" branch --show-current)" \
    -e BENCH_DIRTY="$dirty" \
    -v "$repo:/faery" \
    -v "$here/benchmarks/results:/faery/benchmarks/results" \
    -v faery-venv:/faery/.venv \
    -v "faery-target-$cache:/faery/target" \
    -v "faery-x264-$cache:/faery/src/mp4/x264-build" \
    faery \
    pytest benchmarks \
    --benchmark-storage=file://benchmarks/results \
    --benchmark-save="$label" \
    --benchmark-group-by=group,param:workload \
    --benchmark-columns=median,iqr,min,rounds \
    --benchmark-sort=name \
    "$@"
