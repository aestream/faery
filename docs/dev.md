
(development)=
# Developing Faery

## Setup the environment

Local build (first run).

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # see https://rustup.rs
python3 -m venv .venv
source .venv/bin/activate
# x86 platforms may need to install https://www.nasm.us
pip install --upgrade pip
pip install maturin==1.15.0
maturin develop  # or maturin develop --release to build with optimizations
```

Local build (subsequent runs).

```sh
source .venv/bin/activate
maturin develop  # or maturin develop --release to build with optimizations
```

## Format and lint

```sh
cargo fmt
cargo clippy
pip install --group dev
ruff format
ruff check
ty check
```

## Test

```sh
pip install pytest
pytest tests
```

## Docker

The _Dockerfile_ builds the flake's dev shell (Rust, nasm, Python, uv) into an
image and installs faery with the `dev` and `bench` groups. It needs no Nix or
Rust on the host; GPU access needs the NVIDIA container toolkit.

```sh
docker build -t faery .
docker run --rm -it --gpus all faery            # shell with faery installed
docker run --rm --gpus all faery pytest tests
```

To work on a checkout, mount it at _/faery_. Every command first runs
`maturin develop --release`, so edits are rebuilt (incrementally) before the
command starts. Named volumes keep the virtual environment and build caches
out of the working copy, and `HOST_UID`/`HOST_GID` hand files the container
writes (benchmark results, generated data) back to you:

```sh
docker run --rm -it --gpus all \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -v "$PWD:/faery" -v faery-venv:/faery/.venv \
    -v faery-target:/faery/target -v faery-x264:/faery/src/mp4/x264-build \
    faery pytest tests
```

The volumes are seeded from the image the first time they are used. After
changing dependencies, rebuild the image and remove them
(`docker volume rm faery-venv faery-target faery-x264`).

## Benchmarks

_benchmarks/test_handoff.py_ times the path from decoded event packets to a
`(2, H, W)` frame on the GPU, on the real _dvs.es_ recording and on synthetic
1280x720 packets from 1k to 1M events. The `cpu` group times faery's
preparation alone; the `gpu` group adds the upload and the GPU work.

```sh
benchmarks/run.sh baseline        # runs in Docker, saves benchmarks/results/*/0001_baseline.json
# ...change something...
benchmarks/run.sh my-change
python3 benchmarks/compare.py     # Markdown table: latest run vs the previous one
python3 benchmarks/compare.py baseline my-change
```

Each saved run records the commit and whether the tree was dirty. Commit the
results and note what changed in _benchmarks/results/LOG.md_.

## Build the documentation

The docs are built with [Jupyter Book](https://next.jupyterbook.org/). The
`--execute` flag runs the tutorial notebooks (stored as jupytext Markdown in
_docs/tutorials_) so their outputs appear on the site; it requires the `docs`
dependency group, which includes `torch` and `jax`.

```sh
uv sync --group dev --group docs
cd docs && uv run jupyter book build --html --execute
```

## Upload a new version

1. Update the version in _pyproject.toml_.

2. Push the changes

3. Create a new release on GitHub. GitHub actions should build wheels and push them to PyPI.

## Update flatbuffers definitions for AEDAT

After modifying any of the files in _src/aedat/flatbuffers_, re-generate the Rust interfaces.

(Last run with flatc version 25.12.19)

```sh
flatc --rust -o src/aedat/ src/aedat/flatbuffers/*.fbs
```
