
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
pip install maturin==1.9.3
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
pip install isort black pyright
isort .; black .; pyright .
```

## Test

```sh
pip install pytest
pytest tests
```

## Upload a new version

1. Update the version in _pyproject.toml_.

2. Push the changes

3. Create a new release on GitHub. GitHub actions should build wheels and push them to PyPI.

## Update flatbuffers definitions for AEDAT

After modifying any of the files in _src/aedat/flatbuffers_, re-generate the Rust interfaces.

(Last run with flatc version 25.1.24)

```sh
flatc --rust -o src/aedat/ src/aedat/flatbuffers/*.fbs
```
