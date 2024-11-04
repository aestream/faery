![faery logo](faery_logo.png)

Faery sends event data from A to B.
It is both a command-line tool and a Python library, wrapping an optimized core written in Rust.

## Usage

1. Install pipx (https://pipx.pypa.io/stable/installation/)

2. Install faery

```sh
pipx install faery
```

## Local development

### Setup the environment

Local build (first run).

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # see https://rustup.rs
python3 -m venv .venv
source .venv/bin/activate
# x86 platforms may need to install https://www.nasm.us
pip install --upgrade pip
pip install maturin
maturin develop  # or maturin develop --release to build with optimizations
```

Local build (subsequent runs).

```sh
source .venv/bin/activate
maturin develop  # or maturin develop --release to build with optimizations
```

### Format and lint

```sh
cargo fmt
cargo clippy
pip install isort black pyright
isort .; black .; pyright .
```

### Test

```sh
pip install pytest
pytest tests
```

### Upload a new version

1. Update the version in _pyproject.toml_.

2. Push the changes

3. Create a new release on GitHub. GitHub actions should build wheels and push them to PyPI.

## Acknowledgements

Faery was initiated at the [2024 Telluride neuromorphic workshop](https://sites.google.com/view/telluride-2024/) by

-   [Alexandre Marcireau](https://github.com/amarcireau)
-   [Jens Egholm Pedersen](https://github.com/jegp)
-   [Gregor Lenz](https://github.com/biphasic)
-   [Gregory Cohen](https://github.com/gcohen)

License: LGPLv3.0
