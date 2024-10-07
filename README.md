![faery logo](faery_logo.png)

Faery sends event data from A to B.
It is both a command-line tool and a Python library, wrapping an optimized core written in Rust.

## Local development

### Setup the environment

Local build (first run).

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # see https://rustup.rs
python3 -m venv .venv
source .venv/bin/activate
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

```
cargo fmt
cargo clippy
isort .; black .; pyright .
```

## Acknowledgements

Faery was initiated at the [2024 Telluride neuromorphic workshop](https://sites.google.com/view/telluride-2024/) by

-   [Alexandre Marcireau](https://github.com/amarcireau)
-   [Jens Egholm Pedersen](https://github.com/jegp)
-   [Gregor Lenz](https://github.com/biphasic)
-   [Gregory Cohen](https://github.com/gcohen)

License: LGPLv3.0
