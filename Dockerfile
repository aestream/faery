# Faery development image, built from the Nix flake dev shell.
#
#   docker build -t faery .
#   docker run --rm -it faery                   # shell with faery installed
#   docker run --rm --gpus all faery pytest tests
#
# See docs/dev.md for mounting a working copy (rebuilds on every run) and for
# running the GPU benchmarks.

FROM nixos/nix:2.28.3

ENV NIX_CONFIG="experimental-features = nix-command flakes"
# NVIDIA's container runtime injects the host driver (libcuda.so) into one of
# these; Nix's glibc ignores ld.so.cache, so point the loader at them directly.
ENV LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /faery

# Realize the dev shell's dependencies (Rust, nasm, Python, uv, ...) in their
# own layer so source changes don't re-download the toolchain. The shell is
# always entered from this copy of the flake: a mounted working copy is a git
# repository owned by the host user, which Nix refuses to evaluate as root.
# The profile is a GC root, so nix-collect-garbage below keeps the toolchain.
COPY flake.nix flake.lock /opt/faery-flake/
RUN nix print-dev-env /opt/faery-flake \
    --profile /nix/var/nix/profiles/faery-dev > /dev/null

# Entering the shell runs its hooks: create .venv, install the dev group,
# patch ELF interpreters, and `maturin develop --release`. Hook failures don't
# fail `nix develop`, hence the explicit import check.
COPY . .
RUN nix develop /opt/faery-flake --command python -c "import faery.extension" \
    && nix develop /opt/faery-flake --command uv pip install --group bench \
    && nix-collect-garbage

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
