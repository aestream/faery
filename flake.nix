{
  description = "Ferries AER data from inputs to outputs";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        faery =
          let
            pypkgs = pkgs.python313Packages;
          in
          pkgs.mkShell {
            buildInputs = [
              pypkgs.python
              pypkgs.venvShellHook
              pkgs.autoPatchelfHook
              pkgs.bash
              pkgs.rustup
              pkgs.rustPlatform.bindgenHook
              pkgs.libgcc
              pkgs.libz
              pkgs.nasm
            ];
            venvDir = "./.venv";
            propagatedBuildInputs = [
              pkgs.stdenv.cc.cc.lib
            ];
            postVenvCreation = ''
              pip install -U pip maturin pytest isort black pyright
            '';
            postShellHook = ''
              rustup default stable
              export AS="nasm" # build assembly optimizations in x264
              export CC="gcc" # use gcc to compile x264
              maturin develop --release
              autoPatchelf ./.venv
            '';
          };
      in
      rec {
        devShells = flake-utils.lib.flattenTree {
          default = faery;
        };
        packages = flake-utils.lib.flattenTree {
          default = faery;
        };
      }
    );
}
