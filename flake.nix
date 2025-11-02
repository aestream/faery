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
            pypkgs = pkgs.python312Packages;
          in
          pkgs.mkShell {
            buildInputs = [
              pypkgs.python
              pypkgs.venvShellHook
              pkgs.autoPatchelfHook
              pkgs.bash
              pkgs.cargo
              pkgs.rustc
              pkgs.rustPlatform.bindgenHook
              pkgs.libgcc
              pkgs.libz
              pkgs.nasm
              pkgs.nodejs
              pkgs.uv
              # GUI libraries
              pkgs.fontconfig
              pkgs.libGL
              pkgs.libxkbcommon
              pkgs.wayland
            ];
            venvDir = "./.venv";
            propagatedBuildInputs = [
              pkgs.stdenv.cc.cc.lib
            ];
            postVenvCreation = ''
              uv pip install --group dev .
              autoPatchelf ./.venv
            '';
            postShellHook = ''
              export AS="nasm" # build assembly optimizations in x264
              export CC="gcc" # use gcc to compile x264
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
                pkgs.libGL pkgs.libxkbcommon pkgs.wayland pkgs.fontconfig
              ]}:$LD_LIBRARY_PATH"
              source .venv/bin/activate
              maturin develop --release
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
