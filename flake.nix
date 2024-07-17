{
  description = "Ferries AER data from inputs to outputs";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        faery =
          let
            pypkgs = pkgs.python3Packages;
          in
          pkgs.mkShell {
            buildInputs = [
              pypkgs.python
              pypkgs.numpy
              pypkgs.sympy
              pypkgs.venvShellHook
              pypkgs.torch
              pkgs.autoPatchelfHook
              pkgs.rustc
              pkgs.cargo
            ];
            venvDir = "./.venv";
            propagatedBuildInputs = [
              pkgs.stdenv.cc.cc.lib
            ];
            postVenvCreation = ''
              pip install -U pip setuptools wheel pytest
              pip install -e .
              autoPatchelf ./.venv
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
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
