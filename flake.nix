{
  description = "Behavioral fraud detection";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux"; # change to aarch64-darwin etc. if needed
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python312
          pkgs.uv
          pkgs.git
        ];

        # pip/uv wheels (numpy, pandas, torch) need these libs on NixOS
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ];

        shellHook = ''
          export UV_PYTHON=${pkgs.python312}/bin/python
          export UV_PYTHON_DOWNLOADS=never
          [ -d .venv ] && source .venv/bin/activate
          echo "fraud-detection dev shell ready"
        '';
      };
    };
}
