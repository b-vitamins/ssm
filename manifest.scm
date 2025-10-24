;; Guix manifest for SSM development
(specifications->manifest
 (list
  "python-pytorch-cuda"
  "python-einops"
  "python-pytest"
  "python-numpy@1"
  "python-ruff"
  "node-pyright"
  "gcc-toolchain@14"
  "cuda-toolkit"
  "cutlass-headers"
  "pybind11"
  "cmake"
  "ninja"))
