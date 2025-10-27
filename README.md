SSM Core (Design-First, PyTorch-First)

This repository is a clean, implementation-agnostic scaffold for building high‑performance Selective State Space Models (Mamba‑style) with:

- PyTorch‑first APIs and minimal dependencies.
- Optional GPU acceleration via pure CUDA kernels (CUTLASS optional), never required.
- First‑class CPU with fused, optimized kernels (to be implemented).
- Clear module boundaries, strict contracts, and comprehensive tests.

Start with the docs in `docs/` for the full design and APIs, then follow the ROADMAP for implementation phases.

Quick start
- Run tests (API/contract only): `pytest -q`
- Run golden tests: `pytest -q --run-goldens` (placeholders until replaced)

### CUDA parity goldens

The CUDA parity suite compares local kernels against tensors generated with the
official Mamba CUDA implementation. To refresh the fixtures on a GPU machine:

1. Clone the upstream repository next to this project and pin the commit::

       git clone https://github.com/state-spaces/mamba.git external/mamba_upstream
       (cd external/mamba_upstream && git checkout 10b5d6358f27966f6a40e4bf0baa17a460688128)

   Building the upstream CUDA extensions requires a CUDA toolkit with `nvcc`
   available. If your environment lacks a full toolkit, regenerate the fixtures
   on a development workstation or GPU runner that provides it.

2. Regenerate the serialized tensors with the upstream backend::

       python scripts/refresh_mamba_goldens.py \
           --mamba-repo external/mamba_upstream \
           --device cuda:0 \
           --output tests/mamba_reference_cases.json

   The script captures deterministic forwards and gradients across the CUDA
   kernels and writes a single JSON file that the parity tests consume when it
   is present. Drop the generated `tests/mamba_reference_cases.json` into your
   local checkout before running the CUDA parity suite. The path is git-ignored,
   so you can regenerate as needed without touching version control.

Benchmarks
- Install optional dependencies: `python -m pip install -e .[bench]`
- Compare Python vs. compiled kernels: `python -m benchmarks.ops_bench --device cpu --iters 1`
- Pass `--device cuda` on GPU machines or `--json results.json` to persist timings for parity tracking.
- Representative CPU output (with fused kernels unavailable) looks like:

  ```
  == selective_scan :: prefill_2k_d512 :: device=cpu ==
  | backend   | time (ms)   | vs python   | notes       |
  |-----------|-------------|-------------|-------------|
  | python    | 1012.90     | 1.00x       |             |
  | cpu       | n/a         | n/a         | unavailable |
  | cuda      | n/a         | n/a         | unavailable |
  | dispatch  | 1011.69     | 1.00x       |             |
  ```

  The harness mirrors the upstream `state-spaces/mamba` selective-scan and chunked
  scan workloads so results can be compared directly while we close the parity gap.

