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

