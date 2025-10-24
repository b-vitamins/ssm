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

