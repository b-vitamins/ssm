# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- Hugging Face Hub checkpoint resolution backed by ``cached_file`` with dtype
  remapping support, updated language model helpers, docs, and unit tests.
- Expanded generation helpers with reusable inference parameter management,
  streaming hooks, repetition penalties, teacher forcing, and CUDA graph options
  integrated into ``MambaLMHeadModel.generate``.
- Enabled cached prefill/decode flows for the reference Mamba1 and Mamba2 blocks with
  regression tests covering dense and ragged stepping.
- Mixer-style language model backbone with optional attention, Hugging Face
  save/load helpers, updated docs, and expanded generation tests.
- Implemented the hybrid block, multi-head attention (with cache/conv support), and gated MLP modules with docs and unit tests.
- Initial scaffold with docs (API, DESIGN, ROADMAP), package layout under `src/ssm/`, and comprehensive test suite (contracts + goldens harness).
- Contributor guide (AGENTS.md) with Conventional Commits and CHANGELOG policy.
- Minimal GRU-backed `MambaLMHeadModel` forward pass wired for cache-aware decoding.
- Generation helpers plus greedy/top-k/top-p/min-p sampling integration with coverage in CPU tests.
- CPU fused kernels for the core ops with dispatch, smoke tests, and benchmarking harness.
- CUDA kernels and bindings for selective scan, selective state step, SSD chunk scan, depthwise causal convolution, and fused layer norm with GPU-dispatch integration and CUDA-focused tests.
- Stress tests that cover long sequences, high state dimensions, grouped parameters, and mixed-precision execution paths under `tests/stress/`.
### Changed
- Documented the required developer workflow to install dependencies on session start and to run `ruff`, `pyright`, and `pytest` before opening pull requests.
- Expanded the contributor workflow to mandate running `ruff format .` before linting and tests, aligning local practice with CI.
- Simplified `scripts/regenerate_goldens.py` to rely on the editable install instead of mutating `sys.path`.
- Added NumPy as a core dependency so Torch can initialize its NumPy bridge without warnings.
- Standardized the CI pipeline on Python 3.11 to reduce redundant matrix executions.
- Strengthened AGENTS.md instructions to enforce proactive CHANGELOG updates ahead of every PR.
- Audited the Mamba reference blocks to perform fp32 accumulations with fewer layout conversions and documented the layout policy in `docs/DESIGN.md`.
### Fixed
- Corrected top-p nucleus sampling to retain the threshold token and added a regression test guarding against regressions.
- Tightened the decode streamer typing to satisfy static analysis and prevent CI regressions.
- Prevented fused attention Blocks from double-counting residuals and aligned depthwise attention cache initialization with
  incremental decoding semantics.

## [0.1.0.dev0] - 2025-10-24
### Added
- Design-first repository skeleton, no implementations by design.

