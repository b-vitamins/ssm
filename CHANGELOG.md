# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- CUDA fused layer norm/RMSNorm backward kernel with autograd dispatch,
  expanded golden fixtures for prenorm/postnorm and residual_in_fp32 variants,
  and parity tests that exercise bias and residual gradients.
- CUDA depthwise causal convolution backward kernel with activation-aware
  gradient support, refreshed golden fixtures across layouts, and parity tests.
- CUDA SSD chunk scan backward kernel covering ragged metadata, gating, and
  initial-state propagation with refreshed CUDA goldens and parity tests.
- CUDA selective state step backward kernel with optional projection/gating
  support, autograd dispatch, and refreshed golden fixtures covering broadcast
  and grouped parameter layouts.
- CUDA selective scan backward kernel with grouped/time-varying parameter
  support, autograd dispatch, and golden-backed parity tests.
- Golden-backed CUDA parity tests that load serialized upstream outputs and
  gradients for selective scan, selective state step, SSD chunk scan, depthwise
  causal convolution, and fused layer norm, plus a regeneration script pinned to
  the upstream Mamba commit for generating local JSON references.
- CPU selective scan, selective state step, and SSD chunk scan backward kernels
  with autograd regression tests validating gradients against the reference
  implementation.
- CPU SSD chunk scan backward kernel with Torch dispatch and Python autograd
  integration mirroring the forward fused path.
- CPU selective scan backward kernel with Torch operator registration and
  Python autograd dispatch to the compiled implementation.
- Benchmark harness under `benchmarks/ops_bench.py` mirroring upstream Mamba
  workloads with docs, optional extras, and sample output for parity tracking.
- CUDA SSD chunk scan GPU regression coverage for ragged cu_seqlens,
  initial-state plumbing, and a performance smoke harness.
- Autograd-enabled CUDA dispatch for selective scan, state step, SSD chunk scan,
  depthwise causal convolution, and fused layer norm with reference-validated
  gradients and expanded GPU tests.
- SSD chunk scan stress coverage comparing the fused CPU kernel against the
  reference path across long and ragged sequence configurations.
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
- Consolidated the CUDA parity goldens behind `refresh_mamba_goldens.py`,
  teaching the CUDA parity tests to stream data from `tests/mamba_reference_cases.json`
  when present and otherwise skip with guidance to regenerate locally.
- Replaced the CUDA fused layer norm with a block-reduction kernel mirroring
  the upstream implementation, covering RMSNorm/LayerNorm fusion, mixed
  precision, and residual-in-fp32 handling with new CUDA unit tests.
- Replaced the CUDA depthwise causal convolution with a shared-memory kernel
  matching the upstream tiling strategy, aligned activation handling with the
  reference path, and expanded GPU tests to cover multiple kernel sizes and
  activations.
- Rebuilt the CUDA SSD chunk scan to launch chunk-level fused kernels with
  shared-memory staging, vectorized gating, and ragged sequence support aligned
  with the upstream implementation.
- Replaced the CUDA selective scan path with the upstream parallel kernel,
  adding grouped-parameter coverage and gradient consistency tests.
- Replaced the CUDA selective state step with a fused kernel mirroring the
  upstream tiling strategy, fixing inference-time state updates and expanding
  grouped/gated gradient coverage.
- Optimized the CPU fused layer norm kernel with vectorized reductions and
  optional residual fusion paths aligned across RMSNorm and LayerNorm.
- Optimized the CPU depthwise causal convolution to reuse PyTorch's grouped
  convolution primitives with fused activation handling for mixed precision.
- Reworked the CPU SSD chunk scan kernel with chunk-level threading, contiguous
  traversal, and vectorized math aligned with the upstream chunk-tiling logic.
- Optimized the CPU selective state step kernel with vectorized inner loops and grouped projection row views to match the upstream fused update semantics.
- Documented the required developer workflow to install dependencies on session start and to run `ruff`, `pyright`, and `pytest` before opening pull requests.
- Expanded the contributor workflow to mandate running `ruff format .` before linting and tests, aligning local practice with CI.
- Simplified `scripts/regenerate_goldens.py` to rely on the editable install instead of mutating `sys.path`.
- Added NumPy as a core dependency so Torch can initialize its NumPy bridge without warnings.
- Standardized the CI pipeline on Python 3.11 to reduce redundant matrix executions.
- Strengthened AGENTS.md instructions to enforce proactive CHANGELOG updates ahead of every PR.
- Audited the Mamba reference blocks to perform fp32 accumulations with fewer layout conversions and documented the layout policy in `docs/DESIGN.md`.
- Reimplemented the CPU selective scan, state step, SSD chunk scan, depthwise causal convolution, and fused layer norm kernels
  with fused inner loops, grouped/time-varying parameter support, and explicit autograd regression coverage.
- Refreshed `scripts/bench_cpu.py` and golden fixtures to exercise ragged and grouped workloads while reporting CPU speedups over the reference path.
### Fixed
- Corrected top-p nucleus sampling to retain the threshold token and added a regression test guarding against regressions.
- Tightened the decode streamer typing to satisfy static analysis and prevent CI regressions.
- Prevented fused attention Blocks from double-counting residuals and aligned depthwise attention cache initialization with
  incremental decoding semantics.

## [0.1.0.dev0] - 2025-10-24
### Added
- Design-first repository skeleton, no implementations by design.

