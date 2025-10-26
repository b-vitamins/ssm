# Roadmap

This roadmap outlines milestones for implementing SSM Core on top of the scaffolded design.

## Phase 0 — Scaffolding (this repo state)
- Stub docs: API, DESIGN, ROADMAP.
- Package skeleton with strict signatures and Google-style docstrings.
- Tests for API conformance, contracts, dispatch, and golden harness.

## Phase 1 — Python Reference
- Implement `reference.py` ops:
  - selective_scan (correctness-first, possibly slower)
  - selective_state_step
  - ssd_chunk_scan (chunked, varlen support)
  - dw_causal_conv (depthwise, causal, fused activation)
  - fused_layer_norm
- Wire modules to reference ops.
- Enable end-to-end generation using reference ops.
- Add streaming decode helpers, repetition penalties, and CUDA graph hooks to
  the reference generation stack.
- Generate and commit small golden datasets from reference for core op cases.

## Phase 2 — CPU Fused Kernels
- Implement fused CPU kernels per op with threading and vectorization.
- Integrate with dispatch; ensure autograd coverage. **Done** via dedicated CPU
  backward kernels for selective scan, selective state step, and SSD chunk scan;
  follow-up is expanding gradient performance benchmarks across ragged metadata
  and grouped parameter sweeps.
- Add performance smoke tests and microbenchmarks.

## Phase 3 — CUDA Kernels
- Implement pure CUDA kernels for core ops; add CUTLASS optionally for dense segments.
- Ensure CUDA graph capture and amp compatibility.
- Varlen support and step-update paths.
- Integrate autograd wrappers that validate fused CUDA kernels against the Python
  reference implementations with gradient tests.

## Phase 4 — Optimization & Stability
- Precision refinements (master fp32 where needed).
- Memory layout tuning to reduce transposes and improve cache/SM utilization.
- Extended test coverage: long sequences, extreme d_state, grouping, mixed precision.

## Phase 5 — Ecosystem & Tooling
- HF integration paths (`from_pretrained`, `save_pretrained`).
- Doc build and examples notebooks.
- CI wheels for common Torch/CUDA pairs.

