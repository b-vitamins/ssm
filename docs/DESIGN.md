# Design

This document describes the architecture and rationale behind the SSM Core project. The design aims to provide a clean, maintainable foundation with optional acceleration and robust CPU performance.

## Goals

- PyTorch-first: idiomatic modules and ops, easy to compose.
- Optional acceleration: pure CUDA kernels (CUTLASS optional), never required.
- First-class CPU: fused, optimized kernels to ensure strong performance without GPUs.
- Minimal core surface: a small set of fused ops powering Mamba1/Mamba2.
- Clean separation of concerns: modules compose ops; ops provide autograd surfaces and dispatch to backends.

## Architecture

### Layers and Composition

- Modules (Mamba1, Mamba2, MHA, MLP) implement model logic using ops.
- `Block` handles residual policy + fused normalization.
- `MambaLMHeadModel` builds an LM backbone and exposes generation.

### Ops and Backends

Each op has three levels:

1) Python Reference (pure PyTorch) — canonical spec; no kernels required.
2) Fused CPU (C++): vectorized loops with OpenMP (or at::parallel_for), accumulating in fp32.
3) CUDA (C++/CUDA): kernels tuned for occupancy and memory throughput; CUTLASS optional for dense matmul segments.

Dispatch selects the best available backend at runtime.

### Autograd

Ops expose a single `torch.autograd.Function`-backed surface per op. Different backends are implemented under that surface. Meta (shape-only) functions are provided to support `torch.compile` and shape checking.

### Precision Policy

- Parameters impacting recurrent stability (e.g., A, D, dt_bias) are stored/accumulated in fp32; outputs returned in the requested mixed precision.
- Exponentials and softplus performed in fp32.

### Variable Length Handling (Mamba2/SSD)

- Varlen inputs represented via `seq_meta` (e.g., `seq_idx`, `cu_seqlens`, masks). Kernels interpret these to process ragged batches.

### Build

- CPU extension always attempted; CUDA conditionally built if a toolchain is present.
- No dynamic wheel fetching; a simple and predictable build.

## Why This Design

- Simplicity: fewer moving parts than Triton-based pipelines; one autograd surface per op.
- Portability: CPU is robust and fast enough for many environments.
- Maintainability: smaller kernel set, consistent dispatch, and strict contracts reduce tech debt.

## Security and Reliability

- Reference implementation serves as the ground truth for correctness and golden generation.
- Tests enforce contracts, docstring/documentation presence, and shape/grad behavior (when implemented).

