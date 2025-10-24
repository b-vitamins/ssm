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
- `Block` handles residual policy + fused normalization, falling back to PyTorch when Triton kernels are absent.
- `MHA` mirrors the upstream Mamba reference with grouped KV heads, optional depthwise conv warmup, rotary embeddings,
  and a fused gated-MLP projection path.
- `GatedMLP` provides the SwiGLU-style feed-forward used by both the attention module and SSM blocks.
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

### Layout & Accumulation Policy

- Mamba blocks minimise redundant layout conversions by keeping depthwise-convolution buffers in channel-first layout where the downstream ops benefit, materialising channel-last views only when projections require them.
- State-space projections are evaluated in fp32 via explicit casting before `torch.exp`, `torch.einsum`, or chunked scans to preserve numerical stability when modules run under autocast/bfloat16.
- Linear layers that must consume channel-last tensors cast their inputs to the parameter dtype before the projection, while the surrounding math continues in fp32. Outputs are converted back to the caller's dtype after the final projection to preserve module-level dtype semantics.

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

