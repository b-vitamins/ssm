# API

This document defines the public API for the SSM Core project. All modules and ops described here are considered stable once implemented. Stub signatures and contracts are enforced now by tests.

## Python Package Layout

- `ssm`
  - `modules/`
    - `mamba1.py` – Mamba v1 block
    - `mamba2.py` – Mamba v2 (SSD) block
    - `attention.py` – Optional multi-head attention block (hybrid models)
    - `block.py` – Fused add+norm wrapper with residual policy
    - `mlp.py` – Gated MLP
  - `models/`
    - `lm.py` – Causal LM wrapper (backbone + lm_head + generation mixin)
    - `config.py` – Dataclasses configuration for models
  - `ops/`
    - `python/reference.py` – Reference (pure PyTorch) op signatures
    - `cpu/*` – C++ stubs for fused CPU kernels (to be implemented)
    - `cuda/*` – CUDA stubs for fused GPU kernels (to be implemented)
  - `utils/`
    - `generation.py` – Decoding and CUDA graph integration (stubs)
    - `dispatch.py` – Backend detection and selection (lightweight)
    - `weights.py` – HF load/save helpers (stubs)

## Modules

### ssm.modules.mamba1.Mamba1

```python
class Mamba1(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        use_fast_path: bool = True,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None: ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
        **kwargs,
    ) -> torch.Tensor: ...

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype: torch.dtype | None = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

Contract
- Inputs: `hidden_states` shape `(B, L, D)` matching `d_model`.
- Outputs: tensor shape `(B, L, D)`.
- `allocate_inference_cache` returns `(conv_state, ssm_state)` sized for decoding; shapes defined in docstrings.
- Forward may raise `NotImplementedError` until kernels exist.

### ssm.modules.mamba2.Mamba2

```python
class Mamba2(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        headdim: int = 64,
        expand: int = 2,
        ngroups: int = 1,
        d_ssm: int | None = None,
        chunk_size: int = 256,
        bias: bool = False,
        conv_bias: bool = True,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None: ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor: ...

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype | None = None, **kwargs): ...
```

Contract
- Same input/output as Mamba1.
- Supports variable-length sequences via `seq_idx` and `cu_seqlens`.

### ssm.modules.block.Block

```python
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        mixer_cls: Callable[[int], nn.Module],
        mlp_cls: Callable[[int], nn.Module] | type[nn.Identity] = nn.Identity,
        norm_cls: type[nn.Module] = nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ) -> None: ...

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None, inference_params=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype: torch.dtype | None = None, **kwargs
    ) -> tuple[torch.Tensor, ...]: ...
```

Contract
- Returns `(hidden_states, residual)` using the provided mixer and optional MLP.
- Respects the `fused_add_norm` and `residual_in_fp32` flags with a Triton fast-path when
  available, otherwise executes the PyTorch fallback.
- `allocate_inference_cache` defers to the wrapped mixer implementation.

### ssm.modules.attention.MHA
Implements multi-head attention with optional grouped KV heads, depthwise convolution, rotary embeddings,
and fused gated MLP packing. Supports cache allocation and incremental decoding via `allocate_inference_cache`
and the `inference_params` protocol (matching the reference `mamba_ssm` repository).

### ssm.modules.mlp.GatedMLP
Two-layer gated MLP (SwiGLU style) with configurable hidden size rounding and activation.

## Models

### ssm.models.config.MambaConfig
```python
@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list[int] = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
```

### ssm.models.lm.MambaLMHeadModel
```python
class MambaLMHeadModel(nn.Module):
    def __init__(self, config: MambaConfig, initializer_cfg: dict | None = None, device=None, dtype=None) -> None: ...
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None, inference_params=None, num_last_tokens: int = 0, **mixer_kwargs): ...
    def generate(self, input_ids: torch.Tensor, max_length: int, top_k: int = 1, top_p: float = 0.0, min_p: float = 0.0, temperature: float = 1.0, return_dict_in_generate: bool = False, output_scores: bool = False, **kwargs): ...
    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, device=None, dtype=None, **kwargs) -> "MambaLMHeadModel": ...
    def save_pretrained(self, save_directory: str) -> None: ...
```

## Ops (Python signatures)

All ops live in `ssm.ops.python.reference` and are mirrored by CPU/CUDA backends once implemented.

```python
def selective_scan(u, delta, A, B, C, D=None, z=None, dt_bias=None, softplus=False, return_last_state=False): ...
def selective_state_step(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, softplus=True): ...
def ssd_chunk_scan(X, dt, A, B, C, chunk_size: int, D=None, z=None, seq_meta=None, initial_states=None): ...
def dw_causal_conv(x, weight, bias=None, activation: str = "silu"): ...
def fused_layer_norm(x, weight, bias, residual=None, is_rms=False, eps=1e-5, prenorm=True, residual_in_fp32=True): ...
```

Contracts
- Shapes and dtype policies are documented in docstrings. All raise `NotImplementedError` until backends exist.

## Utils
- Generation API mirrors a standard decode loop with CUDA graph support; initially raises `NotImplementedError`.
- Dispatch exposes backend detection: `has_cuda_kernels()`, `has_cpu_kernels()`, `get_available_backend()`.
