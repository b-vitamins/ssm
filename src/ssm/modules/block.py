from __future__ import annotations

from typing import Callable, Optional, Tuple, TypeAlias, cast

import torch
from torch import nn


try:  # pragma: no cover - optional dependency for fused norms
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - best effort import
    RMSNorm = None  # type: ignore[assignment]
    layer_norm_fn = None


LayerNormFn: TypeAlias = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


def _norm_weight_dtype(norm: nn.Module, fallback: torch.dtype) -> torch.dtype:
    weight = getattr(norm, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.dtype
    return fallback


def _is_rms_norm(norm: nn.Module) -> bool:
    return RMSNorm is not None and isinstance(norm, RMSNorm)


class Block(nn.Module):
    """Wrapper that fuses residual add + (RMS)LayerNorm and applies a mixer then optional MLP.

    Args:
        dim: Model dimension of input and output.
        mixer_cls: Callable that creates the mixer module given ``dim``.
        mlp_cls: Callable (or :class:`torch.nn.Identity`) that creates the MLP given ``dim``.
        norm_cls: Normalization layer class, e.g. :class:`torch.nn.LayerNorm`.
        fused_add_norm: If ``True`` and Triton kernels are available, use a fused add+norm.
        residual_in_fp32: If ``True``, converts the residual tensor to fp32 for stability.

    Raises:
        RuntimeError: If fused add+norm is requested but the Triton kernels are unavailable.
        TypeError: If fused add+norm is requested with an unsupported normalization layer.
    """

    def __init__(
        self,
        dim: int,
        mixer_cls: Callable[[int], nn.Module],
        mlp_cls: Callable[[int], nn.Module] | type[nn.Identity] = nn.Identity,
        norm_cls: type[nn.Module] = nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mixer = mixer_cls(dim)
        if mlp_cls is nn.Identity:
            self.mlp = nn.Identity()
            self.norm2: Optional[nn.Module] = None
        else:
            self.mlp = mlp_cls(dim)
            self.norm2 = norm_cls(dim)
        self.norm = norm_cls(dim)
        self.residual_in_fp32 = residual_in_fp32
        self._layer_norm_fn: LayerNormFn | None = (
            layer_norm_fn if fused_add_norm else None
        )
        if fused_add_norm and self._layer_norm_fn is None:
            raise RuntimeError(
                "fused_add_norm requested but mamba_ssm fused layer norm kernels are not available"
            )
        self.fused_add_norm = self._layer_norm_fn is not None
        if self.fused_add_norm:
            allowed_norms: tuple[type[nn.Module], ...]
            if RMSNorm is not None:
                allowed_norms = (nn.LayerNorm, RMSNorm)
            else:
                allowed_norms = (nn.LayerNorm,)
            if not isinstance(self.norm, allowed_norms):
                raise TypeError(
                    "fused_add_norm only supports LayerNorm or RMSNorm normalizers"
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        inference_params=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply fused add+norm, mixer, and optional MLP.

        Args:
            hidden_states: Input tensor `(B, L, D)`.
            residual: Optional residual tensor to add prior to norm.
            inference_params: Optional inference cache.
            **kwargs: Passed to mixer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(hidden_states, residual)`.

        Raises:
            TypeError: If a fused residual is requested with an unsupported norm.
        """

        residual_input = residual

        if not self.fused_add_norm:
            residual = (
                hidden_states
                if residual_input is None
                else hidden_states + residual_input
            )
            norm_dtype = _norm_weight_dtype(self.norm, residual.dtype)
            hidden_states = self.norm(residual.to(dtype=norm_dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            if self._layer_norm_fn is None:
                raise RuntimeError("fused add+norm backend not available at runtime")
            hidden_states, residual = self._layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual_input,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=_is_rms_norm(self.norm),
            )

        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params, **kwargs
        )

        if isinstance(self.mlp, nn.Identity):
            return hidden_states, residual

        assert self.norm2 is not None
        if not self.fused_add_norm:
            residual = hidden_states + residual
            norm2_dtype = _norm_weight_dtype(self.norm2, residual.dtype)
            hidden_states = self.norm2(residual.to(dtype=norm2_dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            if self._layer_norm_fn is None:
                raise RuntimeError("fused add+norm backend not available at runtime")
            hidden_states, residual = self._layer_norm_fn(
                hidden_states,
                self.norm2.weight,
                self.norm2.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm2.eps,
                is_rms_norm=_is_rms_norm(self.norm2),
            )

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Delegate cache allocation to the mixer.

        Args:
            batch_size: Maximum batch size expected during decoding.
            max_seqlen: Maximum target sequence length.
            dtype: Optional dtype override for cache tensors.
            **kwargs: Passed through to the mixer implementation.

        Returns:
            Tuple[torch.Tensor, ...]: Mixer-specific cache tensors.
        """

        if not hasattr(self.mixer, "allocate_inference_cache"):
            raise AttributeError("Mixer does not provide allocate_inference_cache")
        allocate = getattr(self.mixer, "allocate_inference_cache")
        allocate_callable = cast(Callable[..., Tuple[torch.Tensor, ...]], allocate)
        return allocate_callable(batch_size, max_seqlen, dtype=dtype, **kwargs)
