from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class Block(nn.Module):
    """Wrapper that fuses residual add + (RMS)LayerNorm and applies a mixer then optional MLP.

    This is a design-only stub that validates API contracts. The actual fused
    normalization and the mixer/MLP wiring are provided later.

    Args:
        dim: Model dimension of input and output.
        mixer_cls: Callable that creates the mixer module given `dim`.
        mlp_cls: Callable (or `nn.Identity`) that creates the MLP given `dim`.
        norm_cls: Normalization layer class, e.g. `nn.LayerNorm`.
        fused_add_norm: If True, expects a fused add+norm behavior via backend.
        residual_in_fp32: If True, converts residual tensor to fp32 for stability.

    Raises:
        NotImplementedError: Always, until fused add+norm is provided.
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
        self.mlp = mlp_cls(dim) if mlp_cls is not nn.Identity else nn.Identity()
        self.norm = norm_cls(dim)
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

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
            NotImplementedError: Implementation to be provided later.
        """
        raise NotImplementedError("Block.forward is not implemented in the scaffold.")
