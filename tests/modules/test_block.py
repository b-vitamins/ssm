from __future__ import annotations

import torch
from torch import nn

from ssm.modules.block import Block


class IdentityNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = 0.0

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - simple passthrough
        return x


class IdentityMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.last_kwargs: dict[str, object] | None = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.last_kwargs = kwargs
        return x

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        cache = torch.zeros(
            batch_size, max_seqlen, self.dim, dtype=dtype or torch.float32
        )
        return (cache,)


class RecordingMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return 2 * x


def test_block_adds_residual_and_runs_mixer() -> None:
    block = Block(
        dim=4,
        mixer_cls=IdentityMixer,
        mlp_cls=nn.Identity,
        norm_cls=IdentityNorm,
    )
    hidden = torch.ones(2, 3, 4)
    residual = 2 * torch.ones(2, 3, 4)

    out, new_residual = block(hidden, residual=residual)

    expected = hidden + residual
    assert torch.allclose(out, expected)
    assert torch.allclose(new_residual, expected)


def test_block_runs_mlp_stage() -> None:
    holder: dict[str, RecordingMLP] = {}

    def mlp_factory(dim: int) -> RecordingMLP:
        module = RecordingMLP(dim)
        holder["module"] = module
        return module

    block = Block(
        dim=4,
        mixer_cls=IdentityMixer,
        mlp_cls=mlp_factory,
        norm_cls=IdentityNorm,
    )

    hidden = torch.ones(1, 2, 4)
    residual = 2 * torch.ones(1, 2, 4)
    out, new_residual = block(hidden, residual=residual)

    assert holder["module"].last_input is not None
    torch.testing.assert_close(holder["module"].last_input, 2 * (hidden + residual))
    torch.testing.assert_close(out, 4 * (hidden + residual))
    torch.testing.assert_close(new_residual, 2 * (hidden + residual))


def test_block_residual_precision_toggle() -> None:
    block = Block(
        dim=2,
        mixer_cls=IdentityMixer,
        mlp_cls=nn.Identity,
        norm_cls=IdentityNorm,
        residual_in_fp32=True,
    )
    hidden = torch.ones(1, 1, 2, dtype=torch.float16)
    out, new_residual = block(hidden)

    assert out.dtype == torch.float32
    assert new_residual.dtype == torch.float32


def test_block_delegates_cache_allocation() -> None:
    block = Block(
        dim=4,
        mixer_cls=IdentityMixer,
        mlp_cls=nn.Identity,
        norm_cls=IdentityNorm,
    )
    cache = block.allocate_inference_cache(batch_size=2, max_seqlen=5)[0]
    assert cache.shape == (2, 5, 4)
