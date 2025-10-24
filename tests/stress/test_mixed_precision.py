from __future__ import annotations

import torch

from ssm.modules import Mamba1, Mamba2


def _make_models(module_cls, **kwargs):
    torch.manual_seed(123)
    model_fp32 = module_cls(**kwargs)
    model_fp32.eval()
    model_bf16 = module_cls(**kwargs)
    model_bf16.load_state_dict(model_fp32.state_dict())
    model_bf16.to(torch.bfloat16)
    model_bf16.eval()
    model_fp32.eval()
    return model_fp32, model_bf16


@torch.no_grad()
def test_mamba1_bfloat16_matches_fp32() -> None:
    model_fp32, model_bf16 = _make_models(
        Mamba1,
        d_model=32,
        d_state=64,
        d_conv=4,
        expand=2,
    )

    hidden = torch.randn(2, 128, 32)
    baseline = model_fp32(hidden)

    hidden_bf16 = hidden.to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model_bf16(hidden_bf16)

    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result.to(torch.float32),
        baseline,
        atol=5e-2,
        rtol=5e-2,
    )


@torch.no_grad()
def test_mamba2_bfloat16_matches_fp32() -> None:
    model_fp32, model_bf16 = _make_models(
        Mamba2,
        d_model=32,
        d_state=128,
        d_conv=4,
        headdim=8,
        expand=2,
        chunk_size=64,
    )

    hidden = torch.randn(2, 160, 32)
    seq_lens = torch.tensor([160, 120], dtype=torch.long)

    baseline = model_fp32._forward_impl(hidden, seq_lens=seq_lens)

    hidden_bf16 = hidden.to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model_bf16._forward_impl(hidden_bf16, seq_lens=seq_lens)

    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result.to(torch.float32),
        baseline,
        atol=7.5e-2,
        rtol=7.5e-2,
    )
