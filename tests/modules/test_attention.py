from __future__ import annotations

import copy

import torch

from ssm.modules.attention import MHA


class InferenceParams:
    def __init__(self, batch_size: int, max_seqlen: int) -> None:
        self.batch_size_offset = 0
        self.seqlen_offset = 0
        self.max_seqlen = max_seqlen
        self.lengths_per_sample = None
        self.key_value_memory_dict: dict[
            int, tuple[torch.Tensor, torch.Tensor | None]
        ] = {}


def test_mha_forward_matches_no_cache() -> None:
    torch.manual_seed(0)
    mha = MHA(
        embed_dim=16,
        num_heads=4,
        num_heads_kv=2,
        layer_idx=0,
        mlp_dim=0,
        d_conv=0,
        causal=True,
    )
    mha_baseline = copy.deepcopy(mha)

    batch, prefill, dim = 2, 3, 16
    x_prefill = torch.randn(batch, prefill, dim)
    params = InferenceParams(batch_size=batch, max_seqlen=prefill + 1)

    out_prefill = mha(x_prefill, inference_params=params)

    params.seqlen_offset = prefill
    x_next = torch.randn(batch, 1, dim)
    out_next = mha(x_next, inference_params=params)

    full = torch.cat([x_prefill, x_next], dim=1)
    out_full = mha_baseline(full)

    torch.testing.assert_close(out_prefill, out_full[:, :prefill])
    torch.testing.assert_close(out_next, out_full[:, -1:].contiguous())


def test_mha_allocates_cache_and_depthwise_state() -> None:
    mha = MHA(
        embed_dim=8,
        num_heads=2,
        num_heads_kv=1,
        head_dim=4,
        mlp_dim=0,
        layer_idx=1,
        d_conv=3,
    )
    kv, conv_state = mha.allocate_inference_cache(batch_size=2, max_seqlen=5)
    assert kv.shape == (2, 5, 2, 1, 4)
    assert conv_state is not None
    assert mha.conv1d is not None
    assert conv_state.shape == (2, mha.conv1d.weight.shape[0], 3)


def test_mha_supports_packed_mlp_stream() -> None:
    torch.manual_seed(1)
    mha = MHA(
        embed_dim=12,
        num_heads=3,
        mlp_dim=32,
        num_heads_kv=3,
    )
    x = torch.randn(2, 4, 12)
    out = mha(x)
    assert out.shape == (2, 4, 12)
