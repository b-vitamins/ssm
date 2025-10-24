import copy

import torch

from ssm.modules import Mamba2


class InferenceParams:
    def __init__(self, batch_size: int, max_seqlen: int) -> None:
        self.batch_size_offset = 0
        self.seqlen_offset = 0
        self.max_seqlen = max_seqlen
        self.key_value_memory_dict: dict[int, dict[str, torch.Tensor]] = {}


def test_mamba2_forward_shapes():
    torch.manual_seed(0)
    module = Mamba2(d_model=8, headdim=4, expand=2, d_ssm=8, bias=True)
    hidden = torch.randn(2, 6, 8, requires_grad=True)

    output = module(hidden)

    assert output.shape == hidden.shape

    output.sum().backward()
    assert module.conv_weight.grad is not None


def test_mamba2_varlen_matches_dense():
    torch.manual_seed(1)
    module = Mamba2(d_model=8, headdim=4, expand=2, d_ssm=8, bias=True)

    lengths = torch.tensor([4, 2], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.long)
    padded = torch.randn(2, 4, 8)

    dense_out = module(padded, cu_seqlens=cu_seqlens)

    flat_tokens = torch.cat([padded[0, : lengths[0]], padded[1, : lengths[1]]], dim=0)

    # Permute tokens to simulate routing during chunked execution.
    permutation = torch.tensor([0, 4, 1, 5, 2, 3], dtype=torch.long)
    routed_tokens = flat_tokens.index_select(0, permutation).unsqueeze(0)
    seq_idx = torch.tensor([[0, 1, 0, 1, 0, 0]], dtype=torch.int32)

    flat_out = module(routed_tokens, cu_seqlens=cu_seqlens, seq_idx=seq_idx)

    packed = torch.cat([dense_out[0, : lengths[0]], dense_out[1, : lengths[1]]], dim=0)
    expected = packed.index_select(0, permutation)

    assert torch.allclose(flat_out.squeeze(0), expected, atol=1e-5, rtol=1e-4)


def test_mamba2_allocate_inference_cache_shapes():
    module = Mamba2(d_model=8, headdim=4, expand=2, d_ssm=8)

    cache = module.allocate_inference_cache(batch_size=3, max_seqlen=16)

    assert set(cache) == {"conv_state", "ssd_state"}
    assert cache["conv_state"].shape == (
        3,
        module.expand * module.d_model,
        module.d_conv,
    )
    assert cache["ssd_state"].shape == (3, module.ssm_heads, module.headdim)
    assert cache["conv_state"].dtype == module.conv_weight.dtype
    assert cache["ssd_state"].dtype == module.conv_weight.dtype


def test_mamba2_cached_dense_matches():
    torch.manual_seed(3)
    module = Mamba2(d_model=8, headdim=4, expand=2, d_ssm=8, bias=True, layer_idx=0)
    reference = copy.deepcopy(module)

    batch, prefill, decode = 2, 3, 2
    tokens = torch.randn(batch, prefill + decode, 8)
    params = InferenceParams(batch_size=batch, max_seqlen=prefill + decode)

    dense_out = reference(tokens)

    prefill_out = module(tokens[:, :prefill], inference_params=params)
    torch.testing.assert_close(prefill_out, dense_out[:, :prefill])

    params.seqlen_offset = prefill
    decode_out = module(tokens[:, prefill:], inference_params=params)
    torch.testing.assert_close(decode_out, dense_out[:, prefill:])


def test_mamba2_cached_varlen_matches():
    torch.manual_seed(4)
    module = Mamba2(d_model=8, headdim=4, expand=2, d_ssm=8, bias=True, layer_idx=1)
    reference = copy.deepcopy(module)

    lengths = torch.tensor([4, 2], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.long)
    padded = torch.randn(2, 4, 8)

    dense_varlen = reference(padded, cu_seqlens=cu_seqlens)

    params = InferenceParams(batch_size=2, max_seqlen=padded.shape[1])
    cached_dense = module(padded, cu_seqlens=cu_seqlens, inference_params=params)
    for b, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(cached_dense[b, :length], dense_varlen[b, :length])

    flat_tokens = torch.cat([padded[0, : lengths[0]], padded[1, : lengths[1]]], dim=0)
    permutation = torch.tensor([0, 4, 1, 5, 2, 3], dtype=torch.long)
    routed_tokens = flat_tokens.index_select(0, permutation).unsqueeze(0)
    seq_idx = torch.tensor([[0, 1, 0, 1, 0, 0]], dtype=torch.int32)

    reference_flat = reference(routed_tokens, cu_seqlens=cu_seqlens, seq_idx=seq_idx)
    params = InferenceParams(batch_size=2, max_seqlen=padded.shape[1])
    cached_flat = module(
        routed_tokens,
        cu_seqlens=cu_seqlens,
        seq_idx=seq_idx,
        inference_params=params,
    )
    torch.testing.assert_close(cached_flat, reference_flat)
