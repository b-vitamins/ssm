import copy
import importlib

import pytest
import torch

from ssm.modules import Mamba1


class InferenceParams:
    def __init__(self, batch_size: int, max_seqlen: int) -> None:
        self.batch_size_offset = 0
        self.seqlen_offset = 0
        self.max_seqlen = max_seqlen
        self.key_value_memory_dict: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def test_mamba1_forward_runs_and_backprops():
    torch.manual_seed(0)
    module = Mamba1(d_model=8, d_state=4, d_conv=3, expand=2, bias=True)
    hidden = torch.randn(2, 5, 8, requires_grad=True)

    output = module(hidden)

    assert output.shape == hidden.shape

    loss = output.sum()
    loss.backward()

    assert module.conv_weight.grad is not None


def test_mamba1_allocate_inference_cache_shapes():
    module = Mamba1(d_model=8, d_state=4, d_conv=3, expand=2)

    conv_state, ssm_state = module.allocate_inference_cache(batch_size=3, max_seqlen=16)

    assert conv_state.shape == (3, module.expand * module.d_model, module.d_conv)
    assert ssm_state.shape == (3, module.expand * module.d_model, module.d_state)
    assert conv_state.dtype == module.conv_weight.dtype
    assert ssm_state.dtype == module.conv_weight.dtype


def test_mamba1_cached_step_matches_dense():
    torch.manual_seed(2)
    module = Mamba1(d_model=8, d_state=4, d_conv=3, expand=2, bias=True, layer_idx=0)
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


def test_mamba1_matches_upstream_reference():
    pytest.importorskip("mamba_ssm.modules.mamba_simple")
    upstream_module = importlib.import_module("mamba_ssm.modules.mamba_simple")
    UpstreamMamba = getattr(upstream_module, "Mamba")

    torch.manual_seed(7)
    d_model, d_state, d_conv, expand = 8, 4, 3, 2
    batch, seqlen = 2, 5

    upstream = UpstreamMamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dt_rank="auto",
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init="random",
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=True,
        use_fast_path=False,
    )

    module = Mamba1(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dt_rank=upstream.dt_rank,
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init_floor=1e-4,
        dt_init="random",
        conv_bias=True,
        bias=True,
    )

    module.in_proj.weight.data.copy_(upstream.in_proj.weight.data)
    if module.in_proj.bias is not None and upstream.in_proj.bias is not None:
        module.in_proj.bias.data.copy_(upstream.in_proj.bias.data)
    module.out_proj.weight.data.copy_(upstream.out_proj.weight.data)
    if module.out_proj.bias is not None and upstream.out_proj.bias is not None:
        module.out_proj.bias.data.copy_(upstream.out_proj.bias.data)
    module.conv_weight.data.copy_(
        upstream.conv1d.weight.data.view(module.inner_dim, -1)
    )
    if module.conv_bias is not None and upstream.conv1d.bias is not None:
        module.conv_bias.data.copy_(upstream.conv1d.bias.data)
    module.x_proj.weight.data.copy_(upstream.x_proj.weight.data)
    module.dt_proj.weight.data.copy_(upstream.dt_proj.weight.data)
    module.dt_bias.data.copy_(upstream.dt_proj.bias.data)
    module.A_log.data.copy_(upstream.A_log.data)
    module.D.data.copy_(upstream.D.data)

    hidden = torch.randn(batch, seqlen, d_model)
    ours = module(hidden)
    theirs = upstream(hidden)
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-4)

    ours_conv, ours_ssm = module.allocate_inference_cache(batch, seqlen)
    theirs_conv, theirs_ssm = upstream.allocate_inference_cache(batch, seqlen)
    for t in range(seqlen):
        token = hidden[:, t : t + 1]
        ours_out, ours_conv, ours_ssm = module.step(token, ours_conv, ours_ssm)
        theirs_out, theirs_conv, theirs_ssm = upstream.step(
            token, theirs_conv, theirs_ssm
        )
        torch.testing.assert_close(ours_out, theirs_out, atol=1e-5, rtol=1e-4)
