import copy

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
