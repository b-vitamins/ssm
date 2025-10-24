import torch

from ssm.modules import Mamba1


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
