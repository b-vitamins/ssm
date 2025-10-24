from __future__ import annotations

import torch

from ssm.modules.mlp import GatedMLP


def test_gated_mlp_applies_gating() -> None:
    mlp = GatedMLP(
        in_features=4,
        hidden_features=4,
        out_features=4,
        activation=lambda x: x,
        bias=False,
        multiple_of=1,
    )
    with torch.no_grad():
        mlp.fc1.weight.zero_()
        mlp.fc1.weight[:4].copy_(torch.eye(4))
        mlp.fc1.weight[4:].copy_(torch.eye(4))
        mlp.fc2.weight.copy_(torch.eye(4))
        if mlp.fc2.bias is not None:
            mlp.fc2.bias.zero_()

    x = torch.randn(2, 3, 4)
    out = mlp(x)
    torch.testing.assert_close(out, x * x)


def test_gated_mlp_shape_and_dtype() -> None:
    mlp = GatedMLP(in_features=6, hidden_features=None, out_features=6)
    x = torch.randn(1, 5, 6, dtype=torch.float16)
    out = mlp(x)
    assert out.shape == (1, 5, 6)
    assert out.dtype == torch.float16
