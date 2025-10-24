from __future__ import annotations

from torch import nn
from torch.nn import functional as F


class GatedMLP(nn.Module):
    """Gated MLP block matching the reference Mamba implementation.

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden feature dimension; if ``None``, defaults to ``8/3 * in_features``.
        out_features: Output feature dimension; defaults to ``in_features``.
        activation: Callable activation function for gating.
        bias: Whether to use bias in linear layers.
        multiple_of: Round hidden size up to a multiple of this value.
        device: Optional device for parameters.
        dtype: Optional dtype for parameters.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation=None,
        bias: bool = False,
        multiple_of: int = 128,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (
            (hidden_features + multiple_of - 1) // multiple_of * multiple_of
            if multiple_of > 0
            else hidden_features
        )
        self.fc1 = nn.Linear(
            in_features,
            2 * hidden_features,
            bias=bias,
            **factory_kwargs,
        )
        self.activation = activation if activation is not None else F.silu
        self.fc2 = nn.Linear(
            hidden_features,
            out_features,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, x):
        """Apply the MLP.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Tensor with shape ``(..., out_features)``.
        """

        input_dtype = x.dtype
        proj_dtype = self.fc1.weight.dtype
        projected = self.fc1(x.to(dtype=proj_dtype))
        hidden, gate = projected.chunk(2, dim=-1)
        activated = self.activation(gate)
        hidden = hidden * activated
        output = self.fc2(hidden)
        return output.to(dtype=input_dtype)
