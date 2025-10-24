from __future__ import annotations

from torch import nn


class GatedMLP(nn.Module):
    """Gated MLP block.

    Design-only stub. Implements the interface and contract only.

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden feature dimension; if None, defaults to ~8/3 * in_features.
        out_features: Output feature dimension; defaults to `in_features`.
        activation: Callable activation function for gating (documented only).
        bias: Whether to use bias in linear layers.
        multiple_of: Round hidden size up to a multiple of this value.
        device: Optional device for parameters.
        dtype: Optional dtype for parameters.

    Raises:
        NotImplementedError: Always, until implemented.
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
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.activation = activation
        self.bias = bias
        self.multiple_of = multiple_of

    def forward(self, x):
        """Apply the MLP.

        Args:
            x: Input tensor of shape `(..., in_features)`.

        Returns:
            Tensor with shape `(..., out_features)`.

        Raises:
            NotImplementedError: Implementation to be provided later.
        """
        raise NotImplementedError("GatedMLP.forward is not implemented in the scaffold.")
