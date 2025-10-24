from __future__ import annotations

import torch
from torch import nn

from .config import MambaConfig

# Language model API stubs for SSM.
# This module defines the `MambaLMHeadModel` interface used by tests and
# downstream code. Implementations are intentionally omitted in this scaffold.


class MambaLMHeadModel(nn.Module):
    """Causal language model built from Mamba blocks.

    Design-only stub exposing the stable API; methods raise NotImplementedError
    until backends and reference ops land.

    Args:
        config: `MambaConfig` instance.
        initializer_cfg: Optional dict controlling parameter init policies.
        device: Optional device for parameters.
        dtype: Optional dtype for parameters.
    """

    def __init__(self, config: MambaConfig, initializer_cfg: dict | None = None, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.initializer_cfg = initializer_cfg or {}
        self.device = device
        self.dtype = dtype

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype | None = None, **kwargs):
        """Allocate decoding cache for the backbone.

        Args:
            batch_size: Max batch size expected during decoding.
            max_seqlen: Max target sequence length for decoding.
            dtype: Optional dtype for cache tensors.
            **kwargs: Reserved for backend-specific knobs.

        Returns:
            Any: Backend-specific cache structure.

        Raises:
            NotImplementedError: Implementation to be supplied later.
        """
        raise NotImplementedError("MambaLMHeadModel.allocate_inference_cache is not implemented in the scaffold.")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        inference_params=None,
        num_last_tokens: int = 0,
        **mixer_kwargs,
    ):
        """Run a forward pass through the language model.

        Args:
            input_ids: Token IDs `(B, L)`.
            position_ids: Optional position IDs `(B, L)`.
            inference_params: Optional cache structure for decoding.
            num_last_tokens: If > 0, only return logits for the last `n` tokens.
            **mixer_kwargs: Passed to mixer blocks (e.g., varlen controls).

        Returns:
            A namedtuple `CausalLMOutput(logits=tensor)`.

        Raises:
            NotImplementedError: Implementation to be supplied later.
        """
        raise NotImplementedError("MambaLMHeadModel.forward is not implemented in the scaffold.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, device=None, dtype=None, **kwargs) -> "MambaLMHeadModel":
        """Construct from a pre-trained checkpoint.

        Args:
            pretrained_model_name: Hugging Face Hub repo ID or local directory.
            device: Optional device mapping for weights.
            dtype: Optional dtype cast for weights.
            **kwargs: Passed down to `load_state_dict` helper.

        Returns:
            MambaLMHeadModel: Constructed model.

        Raises:
            NotImplementedError: Implementation to be supplied later.
        """
        raise NotImplementedError("MambaLMHeadModel.from_pretrained is not implemented in the scaffold.")

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and config to a directory.

        Args:
            save_directory: Target directory to create files in.

        Raises:
            NotImplementedError: Implementation to be supplied later.
        """
        raise NotImplementedError("MambaLMHeadModel.save_pretrained is not implemented in the scaffold.")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        top_k: int = 1,
        top_p: float = 0.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        **kwargs,
    ):
        """Generate tokens via greedy or sampling-based decoding.

        Args:
            input_ids: Prompt token IDs `(B, L)`.
            max_length: Target total length (prompt + generated).
            top_k: Top-k sampling parameter; `1` means greedy.
            top_p: Top-p (nucleus) sampling parameter.
            min_p: Min-p sampling parameter.
            temperature: Softmax temperature.
            return_dict_in_generate: If True, return a generation output object.
            output_scores: If True, include per-step logits in the output.
            **kwargs: Passed to decode utility (e.g., CUDA graphs toggle).

        Returns:
            Either sequences tensor or a dataclass-like object with sequences and scores.

        Raises:
            NotImplementedError: Implementation to be supplied later.
        """
        raise NotImplementedError("MambaLMHeadModel.generate is not implemented in the scaffold.")
