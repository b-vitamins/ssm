from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import torch
from torch import nn

from .config import MambaConfig
from ..utils import generation as generation_utils

# Language model API stubs for SSM.
# This module defines the `MambaLMHeadModel` interface used by tests and
# downstream code. Implementations are intentionally omitted in this scaffold.


# Simple output container mirroring Hugging Face style semantics.
CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])


@dataclass
class GenerationOutput:
    """Structure returned by :meth:`MambaLMHeadModel.generate` when requested.

    Attributes:
        sequences: Tensor holding generated sequences.
        scores: Optional list of per-step logits tensors.
    """

    sequences: torch.Tensor
    scores: list[torch.Tensor] | None = None


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

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg: dict | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.initializer_cfg = initializer_cfg or {}
        self.device = device
        self.dtype = dtype

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.backbone = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layer,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        """Allocate decoding cache for the backbone.

        Args:
            batch_size: Max batch size expected during decoding.
            max_seqlen: Max target sequence length for decoding.
            dtype: Optional dtype for cache tensors.
            **kwargs: Reserved for backend-specific knobs.

        Returns:
            Any: Backend-specific cache structure.
        """
        hidden_dtype = dtype or next(self.parameters()).dtype
        device = next(self.parameters()).device
        state = torch.zeros(
            self.config.n_layer,
            batch_size,
            self.config.d_model,
            dtype=hidden_dtype,
            device=device,
        )
        return {"state": state, "max_seqlen": max_seqlen, "position": 0}

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
            ValueError: If inputs do not match the expected rank or cache batch size.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be a 2D tensor of shape (batch, seqlen)")

        embeddings = self.embed_tokens(input_ids)
        cache_state = None
        if inference_params is not None:
            cache_state = inference_params.get("state")
            if cache_state is not None and cache_state.size(1) != input_ids.size(0):
                raise ValueError("Cached state batch size does not match input batch size")
        output, new_state = self.backbone(embeddings, cache_state)
        if inference_params is not None:
            inference_params["state"] = new_state.detach()
            inference_params["position"] = inference_params.get("position", 0) + input_ids.size(1)

        hidden_states = self.norm(output)
        logits = self.lm_head(hidden_states)
        if num_last_tokens > 0:
            logits = logits[:, -num_last_tokens:, :]
        return CausalLMOutput(logits=logits)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name: str, device=None, dtype=None, **kwargs
    ) -> "MambaLMHeadModel":
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
        raise NotImplementedError(
            "MambaLMHeadModel.from_pretrained is not implemented in the scaffold."
        )

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and config to a directory.

        Args:
            save_directory: Target directory to create files in.

        Raises:
            NotImplementedError: Implementation to be supplied later.
        """
        raise NotImplementedError(
            "MambaLMHeadModel.save_pretrained is not implemented in the scaffold."
        )

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
            ValueError: If arguments violate expected constraints.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seqlen)")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        batch_size, prompt_length = input_ids.shape
        if max_length <= prompt_length:
            sequences = input_ids[:, :max_length].clone()
            output = GenerationOutput(sequences=sequences, scores=[] if output_scores else None)
            if return_dict_in_generate:
                return output
            if output_scores:
                return sequences, output.scores
            return sequences

        steps = max_length - prompt_length
        scores: list[torch.Tensor] | None = [] if output_scores else None
        sequences = input_ids.clone()
        eos_token_id = kwargs.get("eos_token_id")
        cache = self.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_length,
            dtype=next(self.parameters()).dtype,
        )
        was_training = self.training
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        try:
            self.eval()
            with torch.no_grad():
                outputs = self.forward(
                    sequences,
                    inference_params=cache,
                    num_last_tokens=1,
                )
                logits = outputs.logits[:, -1, :]
                for _ in range(steps):
                    if scores is not None:
                        scores.append(logits.clone())
                    next_tokens = generation_utils.sample_from_logits(
                        logits,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        temperature=temperature,
                    )
                    if eos_token_id is not None:
                        next_tokens = next_tokens.masked_fill(finished, eos_token_id)
                    sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)
                    if sequences.size(1) >= max_length:
                        if eos_token_id is not None:
                            finished = finished | (next_tokens == eos_token_id)
                        break
                    if eos_token_id is not None:
                        finished = finished | (next_tokens == eos_token_id)
                        if finished.all():
                            break
                    outputs = self.forward(
                        next_tokens.unsqueeze(-1),
                        inference_params=cache,
                        num_last_tokens=1,
                    )
                    logits = outputs.logits[:, -1, :]
        finally:
            if was_training:
                self.train()

        generation_output = GenerationOutput(sequences=sequences, scores=scores)
        if return_dict_in_generate:
            return generation_output
        if output_scores:
            return sequences, scores
        return sequences
