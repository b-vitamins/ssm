from __future__ import annotations

from typing import Optional

import torch


def decode(
    input_ids: torch.Tensor,
    model,
    max_length: int,
    top_k: int = 1,
    top_p: float = 0.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    teacher_outputs: torch.Tensor | None = None,
    vocab_size: int | None = None,
    cg: bool = False,
    enable_timing: bool = False,
    output_scores: bool = False,
    streamer: Optional[object] = None,
):
    """Decoding loop (design-only stub).

    Mirrors a standard greedy/sampling decode interface for causal LMs.

    Args:
        input_ids: Prompt `(B, L)`.
        model: Module exposing forward/generate-compatible interface.
        max_length: Target total length.
        top_k: Top-k sampling; 1 for greedy.
        top_p: Nucleus sampling parameter.
        min_p: Min-p sampling parameter.
        temperature: Softmax temperature.
        repetition_penalty: Repetition penalty coefficient.
        eos_token_id: Optional EOS id to stop early.
        teacher_outputs: Optional teacher tokens for deterministic testing.
        vocab_size: Optional vocab truncation for logits.
        cg: If True, use CUDA graph capture when supported.
        enable_timing: If True, collect timing events.
        output_scores: If True, return per-step logits as scores.
        streamer: Optional streaming callback.

    Returns:
        An object with `sequences` and optional `scores` fields.

    Raises:
        NotImplementedError: Implementation to be supplied later.
    """
    raise NotImplementedError("utils.decode is not implemented in the scaffold.")
