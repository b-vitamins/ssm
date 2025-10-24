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


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask logits that fall outside the top-k candidates.

    Args:
        logits: Logits tensor with shape `(..., vocab_size)`.
        top_k: Number of highest probability tokens to keep.

    Returns:
        torch.Tensor: Logits with non top-k entries replaced by `-inf`.
    """

    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    _, top_indices = torch.topk(logits, top_k, dim=-1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(-1, top_indices, False)
    return logits.masked_fill(mask, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Args:
        logits: Logits tensor with shape `(..., vocab_size)`.
        top_p: Cumulative probability threshold.

    Returns:
        torch.Tensor: Filtered logits respecting the nucleus constraint.
    """

    if top_p is None or top_p <= 0 or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p
    # Always keep at least one token.
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = logits.clone()
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Filter logits using a minimum probability threshold.

    Args:
        logits: Logits tensor with shape `(..., vocab_size)`.
        min_p: Minimum probability mass required to keep a token.

    Returns:
        torch.Tensor: Logits where entries with probability `< min_p` are masked.
    """

    if min_p is None or min_p <= 0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    return logits.masked_fill(probs < min_p, float("-inf"))


def sample_from_logits(
    logits: torch.Tensor,
    *,
    top_k: int = 1,
    top_p: float = 0.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample token indices from logits using common decoding strategies.

    Args:
        logits: Logits tensor `(B, vocab_size)`.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        min_p: Minimum probability constraint.
        temperature: Softmax temperature.

    Returns:
        torch.Tensor: Sampled token indices `(B,)`.

    Raises:
        ValueError: If `temperature` is not positive.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive for sampling")

    if top_k <= 1 and top_p <= 0.0 and min_p <= 0.0:
        scaled = logits / temperature
        return torch.argmax(scaled, dim=-1)

    filtered = logits / temperature
    filtered = top_k_filter(filtered, top_k)
    filtered = top_p_filter(filtered, top_p)
    filtered = min_p_filter(filtered, min_p)
    probs = torch.softmax(filtered, dim=-1)
    invalid = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) == 0)
    if invalid.any():
        fallback = torch.softmax((logits / temperature), dim=-1)
        probs[invalid] = fallback[invalid]
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
